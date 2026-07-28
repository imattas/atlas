//! OpenCL launch adapter.

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{cl_uint, cl_ulong, CL_BLOCKING};
use std::fs;
use std::path::{Path, PathBuf};
use std::ptr;

/// Parsed OpenCL launch protocol arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchArgs {
    /// Generated OpenCL source artifact.
    pub artifact: String,
    /// Inclusive search start.
    pub start: u64,
    /// Exclusive search end.
    pub end: u64,
    /// Maximum number of retained matches.
    pub max_matches: usize,
    /// Global OpenCL work size.
    pub global_size: usize,
    /// Local OpenCL workgroup size.
    pub local_size: usize,
}

/// Adapter CLI command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterCommand {
    /// Report runtime/device features understood by this adapter.
    Features,
    /// Build-check an OpenCL source file without launching a search.
    CompileCheck {
        /// Generated OpenCL source path.
        source: String,
        /// Optional checked source artifact output path.
        output: Option<String>,
    },
    /// Launch an OpenCL search.
    Launch(LaunchArgs),
}

impl AdapterCommand {
    /// Parses an adapter CLI command.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed commands.
    pub fn parse(args: &[String]) -> Result<Self, String> {
        if args.first().is_some_and(|arg| arg == "--features") {
            return Ok(Self::Features);
        }
        if args.first().is_some_and(|arg| arg == "--compile-check") {
            let Some(source) = args.get(1) else {
                return Err("missing compile-check source".to_owned());
            };
            return Ok(Self::CompileCheck {
                source: source.clone(),
                output: optional_output_path(args)?,
            });
        }
        LaunchArgs::parse(args).map(Self::Launch)
    }
}

impl LaunchArgs {
    /// Parses launch protocol arguments.
    ///
    /// # Errors
    ///
    /// Returns an error when required arguments are missing, malformed, or
    /// describe an empty or under-covered launch range.
    pub fn parse(args: &[String]) -> Result<Self, String> {
        let Some(artifact) = args.first() else {
            return Err("missing kernel artifact".to_owned());
        };
        let start = parse_u64_flag(args, "--start")?;
        let end = parse_u64_flag(args, "--end")?;
        if end <= start {
            return Err("end must be greater than start".to_owned());
        }
        let max_matches = parse_usize_flag(args, "--max-matches")?;
        if max_matches == 0 {
            return Err("max-matches must be nonzero".to_owned());
        }
        if u32::try_from(max_matches).is_err() {
            return Err("max-matches exceeds OpenCL uint".to_owned());
        }
        let global_size = parse_usize_flag(args, "--global-size")?;
        let local_size = parse_usize_flag(args, "--local-size")?;
        if global_size == 0 || local_size == 0 {
            return Err("global-size and local-size must be nonzero".to_owned());
        }
        if global_size % local_size != 0 {
            return Err("global-size must be a multiple of local-size".to_owned());
        }
        if u64::try_from(global_size).unwrap_or(u64::MAX) < end.saturating_sub(start) {
            return Err("global-size must cover launch domain".to_owned());
        }
        Ok(Self {
            artifact: artifact.clone(),
            start,
            end,
            max_matches,
            global_size,
            local_size,
        })
    }
}

/// Launches one parsed OpenCL request.
pub trait Launcher {
    /// Reports runtime/device features available to generated kernels.
    ///
    /// # Errors
    ///
    /// Returns an error when no OpenCL runtime/device can be selected.
    fn features(&self) -> Result<Vec<String>, String>;

    /// Build-checks generated OpenCL source.
    ///
    /// # Errors
    ///
    /// Returns an error when source cannot be built for the selected device.
    fn compile_check(&self, source: &str, output: Option<&str>) -> Result<(), String>;

    /// Runs the launch and returns device-reported matches.
    ///
    /// # Errors
    ///
    /// Returns an error when the device launch fails.
    fn launch(&self, args: &LaunchArgs) -> Result<Vec<u64>, String>;
}

/// OpenCL device-backed launcher.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenClLauncher;

impl Launcher for OpenClLauncher {
    fn features(&self) -> Result<Vec<String>, String> {
        let _device = Device::new(select_device()?);
        Ok(vec!["int64".to_owned()])
    }

    fn compile_check(&self, source: &str, output: Option<&str>) -> Result<(), String> {
        let source = fs::read_to_string(source).map_err(|error| error.to_string())?;
        let (_context, _queue, _program) = build_opencl_program(&source)?;
        if let Some(output) = output {
            write_opencl_source(output, &source)?;
        }
        Ok(())
    }

    fn launch(&self, args: &LaunchArgs) -> Result<Vec<u64>, String> {
        launch_opencl(args)
    }
}

/// Runs adapter CLI logic with an injected launcher.
///
/// # Errors
///
/// Returns parse or launcher errors.
pub fn run_cli(args: &[String], launcher: &dyn Launcher) -> Result<String, String> {
    match AdapterCommand::parse(args)? {
        AdapterCommand::Features => {
            let features = launcher.features()?;
            Ok(format_features(&features))
        }
        AdapterCommand::CompileCheck { source, output } => {
            launcher.compile_check(&source, output.as_deref())?;
            Ok(String::new())
        }
        AdapterCommand::Launch(launch_args) => {
            let matches = launcher.launch(&launch_args)?;
            Ok(format_matches(&matches))
        }
    }
}

fn format_features(features: &[String]) -> String {
    features
        .iter()
        .map(|feature| format!("feature={feature}\n"))
        .collect()
}

fn format_matches(matches: &[u64]) -> String {
    matches
        .iter()
        .map(|candidate| format!("match={candidate}\n"))
        .collect()
}

fn parse_u64_flag(args: &[String], flag: &str) -> Result<u64, String> {
    let value = flag_value(args, flag)?;
    value
        .parse()
        .map_err(|_| format!("{flag} must be an unsigned integer"))
}

fn parse_usize_flag(args: &[String], flag: &str) -> Result<usize, String> {
    let value = flag_value(args, flag)?;
    value
        .parse()
        .map_err(|_| format!("{flag} must be an unsigned integer"))
}

fn flag_value<'a>(args: &'a [String], flag: &str) -> Result<&'a str, String> {
    args.windows(2)
        .find_map(|window| (window[0] == flag).then_some(window[1].as_str()))
        .ok_or_else(|| format!("missing {flag}"))
}

fn optional_output_path(args: &[String]) -> Result<Option<String>, String> {
    let Some(index) = args.iter().position(|arg| arg == "-o" || arg == "--output") else {
        return Ok(None);
    };
    args.get(index + 1)
        .cloned()
        .map(Some)
        .ok_or_else(|| "missing output path after -o".to_owned())
}

fn write_opencl_source(path: &str, source: &str) -> Result<(), String> {
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create OpenCL output directory: {error}"))?;
    }
    fs::write(path, source).map_err(|error| format!("cannot write OpenCL artifact {path}: {error}"))
}

fn launch_opencl(args: &LaunchArgs) -> Result<Vec<u64>, String> {
    let source = fs::read_to_string(&args.artifact).map_err(|error| error.to_string())?;
    let (context, queue, program) = build_opencl_program(&source)?;
    let out_buffer = unsafe {
        Buffer::<cl_ulong>::create(
            &context,
            CL_MEM_READ_WRITE,
            args.max_matches,
            ptr::null_mut(),
        )
        .map_err(|error| error.to_string())?
    };
    let mut out_len_buffer = unsafe {
        Buffer::<cl_uint>::create(&context, CL_MEM_READ_WRITE, 1, ptr::null_mut())
            .map_err(|error| error.to_string())?
    };
    let mut zero = vec![0_u32; 1];
    unsafe {
        queue
            .enqueue_write_buffer(&mut out_len_buffer, CL_BLOCKING, 0, &zero, &[])
            .map_err(|error| error.to_string())?;
    }
    let max_matches = cl_uint::try_from(args.max_matches)
        .map_err(|_| "max-matches exceeds OpenCL uint".to_owned())?;
    let start = args.start as cl_ulong;
    let end = args.end as cl_ulong;
    let kernel = Kernel::create(&program, "atlas_search").map_err(|error| error.to_string())?;
    unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&start)
            .set_arg(&end)
            .set_arg(&out_buffer)
            .set_arg(&out_len_buffer)
            .set_arg(&max_matches)
            .set_global_work_size(args.global_size)
            .set_local_work_size(args.local_size)
            .enqueue_nd_range(&queue)
            .map_err(|error| error.to_string())?;
    }
    unsafe {
        queue
            .enqueue_read_buffer(&out_len_buffer, CL_BLOCKING, 0, &mut zero, &[])
            .map_err(|error| error.to_string())?;
    }
    let retained = usize::try_from(zero[0])
        .unwrap_or(args.max_matches)
        .min(args.max_matches);
    let mut matches = vec![0_u64; args.max_matches];
    unsafe {
        queue
            .enqueue_read_buffer(&out_buffer, CL_BLOCKING, 0, &mut matches, &[])
            .map_err(|error| error.to_string())?;
    }
    matches.truncate(retained);
    matches.sort_unstable();
    Ok(matches)
}

fn build_opencl_program(source: &str) -> Result<(Context, CommandQueue, Program), String> {
    let device_id = select_device()?;
    let device = Device::new(device_id);
    let context = Context::from_device(&device).map_err(|error| error.to_string())?;
    let queue = CommandQueue::create_default(&context, 0).map_err(|error| error.to_string())?;
    let program = Program::create_and_build_from_source(&context, source, "")?;
    Ok((context, queue, program))
}

fn select_device() -> Result<opencl3::types::cl_device_id, String> {
    configure_opencl_dylib_path_from_sdk_roots();
    get_all_devices(CL_DEVICE_TYPE_GPU)
        .or_else(|_| get_all_devices(CL_DEVICE_TYPE_ALL))
        .map_err(|error| error.to_string())?
        .into_iter()
        .next()
        .ok_or_else(|| "no OpenCL device found".to_owned())
}

/// Returns OpenCL loader library candidates under explicit SDK root paths.
///
/// The `cl3` dynamic loader honors `OPENCL_DYLIB_PATH`; these candidates let
/// the adapter configure that path from common SDK roots before the first
/// OpenCL API call.
pub fn opencl_loader_candidates_from_roots(
    roots: impl IntoIterator<Item = PathBuf>,
) -> Vec<PathBuf> {
    roots
        .into_iter()
        .flat_map(|root| {
            opencl_loader_dirs(&root).into_iter().flat_map(|dir| {
                opencl_loader_names()
                    .into_iter()
                    .map(move |name| dir.join(name))
            })
        })
        .collect()
}

fn configure_opencl_dylib_path_from_sdk_roots() {
    if std::env::var_os("OPENCL_DYLIB_PATH").is_some() {
        return;
    }
    let candidates = opencl_loader_candidates_from_host_roots()
        .into_iter()
        .filter(|candidate| candidate.is_file())
        .map(|candidate| candidate.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    if !candidates.is_empty() {
        std::env::set_var("OPENCL_DYLIB_PATH", candidates.join(";"));
    }
}

/// Returns OpenCL loader library candidates from host environment and standard
/// SDK install roots.
#[must_use]
pub fn opencl_loader_candidates_from_host_roots() -> Vec<PathBuf> {
    opencl_loader_candidates_from_roots(opencl_loader_roots_from_host())
}

fn opencl_loader_roots_from_host() -> Vec<PathBuf> {
    let mut roots = opencl_loader_roots_from_env();
    roots.extend(opencl_standard_loader_roots());
    dedup_paths(roots)
}

fn opencl_loader_roots_from_env() -> Vec<PathBuf> {
    [
        "OPENCL_SDK",
        "OCL_ROOT",
        "INTELOCLSDKROOT",
        "AMDAPPSDKROOT",
        "ONEAPI_ROOT",
    ]
    .into_iter()
    .filter_map(std::env::var_os)
    .flat_map(|value| std::env::split_paths(&value).collect::<Vec<_>>())
    .collect()
}

fn opencl_standard_loader_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    #[cfg(windows)]
    {
        for base in ["ProgramFiles", "ProgramFiles(x86)"]
            .into_iter()
            .filter_map(std::env::var_os)
            .map(PathBuf::from)
        {
            push_existing_dir(&mut roots, base.join("Khronos").join("OpenCL-SDK"));
            push_existing_dir(
                &mut roots,
                base.join("Intel")
                    .join("oneAPI")
                    .join("compiler")
                    .join("latest"),
            );
        }
    }
    #[cfg(not(windows))]
    {
        roots.push(PathBuf::from("/usr"));
        roots.push(PathBuf::from("/usr/local"));
    }
    roots
}

fn push_existing_dir(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if path.is_dir() {
        paths.push(path);
    }
}

fn dedup_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut deduped = Vec::new();
    for path in paths {
        if !deduped.iter().any(|existing| existing == &path) {
            deduped.push(path);
        }
    }
    deduped
}

fn opencl_loader_dirs(root: &Path) -> Vec<PathBuf> {
    vec![
        root.join("bin"),
        root.join("Bin"),
        root.join("lib"),
        root.join("lib64"),
        root.join("lib").join("x64"),
        root.join("compiler").join("latest").join("bin"),
        root.join("compiler")
            .join("latest")
            .join("windows")
            .join("bin"),
    ]
}

fn opencl_loader_names() -> Vec<&'static str> {
    if cfg!(target_os = "windows") {
        vec!["OpenCL.dll"]
    } else if cfg!(target_os = "macos") {
        vec!["OpenCL", "libOpenCL.dylib"]
    } else {
        vec!["libOpenCL.so.1", "libOpenCL.so"]
    }
}
