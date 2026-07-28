//! OpenCL launch adapter.

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{cl_uint, cl_ulong, CL_BLOCKING};
use std::fs;
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
    /// Build-check an OpenCL source file without launching a search.
    CompileCheck {
        /// Generated OpenCL source path.
        source: String,
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
        if args.first().is_some_and(|arg| arg == "--compile-check") {
            let Some(source) = args.get(1) else {
                return Err("missing compile-check source".to_owned());
            };
            return Ok(Self::CompileCheck {
                source: source.clone(),
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
    /// describe an empty range.
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
        let global_size = parse_usize_flag(args, "--global-size")?;
        let local_size = parse_usize_flag(args, "--local-size")?;
        if global_size == 0 || local_size == 0 {
            return Err("global-size and local-size must be nonzero".to_owned());
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
    /// Build-checks generated OpenCL source.
    ///
    /// # Errors
    ///
    /// Returns an error when source cannot be built for the selected device.
    fn compile_check(&self, source: &str) -> Result<(), String>;

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
    fn compile_check(&self, source: &str) -> Result<(), String> {
        let source = fs::read_to_string(source).map_err(|error| error.to_string())?;
        let (_context, _queue, _program) = build_opencl_program(&source)?;
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
        AdapterCommand::CompileCheck { source } => {
            launcher.compile_check(&source)?;
            Ok(String::new())
        }
        AdapterCommand::Launch(launch_args) => {
            let matches = launcher.launch(&launch_args)?;
            Ok(format_matches(&matches))
        }
    }
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
    get_all_devices(CL_DEVICE_TYPE_GPU)
        .or_else(|_| get_all_devices(CL_DEVICE_TYPE_ALL))
        .map_err(|error| error.to_string())?
        .into_iter()
        .next()
        .ok_or_else(|| "no OpenCL device found".to_owned())
}
