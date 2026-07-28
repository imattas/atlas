//! HIP launch adapter.

use std::ffi::{c_char, c_int, c_uint, c_void, CString};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr;

const HIP_MAX_THREADS_PER_BLOCK: usize = 1024;

type HipModuleHandle = *mut c_void;
type HipFunction = *mut c_void;
type HipStream = *mut c_void;
type HipResult = c_int;

const HIP_SUCCESS: HipResult = 0;
const HIP_MEMCPY_DEVICE_TO_HOST: c_uint = 2;

/// Parsed HIP launch protocol arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchArgs {
    /// Generated HIP code object artifact.
    pub artifact: String,
    /// Inclusive search start.
    pub start: u64,
    /// Exclusive search end.
    pub end: u64,
    /// Maximum number of retained matches.
    pub max_matches: usize,
    /// Global HIP work size.
    pub global_size: usize,
    /// HIP block size.
    pub local_size: usize,
    /// Optional explicit host/kernel launch ABI.
    pub launch_abi: Option<HipLaunchAbi>,
}

/// Host/kernel launch ABI used by generated HIP kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HipLaunchAbi {
    /// Split 64-bit launch bounds and match output into 32-bit words.
    U32,
    /// Use native 64-bit HIP kernel parameters and match output.
    U64,
}

/// Device launch output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchOutput {
    /// Retained device-reported matches.
    pub matches: Vec<u64>,
    /// Total match count reported by the device-side atomic counter.
    pub match_count: usize,
}

impl LaunchOutput {
    fn new(matches: Vec<u64>, match_count: usize) -> Self {
        Self {
            matches,
            match_count,
        }
    }
}

/// Adapter CLI command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterCommand {
    /// Report runtime/device features understood by this adapter.
    Features,
    /// Module-load-check a HIP code object without launching a search.
    CompileCheck {
        /// Generated HIP source or code object path.
        input: String,
        /// Optional compiled HIP code object output path.
        output: Option<String>,
    },
    /// Launch a HIP search.
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
            let Some(artifact) = args.get(1) else {
                return Err("missing compile-check HIP artifact".to_owned());
            };
            return Ok(Self::CompileCheck {
                input: artifact.clone(),
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
            return Err("max-matches exceeds HIP uint".to_owned());
        }
        let global_size = parse_usize_flag(args, "--global-size")?;
        let local_size = parse_usize_flag(args, "--local-size")?;
        if global_size == 0 || local_size == 0 {
            return Err("global-size and local-size must be nonzero".to_owned());
        }
        if local_size > HIP_MAX_THREADS_PER_BLOCK {
            return Err("local-size exceeds HIP block limit".to_owned());
        }
        if global_size.div_ceil(local_size) > u32::MAX as usize {
            return Err("grid size exceeds HIP uint".to_owned());
        }
        if u64::try_from(global_size).unwrap_or(u64::MAX) < end.saturating_sub(start) {
            return Err("global-size must cover launch domain".to_owned());
        }
        let launch_abi = optional_flag_value(args, "--abi")?
            .map(parse_launch_abi)
            .transpose()?;
        Ok(Self {
            artifact: artifact.clone(),
            start,
            end,
            max_matches,
            global_size,
            local_size,
            launch_abi,
        })
    }
}

/// Launches one parsed HIP request.
pub trait Launcher {
    /// Reports runtime/device features available to generated kernels.
    ///
    /// # Errors
    ///
    /// Returns an error when HIP runtime loading or device selection fails.
    fn features(&self) -> Result<Vec<String>, String>;

    /// Checks generated HIP code object can provide the expected
    /// `atlas_search` kernel.
    ///
    /// # Errors
    ///
    /// Returns an error when the artifact cannot be read, loaded, or queried.
    fn compile_check(&self, input: &str, output: Option<&str>) -> Result<(), String>;

    /// Runs the launch and returns device-reported matches.
    ///
    /// # Errors
    ///
    /// Returns an error when HIP runtime loading, module loading, kernel launch,
    /// or device transfer fails.
    fn launch(&self, args: &LaunchArgs) -> Result<LaunchOutput, String>;
}

/// HIP runtime backed code-object launcher.
#[derive(Debug, Clone, Copy, Default)]
pub struct HipModuleLauncher;

impl Launcher for HipModuleLauncher {
    fn features(&self) -> Result<Vec<String>, String> {
        let runtime = HipRuntime::load()?;
        runtime.init()?;
        runtime.set_device(0)?;
        Ok(vec!["int64".to_owned()])
    }

    fn compile_check(&self, input: &str, output: Option<&str>) -> Result<(), String> {
        let artifact = read_hip_artifact(input, output)?;
        ensure_artifact_readable(artifact)?;
        if output.is_some() {
            return Ok(());
        }
        if let Ok(runtime) = HipRuntime::load() {
            runtime.init()?;
            runtime.set_device(0)?;
            let module = runtime.load_module(artifact)?;
            runtime.get_function(module.raw, "atlas_search")?;
        }
        Ok(())
    }

    fn launch(&self, args: &LaunchArgs) -> Result<LaunchOutput, String> {
        ensure_artifact_readable(&args.artifact)?;
        let uses_u32_abi = match args.launch_abi {
            Some(HipLaunchAbi::U32) => true,
            Some(HipLaunchAbi::U64) => false,
            None => artifact_path_uses_u32_launch_abi(&args.artifact)?,
        };
        let runtime = HipRuntime::load()?;
        runtime.init()?;
        runtime.set_device(0)?;
        let module = runtime.load_module(&args.artifact)?;
        let function = runtime.get_function(module.raw, "atlas_search")?;
        if uses_u32_abi {
            launch_hip_u32(&runtime, function, args)
        } else {
            launch_hip(&runtime, function, args)
        }
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
        AdapterCommand::CompileCheck { input, output } => {
            launcher.compile_check(&input, output.as_deref())?;
            Ok(String::new())
        }
        AdapterCommand::Launch(launch_args) => {
            let output = launcher.launch(&launch_args)?;
            Ok(format_launch_output(&output))
        }
    }
}

fn format_features(features: &[String]) -> String {
    features
        .iter()
        .map(|feature| format!("feature={feature}\n"))
        .collect()
}

fn format_launch_output(output: &LaunchOutput) -> String {
    let mut text = format!("match_count={}\n", output.match_count);
    text.push_str(&format_matches(&output.matches));
    text
}

fn format_matches(matches: &[u64]) -> String {
    matches
        .iter()
        .map(|candidate| format!("match={candidate}\n"))
        .collect()
}

fn ensure_artifact_readable(path: &str) -> Result<(), String> {
    fs::metadata(path)
        .map(|_| ())
        .map_err(|error| format!("cannot read HIP code object {path}: {error}"))
}

fn artifact_path_uses_u32_launch_abi(path: &str) -> Result<bool, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("cannot read HIP artifact {path}: {error}"))?;
    Ok(artifact_uses_u32_launch_abi(&bytes))
}

fn artifact_uses_u32_launch_abi(bytes: &[u8]) -> bool {
    bytes
        .windows(b"atlas_search_u32_abi".len())
        .any(|window| window == b"atlas_search_u32_abi")
}

fn read_hip_artifact<'a>(input: &'a str, output: Option<&'a str>) -> Result<&'a str, String> {
    let Some(output) = output else {
        return Ok(input);
    };
    compile_hip_source_to_code_object(input, output)?;
    Ok(output)
}

fn compile_hip_source_to_code_object(source: &str, output: &str) -> Result<(), String> {
    if let Some(parent) = Path::new(output).parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create HIP output directory: {error}"))?;
    }
    let hipcc = find_hipcc_command().ok_or_else(|| {
        format!(
            "failed to find hipcc compiler command; searched hipcc candidates: {}",
            format_path_candidates(hipcc_command_candidates_from_roots(
                hip_root_dirs_from_host()
            ))
        )
    })?;
    let mut command = Command::new(hipcc);
    command
        .arg("--genco")
        .arg("-O2")
        .arg("-nogpuinc")
        .arg("-nogpulib");
    if let Some(arch) = detect_hip_arch() {
        command.arg(format!("--offload-arch={arch}"));
    }
    let compile = command
        .arg(source)
        .arg("-o")
        .arg(output)
        .output()
        .map_err(|error| format!("failed to run hipcc: {error}"))?;
    if compile.status.success() && Path::new(output).is_file() {
        return Ok(());
    }
    Err(format!(
        "hipcc failed: stdout={} stderr={}",
        String::from_utf8_lossy(&compile.stdout),
        String::from_utf8_lossy(&compile.stderr)
    ))
}

fn detect_hip_arch() -> Option<String> {
    let output = Command::new("hipInfo").output().ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.lines().find_map(|line| {
        line.split_once("gcnArchName:")
            .map(|(_, arch)| arch.trim().to_owned())
            .filter(|arch| !arch.is_empty())
    })
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

fn optional_flag_value<'a>(args: &'a [String], flag: &str) -> Result<Option<&'a str>, String> {
    let Some(index) = args.iter().position(|arg| arg == flag) else {
        return Ok(None);
    };
    args.get(index + 1)
        .map(String::as_str)
        .map(Some)
        .ok_or_else(|| format!("missing {flag} value"))
}

fn parse_launch_abi(value: &str) -> Result<HipLaunchAbi, String> {
    match value {
        "u32" => Ok(HipLaunchAbi::U32),
        "u64" => Ok(HipLaunchAbi::U64),
        _ => Err(format!("unsupported --abi '{value}'; expected u32 or u64")),
    }
}

fn launch_hip(
    runtime: &HipRuntime,
    function: HipFunction,
    args: &LaunchArgs,
) -> Result<LaunchOutput, String> {
    let out_len = runtime.malloc(std::mem::size_of::<u32>())?;
    let out_bytes = args
        .max_matches
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or_else(|| "output buffer size overflow".to_owned())?;
    let out = runtime.malloc(out_bytes)?;
    let _out_len_guard = DeviceAllocation {
        runtime,
        ptr: out_len,
    };
    let _out_guard = DeviceAllocation { runtime, ptr: out };
    runtime.memset(out_len, 0, std::mem::size_of::<u32>())?;

    let mut start = args.start;
    let mut end = args.end;
    let mut out_param = out;
    let mut out_len_param = out_len;
    let mut max_matches =
        u32::try_from(args.max_matches).map_err(|_| "max-matches exceeds HIP uint".to_owned())?;
    let mut params = [
        (&mut start as *mut u64).cast::<c_void>(),
        (&mut end as *mut u64).cast::<c_void>(),
        (&mut out_param as *mut *mut c_void).cast::<c_void>(),
        (&mut out_len_param as *mut *mut c_void).cast::<c_void>(),
        (&mut max_matches as *mut u32).cast::<c_void>(),
    ];
    let block_x =
        u32::try_from(args.local_size).map_err(|_| "local-size exceeds HIP uint".to_owned())?;
    let grid_x = args.global_size.div_ceil(args.local_size);
    let grid_x = u32::try_from(grid_x).map_err(|_| "grid size exceeds HIP uint".to_owned())?;
    runtime.launch_kernel(function, grid_x, block_x, params.as_mut_ptr())?;
    runtime.synchronize()?;

    let mut retained = 0_u32;
    runtime.memcpy_device_to_host(
        (&mut retained as *mut u32).cast::<c_void>(),
        out_len,
        std::mem::size_of::<u32>(),
    )?;
    let match_count = usize::try_from(retained).unwrap_or(usize::MAX);
    let retained = match_count.min(args.max_matches);
    let mut matches = vec![0_u64; args.max_matches];
    runtime.memcpy_device_to_host(matches.as_mut_ptr().cast::<c_void>(), out, out_bytes)?;
    matches.truncate(retained);
    matches.sort_unstable();
    Ok(LaunchOutput::new(matches, match_count))
}

fn launch_hip_u32(
    runtime: &HipRuntime,
    function: HipFunction,
    args: &LaunchArgs,
) -> Result<LaunchOutput, String> {
    let out_len = runtime.malloc(std::mem::size_of::<u32>())?;
    let out_words_len = args
        .max_matches
        .checked_mul(2)
        .ok_or_else(|| "output word count overflow".to_owned())?;
    let out_bytes = out_words_len
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| "output buffer size overflow".to_owned())?;
    let out_words = runtime.malloc(out_bytes)?;
    let _out_len_guard = DeviceAllocation {
        runtime,
        ptr: out_len,
    };
    let _out_guard = DeviceAllocation {
        runtime,
        ptr: out_words,
    };
    runtime.memset(out_len, 0, std::mem::size_of::<u32>())?;

    let mut start_lo = low_u32(args.start);
    let mut start_hi = high_u32(args.start);
    let mut end_lo = low_u32(args.end);
    let mut end_hi = high_u32(args.end);
    let mut out_param = out_words;
    let mut out_len_param = out_len;
    let mut max_matches =
        u32::try_from(args.max_matches).map_err(|_| "max-matches exceeds HIP uint".to_owned())?;
    let mut params = [
        (&mut start_lo as *mut u32).cast::<c_void>(),
        (&mut start_hi as *mut u32).cast::<c_void>(),
        (&mut end_lo as *mut u32).cast::<c_void>(),
        (&mut end_hi as *mut u32).cast::<c_void>(),
        (&mut out_param as *mut *mut c_void).cast::<c_void>(),
        (&mut out_len_param as *mut *mut c_void).cast::<c_void>(),
        (&mut max_matches as *mut u32).cast::<c_void>(),
    ];
    let block_x =
        u32::try_from(args.local_size).map_err(|_| "local-size exceeds HIP uint".to_owned())?;
    let grid_x = args.global_size.div_ceil(args.local_size);
    let grid_x = u32::try_from(grid_x).map_err(|_| "grid size exceeds HIP uint".to_owned())?;
    runtime.launch_kernel(function, grid_x, block_x, params.as_mut_ptr())?;
    runtime.synchronize()?;

    let mut retained = 0_u32;
    runtime.memcpy_device_to_host(
        (&mut retained as *mut u32).cast::<c_void>(),
        out_len,
        std::mem::size_of::<u32>(),
    )?;
    let match_count = usize::try_from(retained).unwrap_or(usize::MAX);
    let retained = match_count.min(args.max_matches);
    let mut out_words_host = vec![0_u32; out_words_len];
    runtime.memcpy_device_to_host(
        out_words_host.as_mut_ptr().cast::<c_void>(),
        out_words,
        out_bytes,
    )?;
    let mut matches = out_words_host
        .chunks_exact(2)
        .take(retained)
        .map(|words| u64::from(words[0]) | (u64::from(words[1]) << 32))
        .collect::<Vec<_>>();
    matches.sort_unstable();
    Ok(LaunchOutput::new(matches, match_count))
}

fn low_u32(value: u64) -> u32 {
    let bytes = value.to_le_bytes();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn high_u32(value: u64) -> u32 {
    let bytes = value.to_le_bytes();
    u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]])
}

#[derive(Debug)]
struct DeviceAllocation<'a> {
    runtime: &'a HipRuntime,
    ptr: *mut c_void,
}

impl Drop for DeviceAllocation<'_> {
    fn drop(&mut self) {
        let _ = unsafe { (self.runtime.api.hip_free)(self.ptr) };
    }
}

#[derive(Debug)]
struct LoadedModule<'a> {
    runtime: &'a HipRuntime,
    raw: HipModuleHandle,
}

impl Drop for LoadedModule<'_> {
    fn drop(&mut self) {
        let _ = unsafe { (self.runtime.api.hip_module_unload)(self.raw) };
    }
}

#[derive(Debug)]
struct HipRuntime {
    _library: DynamicLibrary,
    api: HipApi,
}

impl HipRuntime {
    fn load() -> Result<Self, String> {
        let library = DynamicLibrary::open_hip()?;
        let api = unsafe { HipApi::load(&library)? };
        Ok(Self {
            _library: library,
            api,
        })
    }

    fn init(&self) -> Result<(), String> {
        self.check(unsafe { (self.api.hip_init)(0) }, "hipInit")
    }

    fn set_device(&self, device: c_int) -> Result<(), String> {
        self.check(unsafe { (self.api.hip_set_device)(device) }, "hipSetDevice")
    }

    fn load_module(&self, artifact: &str) -> Result<LoadedModule<'_>, String> {
        let artifact = CString::new(artifact)
            .map_err(|_| "HIP artifact path contains interior NUL".to_owned())?;
        let mut module = ptr::null_mut();
        self.check(
            unsafe { (self.api.hip_module_load)(&mut module, artifact.as_ptr()) },
            "hipModuleLoad",
        )?;
        Ok(LoadedModule {
            runtime: self,
            raw: module,
        })
    }

    fn get_function(&self, module: HipModuleHandle, name: &str) -> Result<HipFunction, String> {
        let name = CString::new(name).map_err(|_| "HIP symbol contains interior NUL".to_owned())?;
        let mut function = ptr::null_mut();
        self.check(
            unsafe { (self.api.hip_module_get_function)(&mut function, module, name.as_ptr()) },
            "hipModuleGetFunction",
        )?;
        Ok(function)
    }

    fn malloc(&self, bytes: usize) -> Result<*mut c_void, String> {
        let mut ptr = ptr::null_mut();
        self.check(
            unsafe { (self.api.hip_malloc)(&mut ptr, bytes) },
            "hipMalloc",
        )?;
        Ok(ptr)
    }

    fn memset(&self, dst: *mut c_void, value: c_int, bytes: usize) -> Result<(), String> {
        self.check(
            unsafe { (self.api.hip_memset)(dst, value, bytes) },
            "hipMemset",
        )
    }

    fn launch_kernel(
        &self,
        function: HipFunction,
        grid_x: u32,
        block_x: u32,
        params: *mut *mut c_void,
    ) -> Result<(), String> {
        self.check(
            unsafe {
                (self.api.hip_module_launch_kernel)(
                    function,
                    grid_x,
                    1,
                    1,
                    block_x,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params,
                    ptr::null_mut(),
                )
            },
            "hipModuleLaunchKernel",
        )
    }

    fn synchronize(&self) -> Result<(), String> {
        self.check(
            unsafe { (self.api.hip_device_synchronize)() },
            "hipDeviceSynchronize",
        )
    }

    fn memcpy_device_to_host(
        &self,
        dst: *mut c_void,
        src: *mut c_void,
        bytes: usize,
    ) -> Result<(), String> {
        self.check(
            unsafe { (self.api.hip_memcpy)(dst, src, bytes, HIP_MEMCPY_DEVICE_TO_HOST) },
            "hipMemcpy",
        )
    }

    fn check(&self, result: HipResult, operation: &str) -> Result<(), String> {
        if result == HIP_SUCCESS {
            Ok(())
        } else {
            Err(format!("{operation} failed with HIP error {result}"))
        }
    }
}

#[derive(Debug)]
struct HipApi {
    hip_init: unsafe extern "C" fn(c_uint) -> HipResult,
    hip_set_device: unsafe extern "C" fn(c_int) -> HipResult,
    hip_module_load: unsafe extern "C" fn(*mut HipModuleHandle, *const c_char) -> HipResult,
    hip_module_get_function:
        unsafe extern "C" fn(*mut HipFunction, HipModuleHandle, *const c_char) -> HipResult,
    hip_module_unload: unsafe extern "C" fn(HipModuleHandle) -> HipResult,
    hip_malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> HipResult,
    hip_free: unsafe extern "C" fn(*mut c_void) -> HipResult,
    hip_memset: unsafe extern "C" fn(*mut c_void, c_int, usize) -> HipResult,
    hip_memcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_uint) -> HipResult,
    hip_device_synchronize: unsafe extern "C" fn() -> HipResult,
    hip_module_launch_kernel: unsafe extern "C" fn(
        HipFunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        HipStream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> HipResult,
}

impl HipApi {
    unsafe fn load(library: &DynamicLibrary) -> Result<Self, String> {
        Ok(Self {
            hip_init: library.symbol("hipInit")?,
            hip_set_device: library.symbol("hipSetDevice")?,
            hip_module_load: library.symbol("hipModuleLoad")?,
            hip_module_get_function: library.symbol("hipModuleGetFunction")?,
            hip_module_unload: library.symbol("hipModuleUnload")?,
            hip_malloc: library.symbol("hipMalloc")?,
            hip_free: library.symbol("hipFree")?,
            hip_memset: library.symbol("hipMemset")?,
            hip_memcpy: library.symbol("hipMemcpy")?,
            hip_device_synchronize: library.symbol("hipDeviceSynchronize")?,
            hip_module_launch_kernel: library.symbol("hipModuleLaunchKernel")?,
        })
    }
}

#[derive(Debug)]
struct DynamicLibrary {
    handle: *mut c_void,
}

impl DynamicLibrary {
    fn open_hip() -> Result<Self, String> {
        #[cfg(windows)]
        {
            if let Some(library) = find_hip_root_runtime_library() {
                return Self::open(&library);
            }
            Self::open("amdhip64.dll")
                .or_else(|_| Self::open("amdhip64_7.dll"))
                .or_else(|_| Self::open("amdhip64_6.dll"))
        }
        #[cfg(target_os = "linux")]
        {
            if let Some(library) = find_hip_root_runtime_library() {
                return Self::open(&library);
            }
            Self::open("libamdhip64.so").or_else(|_| Self::open("libamdhip64.so.6"))
        }
        #[cfg(target_os = "macos")]
        {
            Err("HIP runtime loading is unsupported on macOS".to_owned())
        }
        #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
        {
            Err("HIP runtime loading is unsupported on this platform".to_owned())
        }
    }

    fn open(name: &str) -> Result<Self, String> {
        let name =
            CString::new(name).map_err(|_| "library name contains interior NUL".to_owned())?;
        let handle = unsafe { platform_open(name.as_ptr()) };
        if handle.is_null() {
            Err(format!(
                "failed to load HIP runtime library {}",
                name.to_string_lossy()
            ))
        } else {
            Ok(Self { handle })
        }
    }

    unsafe fn symbol<T: Copy>(&self, name: &str) -> Result<T, String> {
        let name =
            CString::new(name).map_err(|_| "symbol name contains interior NUL".to_owned())?;
        let symbol = platform_symbol(self.handle, name.as_ptr());
        if symbol.is_null() {
            Err(format!(
                "missing HIP runtime symbol {}",
                name.to_string_lossy()
            ))
        } else {
            Ok(std::mem::transmute_copy::<*mut c_void, T>(&symbol))
        }
    }
}

impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        unsafe {
            platform_close(self.handle);
        }
    }
}

/// Returns candidate HIP runtime dynamic-library paths from HIP/ROCm SDK roots.
#[must_use]
pub fn hip_runtime_library_candidates_from_roots(
    roots: impl IntoIterator<Item = PathBuf>,
) -> Vec<PathBuf> {
    roots
        .into_iter()
        .flat_map(|root| {
            hip_runtime_library_dirs(&root).into_iter().flat_map(|dir| {
                hip_runtime_library_names()
                    .into_iter()
                    .map(move |name| dir.join(name))
            })
        })
        .collect()
}

fn find_hip_root_runtime_library() -> Option<String> {
    hip_runtime_library_candidates_from_host_roots()
        .into_iter()
        .find(|path| path.is_file())
        .map(|path| path.to_string_lossy().into_owned())
}

/// Returns HIP runtime dynamic-library candidates from host environment and
/// standard ROCm install roots.
#[must_use]
pub fn hip_runtime_library_candidates_from_host_roots() -> Vec<PathBuf> {
    hip_runtime_library_candidates_from_roots(hip_root_dirs_from_host())
}

fn hip_root_dirs_from_host() -> Vec<PathBuf> {
    let mut roots = hip_root_dirs_from_env();
    roots.extend(hip_standard_root_dirs());
    dedup_paths(roots)
}

fn hip_root_dirs_from_env() -> Vec<PathBuf> {
    ["HIP_PATH", "ROCM_PATH", "ROCM_HOME"]
        .into_iter()
        .filter_map(std::env::var_os)
        .flat_map(|value| std::env::split_paths(&value).collect::<Vec<_>>())
        .collect()
}

fn hip_standard_root_dirs() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    #[cfg(windows)]
    {
        for base in ["ProgramFiles", "ProgramFiles(x86)"]
            .into_iter()
            .filter_map(std::env::var_os)
            .map(PathBuf::from)
        {
            let rocm_base = base.join("AMD").join("ROCm");
            let mut versioned_roots = Vec::new();
            if let Ok(entries) = fs::read_dir(&rocm_base) {
                for path in entries.filter_map(Result::ok).map(|entry| entry.path()) {
                    if path.is_dir() {
                        versioned_roots.push(path);
                    }
                }
            }
            versioned_roots.sort_by(|left, right| {
                sdk_version_key(right)
                    .cmp(&sdk_version_key(left))
                    .then_with(|| right.cmp(left))
            });
            for path in versioned_roots {
                roots.push(path);
            }
            push_existing_dir(&mut roots, rocm_base.clone());
        }
    }
    #[cfg(not(windows))]
    {
        roots.push(PathBuf::from("/opt/rocm"));
        roots.push(PathBuf::from("/opt/rocm/hip"));
    }
    roots
}

fn sdk_version_key(path: &Path) -> Vec<u32> {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.trim_start_matches('v'))
        .map(|version| {
            version
                .split('.')
                .map(|component| component.parse::<u32>().unwrap_or(0))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn push_existing_dir(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if path.is_dir() {
        paths.push(path);
    }
}

fn find_hipcc_command() -> Option<PathBuf> {
    hipcc_command_candidates_from_roots(hip_root_dirs_from_host())
        .into_iter()
        .find(|path| path.is_file())
        .or_else(|| find_command_on_path(hipcc_command_names()))
}

fn hipcc_command_candidates_from_roots(roots: impl IntoIterator<Item = PathBuf>) -> Vec<PathBuf> {
    roots
        .into_iter()
        .flat_map(|root| {
            [root.clone(), root.join("bin")]
                .into_iter()
                .flat_map(|dir| {
                    hipcc_command_names()
                        .into_iter()
                        .map(move |name| dir.join(name))
                })
        })
        .collect()
}

fn hipcc_command_names() -> Vec<&'static str> {
    #[cfg(windows)]
    {
        vec!["hipcc.exe", "hipcc.bat", "hipcc.cmd"]
    }
    #[cfg(not(windows))]
    {
        vec!["hipcc"]
    }
}

fn find_command_on_path(names: Vec<&'static str>) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .flat_map(|dir| names.iter().map(move |name| dir.join(name)))
        .find(|candidate| candidate.is_file())
}

fn format_path_candidates(candidates: impl IntoIterator<Item = PathBuf>) -> String {
    let paths = candidates
        .into_iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>();
    if paths.is_empty() {
        "<none>".to_owned()
    } else {
        paths.join(", ")
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

fn hip_runtime_library_dirs(root: &Path) -> Vec<PathBuf> {
    vec![
        root.join("bin"),
        root.join("lib"),
        root.join("lib64"),
        root.join("hip").join("bin"),
        root.join("hip").join("lib"),
        root.join("hip").join("lib64"),
    ]
}

fn hip_runtime_library_names() -> Vec<&'static str> {
    #[cfg(windows)]
    {
        vec!["amdhip64.dll", "amdhip64_7.dll", "amdhip64_6.dll"]
    }
    #[cfg(target_os = "linux")]
    {
        vec!["libamdhip64.so", "libamdhip64.so.6"]
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        Vec::new()
    }
}

#[cfg(windows)]
unsafe fn platform_open(name: *const c_char) -> *mut c_void {
    extern "system" {
        fn LoadLibraryA(lp_lib_file_name: *const c_char) -> *mut c_void;
    }
    LoadLibraryA(name)
}

#[cfg(windows)]
unsafe fn platform_symbol(handle: *mut c_void, name: *const c_char) -> *mut c_void {
    extern "system" {
        fn GetProcAddress(h_module: *mut c_void, lp_proc_name: *const c_char) -> *mut c_void;
    }
    GetProcAddress(handle, name)
}

#[cfg(windows)]
unsafe fn platform_close(handle: *mut c_void) {
    extern "system" {
        fn FreeLibrary(h_lib_module: *mut c_void) -> c_int;
    }
    let _ = FreeLibrary(handle);
}

#[cfg(unix)]
unsafe fn platform_open(name: *const c_char) -> *mut c_void {
    const RTLD_NOW: c_int = 2;
    extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    }
    dlopen(name, RTLD_NOW)
}

#[cfg(unix)]
unsafe fn platform_symbol(handle: *mut c_void, name: *const c_char) -> *mut c_void {
    extern "C" {
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
    dlsym(handle, name)
}

#[cfg(unix)]
unsafe fn platform_close(handle: *mut c_void) {
    extern "C" {
        fn dlclose(handle: *mut c_void) -> c_int;
    }
    let _ = dlclose(handle);
}

#[cfg(test)]
mod tests {
    use super::artifact_uses_u32_launch_abi;

    #[test]
    fn hip_artifact_marker_selects_u32_launch_abi() {
        assert!(artifact_uses_u32_launch_abi(
            b"extern \"C\" __device__ unsigned int atlas_search_u32_abi = 1U;"
        ));
        assert!(artifact_uses_u32_launch_abi(
            b"\0atlas_search_u32_abi\0atlas_search\0"
        ));
        assert!(!artifact_uses_u32_launch_abi(
            b"extern \"C\" __global__ void atlas_search(unsigned long long start)"
        ));
    }
}
