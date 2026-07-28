//! CUDA launch adapter.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr;

const CUDA_MAX_THREADS_PER_BLOCK: usize = 1024;

type CuDevice = c_int;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuDevicePtr = u64;
type CuResult = c_int;
type NvrtcProgram = *mut c_void;
type NvrtcResult = c_int;

const CUDA_SUCCESS: CuResult = 0;
const NVRTC_SUCCESS: NvrtcResult = 0;

/// Parsed CUDA launch protocol arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchArgs {
    /// Generated PTX artifact.
    pub artifact: String,
    /// Inclusive search start.
    pub start: u64,
    /// Exclusive search end.
    pub end: u64,
    /// Maximum number of retained matches.
    pub max_matches: usize,
    /// Global CUDA work size.
    pub global_size: usize,
    /// CUDA block size.
    pub local_size: usize,
    /// Optional explicit host/kernel launch ABI.
    pub launch_abi: Option<LaunchAbi>,
}

/// Host/kernel launch ABI used by generated CUDA kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaunchAbi {
    /// Split 64-bit launch bounds and match output into 32-bit words.
    U32,
    /// Use native 64-bit CUDA kernel parameters and match output.
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
    /// Build-check a PTX file without launching a search.
    CompileCheck {
        /// Generated CUDA source or PTX path.
        input: String,
        /// Optional compiled PTX output path.
        output: Option<String>,
    },
    /// Launch a CUDA search.
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
            let Some(ptx) = args.get(1) else {
                return Err("missing compile-check PTX".to_owned());
            };
            return Ok(Self::CompileCheck {
                input: ptx.clone(),
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
            return Err("max-matches exceeds CUDA uint".to_owned());
        }
        let global_size = parse_usize_flag(args, "--global-size")?;
        let local_size = parse_usize_flag(args, "--local-size")?;
        if global_size == 0 || local_size == 0 {
            return Err("global-size and local-size must be nonzero".to_owned());
        }
        if local_size > CUDA_MAX_THREADS_PER_BLOCK {
            return Err("local-size exceeds CUDA block limit".to_owned());
        }
        if global_size.div_ceil(local_size) > u32::MAX as usize {
            return Err("grid size exceeds CUDA uint".to_owned());
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

/// Launches one parsed CUDA request.
pub trait Launcher {
    /// Reports runtime/device features available to generated kernels.
    ///
    /// # Errors
    ///
    /// Returns an error when CUDA driver loading or context creation fails.
    fn features(&self) -> Result<FeatureReport, String>;

    /// Checks generated PTX can provide the expected `atlas_search` kernel.
    ///
    /// # Errors
    ///
    /// Returns an error when the PTX cannot be loaded or lacks the expected
    /// kernel entry.
    fn compile_check(&self, input: &str, output: Option<&str>) -> Result<(), String>;

    /// Runs the launch and returns device-reported matches.
    ///
    /// # Errors
    ///
    /// Returns an error when CUDA driver loading, module loading, kernel launch,
    /// or device transfer fails.
    fn launch(&self, args: &LaunchArgs) -> Result<LaunchOutput, String>;
}

/// Runtime/device feature report emitted by `--features`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureReport {
    /// Concrete hardware/runtime identity selected by the adapter.
    pub hardware: String,
    /// Kernel capabilities available for generated CUDA code.
    pub features: Vec<String>,
}

/// CUDA Driver API backed PTX launcher.
#[derive(Debug, Clone, Copy, Default)]
pub struct CudaPtxLauncher;

impl Launcher for CudaPtxLauncher {
    fn features(&self) -> Result<FeatureReport, String> {
        let driver = CudaDriver::load()?;
        let device = driver.first_device()?;
        let hardware = driver.device_identity(device);
        let _context = driver.create_context_for(device)?;
        Ok(FeatureReport {
            hardware,
            features: features_from_int64_probe(probe_cuda_int64(&driver)),
        })
    }

    fn compile_check(&self, artifact: &str, output: Option<&str>) -> Result<(), String> {
        let ptx = read_cuda_artifact_as_ptx(artifact)?;
        ensure_atlas_entry(&ptx)?;
        if let Some(output) = output {
            write_ptx(output, &ptx)?;
        }
        if let Ok(driver) = CudaDriver::load() {
            let _context = driver.create_context()?;
            let ptx = CString::new(ptx).map_err(|_| "PTX contains interior NUL".to_owned())?;
            let module = driver.load_module(&ptx)?;
            driver.get_function(module.raw, "atlas_search")?;
        }
        Ok(())
    }

    fn launch(&self, args: &LaunchArgs) -> Result<LaunchOutput, String> {
        let ptx = read_cuda_artifact_as_ptx(&args.artifact)?;
        ensure_atlas_entry(&ptx)?;
        let uses_u32_abi = match args.launch_abi {
            Some(LaunchAbi::U32) => true,
            Some(LaunchAbi::U64) => false,
            None => uses_u32_launch_abi(&ptx),
        };
        let driver = CudaDriver::load()?;
        let _context = driver.create_context()?;
        let ptx = CString::new(ptx).map_err(|_| "PTX contains interior NUL".to_owned())?;
        let module = driver.load_module(&ptx)?;
        let function = driver.get_function(module.raw, "atlas_search")?;
        if uses_u32_abi {
            launch_cuda_u32(&driver, function, args)
        } else {
            launch_cuda(&driver, function, args)
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
            let report = launcher.features()?;
            Ok(format_features(&report))
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

fn format_features(report: &FeatureReport) -> String {
    let mut text = format!("hardware={}\n", report.hardware);
    text.push_str(
        &report
            .features
            .iter()
            .map(|feature| format!("feature={feature}\n"))
            .collect::<String>(),
    );
    text.push_str("feature=launchAbiU32\nfeature=launchAbiU64\n");
    text
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

fn features_from_int64_probe(probe: Result<(), String>) -> Vec<String> {
    if probe.is_ok() {
        vec!["int64".to_owned()]
    } else {
        Vec::new()
    }
}

fn probe_cuda_int64(driver: &CudaDriver) -> Result<(), String> {
    let ptx = compile_cuda_probe_source_to_ptx(cuda_int64_probe_source())?;
    let ptx =
        CString::new(ptx).map_err(|_| "CUDA int64 probe PTX contains interior NUL".to_owned())?;
    let module = driver.load_module(&ptx)?;
    driver
        .get_function(module.raw, "atlas_int64_probe")
        .map(|_| ())
}

fn compile_cuda_probe_source_to_ptx(source: &str) -> Result<String, String> {
    match NvrtcCompiler::load().and_then(|compiler| compiler.compile_source_to_ptx(source)) {
        Ok(ptx) => Ok(ptx),
        Err(nvrtc_error) => {
            let source_path = write_cuda_probe_source(source)?;
            let result = NvccCompiler::load()
                .and_then(|compiler| compiler.compile_source_file_to_ptx(&source_path))
                .or_else(|nvcc_error| {
                    ClangCudaCompiler::load()
                        .and_then(|compiler| compiler.compile_source_file_to_ptx(&source_path))
                        .map_err(|clang_error| {
                            format!(
                                "failed to compile CUDA int64 probe with NVRTC, nvcc, or clang; NVRTC: {nvrtc_error}; nvcc: {nvcc_error}; clang: {clang_error}"
                            )
                        })
                });
            let _ = fs::remove_file(source_path);
            result
        }
    }
}

fn write_cuda_probe_source(source: &str) -> Result<String, String> {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!(
        "atlas-cuda-int64-probe-{}-{nonce}.cu",
        std::process::id()
    ));
    fs::write(&path, source).map_err(|error| {
        format!(
            "cannot write CUDA int64 probe source {}: {error}",
            path.display()
        )
    })?;
    Ok(path.to_string_lossy().into_owned())
}

fn cuda_int64_probe_source() -> &'static str {
    r#"#if defined(__clang__)
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#endif
__device__ unsigned int atlas_probe_global_id_x() {
#if defined(__clang__)
  return __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_x() + __nvvm_read_ptx_sreg_tid_x();
#else
  return blockIdx.x * blockDim.x + threadIdx.x;
#endif
}
extern "C" __global__ void atlas_int64_probe(unsigned long long* out) {
  unsigned long long candidate = (unsigned long long)(atlas_probe_global_id_x());
  out[0] = (candidate << 32) ^ candidate;
}
"#
}

fn read_ptx(path: &str) -> Result<String, String> {
    fs::read_to_string(path).map_err(|error| error.to_string())
}

fn write_ptx(path: &str, ptx: &str) -> Result<(), String> {
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create PTX output directory: {error}"))?;
    }
    fs::write(path, ptx).map_err(|error| format!("cannot write PTX {path}: {error}"))
}

fn read_cuda_artifact_as_ptx(path: &str) -> Result<String, String> {
    if Path::new(path)
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("ptx"))
    {
        return read_ptx(path);
    }
    let source = fs::read_to_string(path)
        .map_err(|error| format!("cannot read CUDA source {path}: {error}"))?;
    match NvrtcCompiler::load().and_then(|compiler| compiler.compile_source_to_ptx(&source)) {
        Ok(ptx) => Ok(ptx),
        Err(nvrtc_error) => NvccCompiler::load()
            .and_then(|compiler| compiler.compile_source_file_to_ptx(path))
            .or_else(|nvcc_error| {
                ClangCudaCompiler::load()
                    .and_then(|compiler| compiler.compile_source_file_to_ptx(path))
                    .map_err(|clang_error| {
                        format!(
                            "failed to compile CUDA source with NVRTC, nvcc, or clang; NVRTC: {nvrtc_error}; nvcc: {nvcc_error}; clang: {clang_error}"
                        )
                    })
            }),
    }
}

fn ensure_atlas_entry(ptx: &str) -> Result<(), String> {
    if ptx.contains(".entry atlas_search") {
        Ok(())
    } else {
        Err("missing atlas_search kernel entry".to_owned())
    }
}

fn uses_u32_launch_abi(artifact: &str) -> bool {
    artifact.contains("atlas_search_u32_abi")
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

fn parse_launch_abi(value: &str) -> Result<LaunchAbi, String> {
    match value {
        "u32" => Ok(LaunchAbi::U32),
        "u64" => Ok(LaunchAbi::U64),
        _ => Err(format!("unsupported --abi '{value}'; expected u32 or u64")),
    }
}

fn launch_cuda(
    driver: &CudaDriver,
    function: CuFunction,
    args: &LaunchArgs,
) -> Result<LaunchOutput, String> {
    let out_len = driver.mem_alloc(std::mem::size_of::<u32>())?;
    let out_bytes = args
        .max_matches
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or_else(|| "output buffer size overflow".to_owned())?;
    let out = driver.mem_alloc(out_bytes)?;
    let _out_len_guard = DeviceAllocation {
        driver,
        ptr: out_len,
    };
    let _out_guard = DeviceAllocation { driver, ptr: out };
    driver.memset_d32(out_len, 0, 1)?;

    let mut start = args.start;
    let mut end = args.end;
    let mut out_param = out;
    let mut out_len_param = out_len;
    let mut max_matches =
        u32::try_from(args.max_matches).map_err(|_| "max-matches exceeds CUDA uint".to_owned())?;
    let mut params = [
        (&mut start as *mut u64).cast::<c_void>(),
        (&mut end as *mut u64).cast::<c_void>(),
        (&mut out_param as *mut CuDevicePtr).cast::<c_void>(),
        (&mut out_len_param as *mut CuDevicePtr).cast::<c_void>(),
        (&mut max_matches as *mut u32).cast::<c_void>(),
    ];
    let block_x =
        u32::try_from(args.local_size).map_err(|_| "local-size exceeds CUDA uint".to_owned())?;
    let grid_x = args.global_size.div_ceil(args.local_size);
    let grid_x = u32::try_from(grid_x).map_err(|_| "grid size exceeds CUDA uint".to_owned())?;
    driver.launch_kernel(function, grid_x, block_x, params.as_mut_ptr())?;
    driver.synchronize()?;

    let mut retained = 0_u32;
    driver.memcpy_dtoh(
        (&mut retained as *mut u32).cast::<c_void>(),
        out_len,
        std::mem::size_of::<u32>(),
    )?;
    let match_count = usize::try_from(retained).unwrap_or(usize::MAX);
    let retained = match_count.min(args.max_matches);
    let mut matches = vec![0_u64; args.max_matches];
    driver.memcpy_dtoh(matches.as_mut_ptr().cast::<c_void>(), out, out_bytes)?;
    matches.truncate(retained);
    matches.sort_unstable();
    Ok(LaunchOutput::new(matches, match_count))
}

fn launch_cuda_u32(
    driver: &CudaDriver,
    function: CuFunction,
    args: &LaunchArgs,
) -> Result<LaunchOutput, String> {
    let out_len = driver.mem_alloc(std::mem::size_of::<u32>())?;
    let out_words_len = args
        .max_matches
        .checked_mul(2)
        .ok_or_else(|| "output word count overflow".to_owned())?;
    let out_bytes = out_words_len
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| "output buffer size overflow".to_owned())?;
    let out_words = driver.mem_alloc(out_bytes)?;
    let _out_len_guard = DeviceAllocation {
        driver,
        ptr: out_len,
    };
    let _out_guard = DeviceAllocation {
        driver,
        ptr: out_words,
    };
    driver.memset_d32(out_len, 0, 1)?;

    let mut start_lo = low_u32(args.start);
    let mut start_hi = high_u32(args.start);
    let mut end_lo = low_u32(args.end);
    let mut end_hi = high_u32(args.end);
    let mut out_param = out_words;
    let mut out_len_param = out_len;
    let mut max_matches =
        u32::try_from(args.max_matches).map_err(|_| "max-matches exceeds CUDA uint".to_owned())?;
    let mut params = [
        (&mut start_lo as *mut u32).cast::<c_void>(),
        (&mut start_hi as *mut u32).cast::<c_void>(),
        (&mut end_lo as *mut u32).cast::<c_void>(),
        (&mut end_hi as *mut u32).cast::<c_void>(),
        (&mut out_param as *mut CuDevicePtr).cast::<c_void>(),
        (&mut out_len_param as *mut CuDevicePtr).cast::<c_void>(),
        (&mut max_matches as *mut u32).cast::<c_void>(),
    ];
    let block_x =
        u32::try_from(args.local_size).map_err(|_| "local-size exceeds CUDA uint".to_owned())?;
    let grid_x = args.global_size.div_ceil(args.local_size);
    let grid_x = u32::try_from(grid_x).map_err(|_| "grid size exceeds CUDA uint".to_owned())?;
    driver.launch_kernel(function, grid_x, block_x, params.as_mut_ptr())?;
    driver.synchronize()?;

    let mut retained = 0_u32;
    driver.memcpy_dtoh(
        (&mut retained as *mut u32).cast::<c_void>(),
        out_len,
        std::mem::size_of::<u32>(),
    )?;
    let match_count = usize::try_from(retained).unwrap_or(usize::MAX);
    let retained = match_count.min(args.max_matches);
    let mut out_words_host = vec![0_u32; out_words_len];
    driver.memcpy_dtoh(
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
struct NvrtcCompiler {
    _library: DynamicLibrary,
    api: NvrtcApi,
}

impl NvrtcCompiler {
    fn load() -> Result<Self, String> {
        let library = DynamicLibrary::open_nvrtc().map_err(|error| {
            format!(
                "{error}; searched NVRTC candidates: {}",
                format_nvrtc_search_candidates_from_roots(cuda_root_dirs())
            )
        })?;
        let api = unsafe { NvrtcApi::load(&library)? };
        Ok(Self {
            _library: library,
            api,
        })
    }

    fn compile_source_to_ptx(&self, source: &str) -> Result<String, String> {
        let source =
            CString::new(source).map_err(|_| "CUDA source contains interior NUL".to_owned())?;
        let name = CString::new("atlas_search.cu").expect("static CUDA source name has no NUL");
        let mut program = ptr::null_mut();
        self.check(
            unsafe {
                (self.api.nvrtc_create_program)(
                    &mut program,
                    source.as_ptr(),
                    name.as_ptr(),
                    0,
                    ptr::null(),
                    ptr::null(),
                )
            },
            "nvrtcCreateProgram",
        )?;
        let program = NvrtcProgramGuard {
            compiler: self,
            raw: program,
        };
        let option_strings = [
            CString::new("--std=c++11").expect("static NVRTC option has no NUL"),
            CString::new("--gpu-architecture=compute_52").expect("static NVRTC option has no NUL"),
        ];
        let options = option_strings
            .iter()
            .map(|option| option.as_ptr())
            .collect::<Vec<_>>();
        let compile_result = unsafe {
            (self.api.nvrtc_compile_program)(
                program.raw,
                c_int::try_from(options.len()).expect("NVRTC option count fits c_int"),
                options.as_ptr(),
            )
        };
        if compile_result != NVRTC_SUCCESS {
            return Err(format!(
                "CUDA source compile failed: {}{}",
                self.error_name(compile_result),
                self.program_log(program.raw)
                    .map(|log| format!(": {log}"))
                    .unwrap_or_default()
            ));
        }
        let mut ptx_size = 0_usize;
        self.check(
            unsafe { (self.api.nvrtc_get_ptx_size)(program.raw, &mut ptx_size) },
            "nvrtcGetPTXSize",
        )?;
        let mut ptx = vec![0_u8; ptx_size];
        self.check(
            unsafe { (self.api.nvrtc_get_ptx)(program.raw, ptx.as_mut_ptr().cast::<c_char>()) },
            "nvrtcGetPTX",
        )?;
        String::from_utf8(ptx)
            .map(|ptx| ptx.trim_end_matches('\0').to_owned())
            .map_err(|error| format!("NVRTC produced non-UTF8 PTX: {error}"))
    }

    fn program_log(&self, program: NvrtcProgram) -> Option<String> {
        let mut log_size = 0_usize;
        if unsafe { (self.api.nvrtc_get_program_log_size)(program, &mut log_size) } != NVRTC_SUCCESS
            || log_size == 0
        {
            return None;
        }
        let mut log = vec![0_u8; log_size];
        if unsafe { (self.api.nvrtc_get_program_log)(program, log.as_mut_ptr().cast::<c_char>()) }
            != NVRTC_SUCCESS
        {
            return None;
        }
        String::from_utf8(log)
            .ok()
            .map(|log| log.trim_end_matches('\0').trim().to_owned())
            .filter(|log| !log.is_empty())
    }

    fn error_name(&self, result: NvrtcResult) -> String {
        let ptr = unsafe { (self.api.nvrtc_get_error_string)(result) };
        if ptr.is_null() {
            format!("NVRTC error {result}")
        } else {
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
        }
    }

    fn check(&self, result: NvrtcResult, operation: &str) -> Result<(), String> {
        if result == NVRTC_SUCCESS {
            Ok(())
        } else {
            Err(format!("{operation} failed: {}", self.error_name(result)))
        }
    }
}

#[derive(Debug)]
struct NvrtcProgramGuard<'a> {
    compiler: &'a NvrtcCompiler,
    raw: NvrtcProgram,
}

impl Drop for NvrtcProgramGuard<'_> {
    fn drop(&mut self) {
        let _ = unsafe { (self.compiler.api.nvrtc_destroy_program)(&mut self.raw) };
    }
}

#[derive(Debug, Clone)]
struct NvccCompiler {
    command: PathBuf,
}

impl NvccCompiler {
    fn load() -> Result<Self, String> {
        find_nvcc_command()
            .map(|command| Self { command })
            .ok_or_else(|| {
                format!(
                    "failed to find nvcc CUDA compiler command; searched nvcc candidates: {}",
                    format_nvcc_search_candidates_from_roots(cuda_root_dirs())
                )
            })
    }

    fn compile_source_file_to_ptx(&self, source_path: &str) -> Result<String, String> {
        let output_path = nvcc_ptx_output_path_for_source(source_path);
        let output = Command::new(&self.command)
            .arg("-ptx")
            .arg("-std=c++11")
            .arg("-arch=compute_52")
            .arg(source_path)
            .arg("-o")
            .arg(&output_path)
            .output()
            .map_err(|error| format!("failed to execute {}: {error}", self.command.display()))?;

        if !output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let _ = fs::remove_file(&output_path);
            return Err(format!(
                "{} exited with status {}; stdout: {}; stderr: {}",
                self.command.display(),
                output.status,
                stdout.trim(),
                stderr.trim()
            ));
        }

        let ptx = fs::read_to_string(&output_path)
            .map_err(|error| format!("cannot read nvcc PTX {}: {error}", output_path.display()))?;
        let _ = fs::remove_file(output_path);
        Ok(ptx)
    }
}

#[derive(Debug, Clone)]
struct ClangCudaCompiler {
    command: PathBuf,
}

impl ClangCudaCompiler {
    fn load() -> Result<Self, String> {
        find_clang_cuda_command()
            .map(|command| Self { command })
            .ok_or_else(|| "failed to find clang CUDA compiler command".to_owned())
    }

    fn compile_source_file_to_ptx(&self, source_path: &str) -> Result<String, String> {
        let output_path = clang_cuda_ptx_output_path_for_source(source_path);
        let output = Command::new(&self.command)
            .arg("--cuda-device-only")
            .arg("--cuda-gpu-arch=sm_52")
            .arg("-nocudainc")
            .arg("-nocudalib")
            .arg("-x")
            .arg("cuda")
            .arg(source_path)
            .arg("-S")
            .arg("-o")
            .arg(&output_path)
            .output()
            .map_err(|error| format!("failed to execute {}: {error}", self.command.display()))?;

        if !output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let _ = fs::remove_file(&output_path);
            return Err(format!(
                "{} exited with status {}; stdout: {}; stderr: {}",
                self.command.display(),
                output.status,
                stdout.trim(),
                stderr.trim()
            ));
        }

        let ptx = fs::read_to_string(&output_path).map_err(|error| {
            format!(
                "cannot read clang CUDA PTX {}: {error}",
                output_path.display()
            )
        })?;
        let _ = fs::remove_file(output_path);
        Ok(ptx)
    }
}

/// Returns a deterministic temporary PTX output path for one CUDA source path.
#[must_use]
pub fn nvcc_ptx_output_path_for_source(source_path: &str) -> PathBuf {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    source_path.hash(&mut hasher);
    std::env::temp_dir().join(format!(
        "atlas-cuda-nvcc-{}-{:016x}.ptx",
        std::process::id(),
        hasher.finish()
    ))
}

fn clang_cuda_ptx_output_path_for_source(source_path: &str) -> PathBuf {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    source_path.hash(&mut hasher);
    std::env::temp_dir().join(format!(
        "atlas-cuda-clang-{}-{:016x}.ptx",
        std::process::id(),
        hasher.finish()
    ))
}

#[derive(Debug)]
struct NvrtcApi {
    nvrtc_create_program: unsafe extern "C" fn(
        *mut NvrtcProgram,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> NvrtcResult,
    nvrtc_compile_program:
        unsafe extern "C" fn(NvrtcProgram, c_int, *const *const c_char) -> NvrtcResult,
    nvrtc_get_ptx_size: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult,
    nvrtc_get_ptx: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult,
    nvrtc_get_program_log_size: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult,
    nvrtc_get_program_log: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult,
    nvrtc_destroy_program: unsafe extern "C" fn(*mut NvrtcProgram) -> NvrtcResult,
    nvrtc_get_error_string: unsafe extern "C" fn(NvrtcResult) -> *const c_char,
}

impl NvrtcApi {
    unsafe fn load(library: &DynamicLibrary) -> Result<Self, String> {
        Ok(Self {
            nvrtc_create_program: library.symbol("nvrtcCreateProgram")?,
            nvrtc_compile_program: library.symbol("nvrtcCompileProgram")?,
            nvrtc_get_ptx_size: library.symbol("nvrtcGetPTXSize")?,
            nvrtc_get_ptx: library.symbol("nvrtcGetPTX")?,
            nvrtc_get_program_log_size: library.symbol("nvrtcGetProgramLogSize")?,
            nvrtc_get_program_log: library.symbol("nvrtcGetProgramLog")?,
            nvrtc_destroy_program: library.symbol("nvrtcDestroyProgram")?,
            nvrtc_get_error_string: library.symbol("nvrtcGetErrorString")?,
        })
    }
}

#[derive(Debug)]
struct DeviceAllocation<'a> {
    driver: &'a CudaDriver,
    ptr: CuDevicePtr,
}

impl Drop for DeviceAllocation<'_> {
    fn drop(&mut self) {
        let _ = unsafe { (self.driver.api.cu_mem_free)(self.ptr) };
    }
}

#[derive(Debug)]
struct CudaContext<'a> {
    driver: &'a CudaDriver,
    raw: CuContext,
}

impl Drop for CudaContext<'_> {
    fn drop(&mut self) {
        let _ = unsafe { (self.driver.api.cu_ctx_destroy)(self.raw) };
    }
}

#[derive(Debug)]
struct CudaModule<'a> {
    driver: &'a CudaDriver,
    raw: CuModule,
}

impl Drop for CudaModule<'_> {
    fn drop(&mut self) {
        let _ = unsafe { (self.driver.api.cu_module_unload)(self.raw) };
    }
}

#[derive(Debug)]
struct CudaDriver {
    _library: DynamicLibrary,
    api: CudaApi,
}

impl CudaDriver {
    fn load() -> Result<Self, String> {
        let library = DynamicLibrary::open_cuda()
            .map_err(|error| cuda_driver_load_error(error, cuda_root_dirs()))?;
        let api = unsafe { CudaApi::load(&library)? };
        let driver = Self {
            _library: library,
            api,
        };
        driver.check(unsafe { (driver.api.cu_init)(0) }, "cuInit")?;
        Ok(driver)
    }

    fn create_context(&self) -> Result<CudaContext<'_>, String> {
        self.create_context_for(self.first_device()?)
    }

    fn first_device(&self) -> Result<CuDevice, String> {
        let mut device = 0;
        self.check(
            unsafe { (self.api.cu_device_get)(&mut device, 0) },
            "cuDeviceGet",
        )?;
        Ok(device)
    }

    fn create_context_for(&self, device: CuDevice) -> Result<CudaContext<'_>, String> {
        let mut context = ptr::null_mut();
        self.check(
            unsafe { (self.api.cu_ctx_create)(&mut context, 0, device) },
            "cuCtxCreate",
        )?;
        Ok(CudaContext {
            driver: self,
            raw: context,
        })
    }

    fn device_identity(&self, device: CuDevice) -> String {
        let mut name = [0 as c_char; 256];
        let result = unsafe {
            (self.api.cu_device_get_name)(
                name.as_mut_ptr(),
                c_int::try_from(name.len()).expect("CUDA device name buffer length fits c_int"),
                device,
            )
        };
        if result != CUDA_SUCCESS {
            return "CUDA driver device via CUDA".to_owned();
        }
        let name = unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .trim()
            .to_owned();
        let name = if name.is_empty() {
            "CUDA driver device"
        } else {
            name.as_str()
        };
        format!("{name} via CUDA")
    }

    fn load_module(&self, ptx: &CStr) -> Result<CudaModule<'_>, String> {
        let mut module = ptr::null_mut();
        self.check(
            unsafe { (self.api.cu_module_load_data)(&mut module, ptx.as_ptr().cast::<c_void>()) },
            "cuModuleLoadData",
        )?;
        Ok(CudaModule {
            driver: self,
            raw: module,
        })
    }

    fn get_function(&self, module: CuModule, name: &str) -> Result<CuFunction, String> {
        let name =
            CString::new(name).map_err(|_| "CUDA symbol contains interior NUL".to_owned())?;
        let mut function = ptr::null_mut();
        self.check(
            unsafe { (self.api.cu_module_get_function)(&mut function, module, name.as_ptr()) },
            "cuModuleGetFunction",
        )?;
        Ok(function)
    }

    fn mem_alloc(&self, bytes: usize) -> Result<CuDevicePtr, String> {
        let mut ptr = 0;
        self.check(
            unsafe { (self.api.cu_mem_alloc)(&mut ptr, bytes) },
            "cuMemAlloc",
        )?;
        Ok(ptr)
    }

    fn memset_d32(&self, dst: CuDevicePtr, value: u32, count: usize) -> Result<(), String> {
        self.check(
            unsafe { (self.api.cu_memset_d32)(dst, value, count) },
            "cuMemsetD32",
        )
    }

    fn launch_kernel(
        &self,
        function: CuFunction,
        grid_x: u32,
        block_x: u32,
        params: *mut *mut c_void,
    ) -> Result<(), String> {
        self.check(
            unsafe {
                (self.api.cu_launch_kernel)(
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
            "cuLaunchKernel",
        )
    }

    fn synchronize(&self) -> Result<(), String> {
        self.check(
            unsafe { (self.api.cu_ctx_synchronize)() },
            "cuCtxSynchronize",
        )
    }

    fn memcpy_dtoh(&self, dst: *mut c_void, src: CuDevicePtr, bytes: usize) -> Result<(), String> {
        self.check(
            unsafe { (self.api.cu_memcpy_dtoh)(dst, src, bytes) },
            "cuMemcpyDtoH",
        )
    }

    fn check(&self, result: CuResult, operation: &str) -> Result<(), String> {
        if result == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{operation} failed with CUDA error {result}"))
        }
    }
}

fn cuda_driver_load_error(
    error: impl AsRef<str>,
    roots: impl IntoIterator<Item = PathBuf>,
) -> String {
    format!(
        "{}; searched CUDA driver candidates: {}",
        error.as_ref(),
        format_cuda_driver_search_candidates_from_roots(roots)
    )
}

#[derive(Debug)]
struct CudaApi {
    cu_init: unsafe extern "C" fn(c_uint) -> CuResult,
    cu_device_get: unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult,
    cu_device_get_name: unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult,
    cu_ctx_create: unsafe extern "C" fn(*mut CuContext, c_uint, CuDevice) -> CuResult,
    cu_ctx_destroy: unsafe extern "C" fn(CuContext) -> CuResult,
    cu_ctx_synchronize: unsafe extern "C" fn() -> CuResult,
    cu_module_load_data: unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult,
    cu_module_get_function:
        unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult,
    cu_module_unload: unsafe extern "C" fn(CuModule) -> CuResult,
    cu_mem_alloc: unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult,
    cu_mem_free: unsafe extern "C" fn(CuDevicePtr) -> CuResult,
    cu_memset_d32: unsafe extern "C" fn(CuDevicePtr, c_uint, usize) -> CuResult,
    cu_memcpy_dtoh: unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> CuResult,
    cu_launch_kernel: unsafe extern "C" fn(
        CuFunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CuResult,
}

impl CudaApi {
    unsafe fn load(library: &DynamicLibrary) -> Result<Self, String> {
        Ok(Self {
            cu_init: library.symbol("cuInit")?,
            cu_device_get: library.symbol("cuDeviceGet")?,
            cu_device_get_name: library.symbol("cuDeviceGetName")?,
            cu_ctx_create: library.symbol("cuCtxCreate_v2")?,
            cu_ctx_destroy: library.symbol("cuCtxDestroy_v2")?,
            cu_ctx_synchronize: library.symbol("cuCtxSynchronize")?,
            cu_module_load_data: library.symbol("cuModuleLoadData")?,
            cu_module_get_function: library.symbol("cuModuleGetFunction")?,
            cu_module_unload: library.symbol("cuModuleUnload")?,
            cu_mem_alloc: library.symbol("cuMemAlloc_v2")?,
            cu_mem_free: library.symbol("cuMemFree_v2")?,
            cu_memset_d32: library.symbol("cuMemsetD32_v2")?,
            cu_memcpy_dtoh: library.symbol("cuMemcpyDtoH_v2")?,
            cu_launch_kernel: library.symbol("cuLaunchKernel")?,
        })
    }
}

#[derive(Debug)]
struct DynamicLibrary {
    handle: *mut c_void,
}

impl DynamicLibrary {
    fn open_cuda() -> Result<Self, String> {
        #[cfg(windows)]
        {
            if let Some(library) = find_windows_cuda_driver_library() {
                return Self::open(&library);
            }
            if let Some(library) = find_cuda_root_driver_library() {
                return Self::open(&library);
            }
            Self::open("nvcuda.dll")
        }
        #[cfg(target_os = "linux")]
        {
            if let Some(library) = find_cuda_root_driver_library() {
                return Self::open(&library);
            }
            Self::open("libcuda.so.1").or_else(|_| Self::open("libcuda.so"))
        }
        #[cfg(target_os = "macos")]
        {
            if let Some(library) = find_cuda_root_driver_library() {
                return Self::open(&library);
            }
            Self::open("/usr/local/cuda/lib/libcuda.dylib").or_else(|_| Self::open("libcuda.dylib"))
        }
        #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
        {
            Err("CUDA driver loading is unsupported on this platform".to_owned())
        }
    }

    fn open_nvrtc() -> Result<Self, String> {
        #[cfg(windows)]
        {
            if let Some(library) = find_windows_nvrtc_library() {
                return Self::open(&library);
            }
            if let Some(library) = find_cuda_root_nvrtc_library() {
                return Self::open(&library);
            }
            [
                "nvrtc64_130_0.dll",
                "nvrtc64_120_0.dll",
                "nvrtc64_122_0.dll",
                "nvrtc64_121_0.dll",
                "nvrtc64_112_0.dll",
                "nvrtc64_111_0.dll",
                "nvrtc64_110_0.dll",
            ]
            .into_iter()
            .find_map(|name| Self::open(name).ok())
            .ok_or_else(|| "failed to load NVRTC runtime compiler library".to_owned())
        }
        #[cfg(target_os = "linux")]
        {
            if let Some(library) = find_cuda_root_nvrtc_library() {
                return Self::open(&library);
            }
            Self::open("libnvrtc.so.12")
                .or_else(|_| Self::open("libnvrtc.so.11"))
                .or_else(|_| Self::open("libnvrtc.so"))
                .map_err(|_| "failed to load NVRTC runtime compiler library".to_owned())
        }
        #[cfg(target_os = "macos")]
        {
            if let Some(library) = find_cuda_root_nvrtc_library() {
                return Self::open(&library);
            }
            Self::open("/usr/local/cuda/lib/libnvrtc.dylib")
                .or_else(|_| Self::open("libnvrtc.dylib"))
                .map_err(|_| "failed to load NVRTC runtime compiler library".to_owned())
        }
        #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
        {
            Err("NVRTC loading is unsupported on this platform".to_owned())
        }
    }

    fn open(name: &str) -> Result<Self, String> {
        let name =
            CString::new(name).map_err(|_| "library name contains interior NUL".to_owned())?;
        let handle = unsafe { platform_open(name.as_ptr()) };
        if handle.is_null() {
            Err(format!(
                "failed to load CUDA driver library {}",
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
                "missing CUDA driver symbol {}",
                name.to_string_lossy()
            ))
        } else {
            Ok(std::mem::transmute_copy::<*mut c_void, T>(&symbol))
        }
    }
}

/// Returns candidate CUDA driver dynamic-library paths from CUDA SDK roots.
#[must_use]
pub fn cuda_driver_library_candidates_from_roots(
    roots: impl IntoIterator<Item = PathBuf>,
) -> Vec<PathBuf> {
    let mut candidates: Vec<PathBuf> = roots
        .into_iter()
        .flat_map(|root| {
            cuda_driver_library_dirs(&root).into_iter().flat_map(|dir| {
                cuda_driver_library_names()
                    .into_iter()
                    .map(move |name| dir.join(name))
            })
        })
        .collect();
    candidates.extend(cuda_system_driver_library_candidates());
    candidates
}

fn find_windows_cuda_driver_library() -> Option<String> {
    cuda_system_driver_library_candidates()
        .into_iter()
        .find(|path| path.is_file())
        .map(|path| path.to_string_lossy().into_owned())
}

fn find_cuda_root_driver_library() -> Option<String> {
    cuda_driver_library_candidates_from_roots(cuda_root_dirs())
        .into_iter()
        .find(|path| path.is_file())
        .map(|path| path.to_string_lossy().into_owned())
}

fn format_cuda_driver_search_candidates_from_roots(
    roots: impl IntoIterator<Item = PathBuf>,
) -> String {
    format_path_candidates(cuda_driver_library_candidates_from_roots(roots))
}

fn cuda_driver_library_dirs(root: &Path) -> Vec<PathBuf> {
    vec![
        root.join("compat"),
        root.join("lib64").join("stubs"),
        root.join("lib64"),
        root.join("lib").join("x64"),
        root.join("lib"),
        root.join("bin"),
    ]
}

fn cuda_driver_library_names() -> Vec<&'static str> {
    #[cfg(windows)]
    {
        vec!["nvcuda.dll"]
    }
    #[cfg(target_os = "linux")]
    {
        vec!["libcuda.so.1", "libcuda.so"]
    }
    #[cfg(target_os = "macos")]
    {
        vec!["libcuda.dylib"]
    }
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
    {
        Vec::new()
    }
}

fn cuda_system_driver_library_candidates() -> Vec<PathBuf> {
    #[cfg(windows)]
    {
        let Some(system_root) = std::env::var_os("SystemRoot") else {
            return Vec::new();
        };
        let system_root = PathBuf::from(system_root);
        vec![
            system_root.join("System32").join("nvcuda.dll"),
            system_root.join("SysWOW64").join("nvcuda.dll"),
        ]
    }
    #[cfg(not(windows))]
    {
        Vec::new()
    }
}

/// Returns candidate NVRTC dynamic-library paths from CUDA SDK roots.
#[must_use]
pub fn nvrtc_library_candidates_from_roots(
    roots: impl IntoIterator<Item = PathBuf>,
) -> Vec<PathBuf> {
    roots
        .into_iter()
        .flat_map(|root| {
            nvrtc_library_dirs(&root).into_iter().flat_map(|dir| {
                nvrtc_library_names()
                    .into_iter()
                    .map(move |name| dir.join(name))
            })
        })
        .collect()
}

/// Returns candidate `nvcc` compiler commands from CUDA SDK roots.
#[must_use]
pub fn nvcc_command_candidates_from_roots(
    roots: impl IntoIterator<Item = PathBuf>,
) -> Vec<PathBuf> {
    roots
        .into_iter()
        .flat_map(|root| {
            nvcc_command_names()
                .into_iter()
                .map(move |name| root.join("bin").join(name))
        })
        .collect()
}

fn find_nvcc_command() -> Option<PathBuf> {
    nvcc_command_candidates_from_roots(cuda_root_dirs())
        .into_iter()
        .find(|path| path.is_file())
        .or_else(|| find_command_on_path(nvcc_command_names()))
}

fn find_clang_cuda_command() -> Option<PathBuf> {
    find_command_on_path(clang_cuda_command_names())
}

fn nvcc_command_names() -> Vec<&'static str> {
    #[cfg(windows)]
    {
        vec!["nvcc.exe", "nvcc.bat", "nvcc.cmd"]
    }
    #[cfg(not(windows))]
    {
        vec!["nvcc"]
    }
}

fn clang_cuda_command_names() -> Vec<&'static str> {
    #[cfg(windows)]
    {
        vec![
            "clang++.exe",
            "clang++.bat",
            "clang++.cmd",
            "clang.exe",
            "clang.bat",
            "clang.cmd",
        ]
    }
    #[cfg(not(windows))]
    {
        vec!["clang++", "clang"]
    }
}

fn find_command_on_path(names: Vec<&'static str>) -> Option<PathBuf> {
    std::env::var_os("PATH")
        .into_iter()
        .flat_map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
        .flat_map(|dir| names.iter().map(move |name| dir.join(name)))
        .find(|path| path.is_file())
}

fn find_cuda_root_nvrtc_library() -> Option<String> {
    find_cuda_root_nvrtc_library_from_roots(cuda_root_dirs())
}

fn find_cuda_root_nvrtc_library_from_roots(
    roots: impl IntoIterator<Item = PathBuf>,
) -> Option<String> {
    let roots = roots.into_iter().collect::<Vec<_>>();
    nvrtc_library_candidates_from_roots(roots.clone())
        .into_iter()
        .find(|path| path.is_file())
        .or_else(|| find_nvrtc_library_by_scanning_roots(roots))
        .map(|path| path.to_string_lossy().into_owned())
}

fn format_nvrtc_search_candidates_from_roots(roots: impl IntoIterator<Item = PathBuf>) -> String {
    format_path_candidates(nvrtc_library_candidates_from_roots(roots))
}

fn format_nvcc_search_candidates_from_roots(roots: impl IntoIterator<Item = PathBuf>) -> String {
    format_path_candidates(nvcc_command_candidates_from_roots(roots))
}

fn format_path_candidates(candidates: Vec<PathBuf>) -> String {
    if candidates.is_empty() {
        return "<none>".to_owned();
    }
    candidates
        .into_iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn find_nvrtc_library_by_scanning_roots(
    roots: impl IntoIterator<Item = PathBuf>,
) -> Option<PathBuf> {
    roots
        .into_iter()
        .flat_map(|root| nvrtc_library_dirs(&root))
        .filter_map(|dir| fs::read_dir(dir).ok())
        .flat_map(|entries| entries.filter_map(Result::ok))
        .map(|entry| entry.path())
        .find(|path| {
            path.is_file()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(is_nvrtc_library_name)
        })
}

fn is_nvrtc_library_name(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    #[cfg(windows)]
    {
        name.starts_with("nvrtc64_") && name.ends_with(".dll")
    }
    #[cfg(target_os = "linux")]
    {
        name == "libnvrtc.so" || name.starts_with("libnvrtc.so.")
    }
    #[cfg(target_os = "macos")]
    {
        name == "libnvrtc.dylib"
    }
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
    {
        let _ = name;
        false
    }
}

fn cuda_root_dirs() -> Vec<PathBuf> {
    let mut roots = ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"]
        .into_iter()
        .filter_map(std::env::var_os)
        .flat_map(|value| std::env::split_paths(&value).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    roots.extend(versioned_cuda_path_env_dirs());
    roots.extend(default_cuda_sdk_root_candidates());
    dedup_paths(roots)
}

fn versioned_cuda_path_env_dirs() -> Vec<PathBuf> {
    let mut values = std::env::vars_os()
        .filter_map(|(name, value)| {
            let name = name.to_string_lossy().to_ascii_uppercase();
            name.starts_with("CUDA_PATH_V")
                .then_some((name, value))
                .filter(|(_, value)| !value.is_empty())
        })
        .collect::<Vec<_>>();
    values.sort_by(|left, right| left.0.cmp(&right.0));
    values
        .into_iter()
        .flat_map(|(_, value)| std::env::split_paths(&value).collect::<Vec<_>>())
        .collect()
}

fn default_cuda_sdk_root_candidates() -> Vec<PathBuf> {
    #[cfg(windows)]
    {
        let bases = ["ProgramFiles", "ProgramFiles(x86)"]
            .into_iter()
            .filter_map(std::env::var_os)
            .map(PathBuf::from)
            .collect::<Vec<_>>();
        cuda_sdk_root_candidates_from_bases(bases)
    }
    #[cfg(target_os = "linux")]
    {
        let roots = [PathBuf::from("/usr/local/cuda")];
        dedup_paths(roots)
    }
    #[cfg(target_os = "macos")]
    {
        let roots = [PathBuf::from("/usr/local/cuda")];
        dedup_paths(roots)
    }
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
    {
        Vec::new()
    }
}

/// Returns CUDA SDK roots beneath standard installer base directories.
#[must_use]
pub fn cuda_sdk_root_candidates_from_bases(
    bases: impl IntoIterator<Item = PathBuf>,
) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for base in bases {
        let cuda_base = base.join("NVIDIA GPU Computing Toolkit").join("CUDA");
        let mut versioned_roots = Vec::new();
        if let Ok(entries) = fs::read_dir(&cuda_base) {
            versioned_roots.extend(
                entries
                    .filter_map(Result::ok)
                    .map(|entry| entry.path())
                    .filter(|path| path.is_dir())
                    .filter(|path| {
                        path.file_name()
                            .and_then(|name| name.to_str())
                            .is_some_and(|name| name.starts_with('v'))
                    }),
            );
        }
        versioned_roots.sort_by(|left, right| {
            cuda_version_key(right)
                .cmp(&cuda_version_key(left))
                .then_with(|| right.cmp(left))
        });
        roots.extend(versioned_roots);
        roots.push(cuda_base);
    }
    dedup_paths(roots)
}

fn cuda_version_key(path: &Path) -> Vec<u32> {
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

fn dedup_paths(paths: impl IntoIterator<Item = PathBuf>) -> Vec<PathBuf> {
    let mut deduped = Vec::new();
    for path in paths {
        if !deduped.iter().any(|existing| existing == &path) {
            deduped.push(path);
        }
    }
    deduped
}

fn nvrtc_library_dirs(root: &Path) -> Vec<PathBuf> {
    vec![
        root.join("bin"),
        root.join("lib64"),
        root.join("lib").join("x64"),
        root.join("lib"),
    ]
}

fn nvrtc_library_names() -> Vec<&'static str> {
    #[cfg(windows)]
    {
        vec![
            "nvrtc64_130_0.dll",
            "nvrtc64_120_0.dll",
            "nvrtc64_122_0.dll",
            "nvrtc64_121_0.dll",
            "nvrtc64_112_0.dll",
            "nvrtc64_111_0.dll",
            "nvrtc64_110_0.dll",
        ]
    }
    #[cfg(target_os = "linux")]
    {
        vec!["libnvrtc.so.12", "libnvrtc.so.11", "libnvrtc.so"]
    }
    #[cfg(target_os = "macos")]
    {
        vec!["libnvrtc.dylib"]
    }
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
    {
        Vec::new()
    }
}

#[cfg(windows)]
fn find_windows_nvrtc_library() -> Option<String> {
    std::env::var_os("PATH")
        .into_iter()
        .flat_map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
        .filter_map(|dir| fs::read_dir(dir).ok())
        .flat_map(|entries| entries.filter_map(Result::ok))
        .map(|entry| entry.path())
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    let name = name.to_ascii_lowercase();
                    name.starts_with("nvrtc64_") && name.ends_with(".dll")
                })
        })
        .map(|path| path.to_string_lossy().into_owned())
}

impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        unsafe {
            platform_close(self.handle);
        }
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
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn restore_env(name: &str, original: Option<std::ffi::OsString>) {
        if let Some(value) = original {
            std::env::set_var(name, value);
        } else {
            std::env::remove_var(name);
        }
    }

    #[test]
    fn cuda_artifact_marker_selects_u32_launch_abi() {
        assert!(uses_u32_launch_abi(
            ".visible .global .u32 atlas_search_u32_abi;"
        ));
        assert!(uses_u32_launch_abi(
            "extern \"C\" __device__ __constant__ unsigned int atlas_search_u32_abi = 1U;"
        ));
        assert!(!uses_u32_launch_abi(
            ".visible .entry atlas_search(.param .u64 start)"
        ));
    }

    #[test]
    fn cuda_features_report_int64_only_after_successful_probe() {
        assert_eq!(features_from_int64_probe(Ok(())), vec!["int64".to_owned()]);
        assert_eq!(
            features_from_int64_probe(Err("module load failed".to_owned())),
            Vec::<String>::new()
        );
    }

    #[test]
    fn cuda_int64_probe_source_uses_self_contained_64_bit_kernel() {
        let source = cuda_int64_probe_source();

        assert!(source.contains("atlas_int64_probe"));
        assert!(source.contains("unsigned long long"));
        assert!(source.contains("__nvvm_read_ptx_sreg_tid_x"));
        assert!(source.contains("blockIdx.x"));
    }

    #[test]
    fn cuda_compiler_diagnostics_report_injected_sdk_candidates() {
        let cuda_root = PathBuf::from("C:/atlas-test/cuda");

        let nvrtc = format_nvrtc_search_candidates_from_roots([cuda_root.clone()]);
        let nvcc = format_nvcc_search_candidates_from_roots([cuda_root.clone()]);

        assert!(nvrtc.contains("atlas-test"));
        assert!(nvrtc.contains("nvrtc"));
        assert!(nvcc.contains("atlas-test"));
        assert!(nvcc.contains("nvcc"));
    }

    #[test]
    fn cuda_root_nvrtc_discovery_accepts_newer_runtime_compiler_names() {
        let cuda_root =
            std::env::temp_dir().join(format!("atlas-cuda-new-nvrtc-{}", std::process::id()));
        let bin_dir = cuda_root.join("bin");
        std::fs::create_dir_all(&bin_dir).unwrap();
        let library = bin_dir.join("nvrtc64_999_0.dll");
        std::fs::write(&library, []).unwrap();

        let found = find_cuda_root_nvrtc_library_from_roots([cuda_root.clone()]);

        assert_eq!(found, Some(library.to_string_lossy().into_owned()));
        let _ = std::fs::remove_dir_all(cuda_root);
    }

    #[test]
    fn cuda_root_dirs_include_versioned_cuda_path_env_vars() {
        let _env_guard = env_lock();
        let cuda_root =
            std::env::temp_dir().join(format!("atlas-cuda-versioned-env-{}", std::process::id()));
        std::fs::create_dir_all(&cuda_root).unwrap();
        let original_cuda_path = std::env::var_os("CUDA_PATH");
        let original_cuda_home = std::env::var_os("CUDA_HOME");
        let original_cuda_root = std::env::var_os("CUDA_ROOT");
        let original_cuda_path_v12_4 = std::env::var_os("CUDA_PATH_V12_4");
        std::env::remove_var("CUDA_PATH");
        std::env::remove_var("CUDA_HOME");
        std::env::remove_var("CUDA_ROOT");
        std::env::set_var("CUDA_PATH_V12_4", &cuda_root);

        let roots = cuda_root_dirs();

        restore_env("CUDA_PATH", original_cuda_path);
        restore_env("CUDA_HOME", original_cuda_home);
        restore_env("CUDA_ROOT", original_cuda_root);
        restore_env("CUDA_PATH_V12_4", original_cuda_path_v12_4);
        let _ = std::fs::remove_dir_all(&cuda_root);
        assert!(
            roots.contains(&cuda_root),
            "expected CUDA_PATH_V12_4 root in {roots:?}"
        );
    }

    #[test]
    fn cuda_driver_load_errors_report_searched_driver_candidates() {
        let cuda_root = PathBuf::from("C:/atlas-test/cuda");

        let error = cuda_driver_load_error(
            "failed to load CUDA driver library nvcuda.dll",
            [cuda_root.clone()],
        );

        assert!(error.contains("failed to load CUDA driver library"));
        assert!(error.contains("searched CUDA driver candidates"));
        assert!(error.contains("atlas-test"));
        assert!(error.contains("nvcuda"));
    }

    #[cfg(windows)]
    #[test]
    fn cuda_driver_load_errors_report_windows_display_driver_candidates() {
        let original_system_root = std::env::var_os("SystemRoot");
        std::env::set_var("SystemRoot", "C:\\Windows");

        let error = cuda_driver_load_error("failed to load CUDA driver library nvcuda.dll", []);

        assert!(error.contains("C:\\Windows\\System32\\nvcuda.dll"));

        if let Some(value) = original_system_root {
            std::env::set_var("SystemRoot", value);
        } else {
            std::env::remove_var("SystemRoot");
        }
    }
}
