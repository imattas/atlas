//! CUDA launch adapter.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::fs;
use std::path::{Path, PathBuf};
use std::ptr;

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
}

/// Adapter CLI command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterCommand {
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

/// Launches one parsed CUDA request.
pub trait Launcher {
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
    fn launch(&self, args: &LaunchArgs) -> Result<Vec<u64>, String>;
}

/// CUDA Driver API backed PTX launcher.
#[derive(Debug, Clone, Copy, Default)]
pub struct CudaPtxLauncher;

impl Launcher for CudaPtxLauncher {
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

    fn launch(&self, args: &LaunchArgs) -> Result<Vec<u64>, String> {
        let ptx = read_cuda_artifact_as_ptx(&args.artifact)?;
        ensure_atlas_entry(&ptx)?;
        let driver = CudaDriver::load()?;
        let _context = driver.create_context()?;
        let ptx = CString::new(ptx).map_err(|_| "PTX contains interior NUL".to_owned())?;
        let module = driver.load_module(&ptx)?;
        let function = driver.get_function(module.raw, "atlas_search")?;
        launch_cuda(&driver, function, args)
    }
}

/// Runs adapter CLI logic with an injected launcher.
///
/// # Errors
///
/// Returns parse or launcher errors.
pub fn run_cli(args: &[String], launcher: &dyn Launcher) -> Result<String, String> {
    match AdapterCommand::parse(args)? {
        AdapterCommand::CompileCheck { input, output } => {
            launcher.compile_check(&input, output.as_deref())?;
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
    NvrtcCompiler::load()?.compile_source_to_ptx(&source)
}

fn ensure_atlas_entry(ptx: &str) -> Result<(), String> {
    if ptx.contains(".entry atlas_search") {
        Ok(())
    } else {
        Err("missing atlas_search kernel entry".to_owned())
    }
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

fn launch_cuda(
    driver: &CudaDriver,
    function: CuFunction,
    args: &LaunchArgs,
) -> Result<Vec<u64>, String> {
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
    let retained = usize::try_from(retained)
        .unwrap_or(args.max_matches)
        .min(args.max_matches);
    let mut matches = vec![0_u64; args.max_matches];
    driver.memcpy_dtoh(matches.as_mut_ptr().cast::<c_void>(), out, out_bytes)?;
    matches.truncate(retained);
    matches.sort_unstable();
    Ok(matches)
}

#[derive(Debug)]
struct NvrtcCompiler {
    _library: DynamicLibrary,
    api: NvrtcApi,
}

impl NvrtcCompiler {
    fn load() -> Result<Self, String> {
        let library = DynamicLibrary::open_nvrtc()?;
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
        let library = DynamicLibrary::open_cuda()?;
        let api = unsafe { CudaApi::load(&library)? };
        let driver = Self {
            _library: library,
            api,
        };
        driver.check(unsafe { (driver.api.cu_init)(0) }, "cuInit")?;
        Ok(driver)
    }

    fn create_context(&self) -> Result<CudaContext<'_>, String> {
        let mut device = 0;
        self.check(
            unsafe { (self.api.cu_device_get)(&mut device, 0) },
            "cuDeviceGet",
        )?;
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

#[derive(Debug)]
struct CudaApi {
    cu_init: unsafe extern "C" fn(c_uint) -> CuResult,
    cu_device_get: unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult,
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
            Self::open("nvcuda.dll")
        }
        #[cfg(target_os = "linux")]
        {
            Self::open("libcuda.so.1").or_else(|_| Self::open("libcuda.so"))
        }
        #[cfg(target_os = "macos")]
        {
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

fn find_cuda_root_nvrtc_library() -> Option<String> {
    nvrtc_library_candidates_from_roots(cuda_root_dirs())
        .into_iter()
        .find(|path| path.is_file())
        .map(|path| path.to_string_lossy().into_owned())
}

fn cuda_root_dirs() -> Vec<PathBuf> {
    ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"]
        .into_iter()
        .filter_map(std::env::var_os)
        .flat_map(|value| std::env::split_paths(&value).collect::<Vec<_>>())
        .collect()
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
