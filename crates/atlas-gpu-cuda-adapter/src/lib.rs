//! CUDA launch adapter.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::fs;
use std::ptr;

type CuDevice = c_int;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuDevicePtr = u64;
type CuResult = c_int;

const CUDA_SUCCESS: CuResult = 0;

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
        /// Generated PTX path.
        ptx: String,
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
            return Ok(Self::CompileCheck { ptx: ptx.clone() });
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
    fn compile_check(&self, ptx: &str) -> Result<(), String>;

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
    fn compile_check(&self, ptx: &str) -> Result<(), String> {
        let ptx = read_ptx(ptx)?;
        ensure_atlas_entry(&ptx)?;
        if let Ok(driver) = CudaDriver::load() {
            let _context = driver.create_context()?;
            let ptx = CString::new(ptx).map_err(|_| "PTX contains interior NUL".to_owned())?;
            let module = driver.load_module(&ptx)?;
            driver.get_function(module.raw, "atlas_search")?;
        }
        Ok(())
    }

    fn launch(&self, args: &LaunchArgs) -> Result<Vec<u64>, String> {
        let ptx = read_ptx(&args.artifact)?;
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
        AdapterCommand::CompileCheck { ptx } => {
            launcher.compile_check(&ptx)?;
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
