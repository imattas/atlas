//! Vulkan launch adapter.

use ash::{vk, Entry};
use std::ffi::CString;
use std::fs;
use std::path::{Path, PathBuf};
use std::ptr;

const SPIRV_MAGIC: u32 = 0x0723_0203;
const OUT_VALUES_OFFSET: usize = 8;
const VULKAN_LOCAL_SIZE: usize = 256;

/// Parsed Vulkan launch protocol arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchArgs {
    /// Generated SPIR-V compute shader artifact.
    pub artifact: String,
    /// Inclusive search start.
    pub start: u64,
    /// Exclusive search end.
    pub end: u64,
    /// Maximum number of retained matches.
    pub max_matches: usize,
    /// Global Vulkan invocation count.
    pub global_size: usize,
    /// Vulkan shader local size. The generated shader currently uses 256.
    pub local_size: usize,
}

/// Adapter CLI command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterCommand {
    /// Shader-module-check a SPIR-V file without launching a search.
    CompileCheck {
        /// Generated GLSL or SPIR-V path.
        input: String,
        /// Optional compiled SPIR-V output path.
        output: Option<String>,
    },
    /// Launch a Vulkan search.
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
            let Some(spirv) = args.get(1) else {
                return Err("missing compile-check SPIR-V".to_owned());
            };
            return Ok(Self::CompileCheck {
                input: spirv.clone(),
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

/// Launches one parsed Vulkan request.
pub trait Launcher {
    /// Checks generated SPIR-V has valid magic and can be loaded as a shader
    /// module when a Vulkan runtime is present.
    ///
    /// # Errors
    ///
    /// Returns an error when the SPIR-V artifact is malformed or Vulkan module
    /// loading fails.
    fn compile_check(&self, input: &str, output: Option<&str>) -> Result<(), String>;

    /// Runs the launch and returns device-reported matches.
    ///
    /// # Errors
    ///
    /// Returns an error when Vulkan runtime loading, pipeline creation, dispatch,
    /// or memory transfer fails.
    fn launch(&self, args: &LaunchArgs) -> Result<Vec<u64>, String>;
}

/// Vulkan runtime backed SPIR-V launcher.
#[derive(Debug, Clone, Copy, Default)]
pub struct VulkanSpirvLauncher;

impl Launcher for VulkanSpirvLauncher {
    fn compile_check(&self, shader: &str, output: Option<&str>) -> Result<(), String> {
        let code = read_shader_words(shader)?;
        if let Some(output) = output {
            write_spirv_words(output, &code)?;
        }
        if let Ok(runtime) = VulkanRuntime::new() {
            let _shader = runtime.create_shader_module(&code)?;
        }
        Ok(())
    }

    fn launch(&self, args: &LaunchArgs) -> Result<Vec<u64>, String> {
        if args.local_size != VULKAN_LOCAL_SIZE {
            return Err(format!(
                "Vulkan shader local-size must be {VULKAN_LOCAL_SIZE}"
            ));
        }
        let code = read_shader_words(&args.artifact)?;
        let runtime = VulkanRuntime::new()?;
        runtime.launch(&code, args)
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

fn read_spirv_words(path: &str) -> Result<Vec<u32>, String> {
    let bytes = fs::read(path).map_err(|error| format!("cannot read SPIR-V {path}: {error}"))?;
    if bytes.len() < 4 {
        return Err("invalid SPIR-V byte length".to_owned());
    }
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != SPIRV_MAGIC {
        return Err("invalid SPIR-V magic".to_owned());
    }
    if bytes.len() % 4 != 0 {
        return Err("invalid SPIR-V byte length".to_owned());
    }
    let words = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    Ok(words)
}

fn read_shader_words(path: &str) -> Result<Vec<u32>, String> {
    if Path::new(path)
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("spv"))
    {
        return read_spirv_words(path);
    }
    let source = fs::read_to_string(path)
        .map_err(|error| format!("cannot read Vulkan GLSL {path}: {error}"))?;
    compile_glsl_to_spirv(&source)
}

fn write_spirv_words(path: &str, words: &[u32]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(words.len().saturating_mul(4));
    for word in words {
        bytes.extend(word.to_le_bytes());
    }
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create SPIR-V output: {error}"))?;
    }
    fs::write(path, bytes).map_err(|error| format!("cannot write SPIR-V {path}: {error}"))
}

/// Compiles generated Vulkan GLSL compute source to SPIR-V words.
///
/// # Errors
///
/// Returns an error when shaderc cannot initialize or the generated GLSL is
/// rejected by the Vulkan GLSL compiler.
pub fn compile_glsl_to_spirv(source: &str) -> Result<Vec<u32>, String> {
    let compiler = shaderc::Compiler::new()
        .map_err(|error| format!("cannot initialize shaderc compiler: {error}"))?;
    let mut options = shaderc::CompileOptions::new()
        .map_err(|error| format!("cannot initialize shaderc compile options: {error}"))?;
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_1 as u32,
    );
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);
    let artifact = compiler
        .compile_into_spirv(
            source,
            shaderc::ShaderKind::Compute,
            "atlas_search.comp",
            "main",
            Some(&options),
        )
        .map_err(|error| error.to_string())?;
    Ok(artifact.as_binary().to_vec())
}

/// Returns Vulkan loader library candidates under explicit SDK root paths.
///
/// The Vulkan loader is often present under `VULKAN_SDK` even when the process
/// environment has not been configured so the system dynamic loader can find it.
pub fn vulkan_loader_candidates_from_roots(
    roots: impl IntoIterator<Item = PathBuf>,
) -> Vec<PathBuf> {
    roots
        .into_iter()
        .flat_map(|root| {
            vulkan_loader_dirs(&root).into_iter().flat_map(|dir| {
                vulkan_loader_names()
                    .into_iter()
                    .map(move |name| dir.join(name))
            })
        })
        .collect()
}

/// Returns Vulkan loader library candidates from host environment and standard
/// SDK install roots.
#[must_use]
pub fn vulkan_loader_candidates_from_host_roots() -> Vec<PathBuf> {
    vulkan_loader_candidates_from_roots(vulkan_loader_roots_from_host())
}

fn vulkan_loader_roots_from_host() -> Vec<PathBuf> {
    let mut roots = vulkan_loader_roots_from_env();
    roots.extend(vulkan_standard_loader_roots());
    dedup_paths(roots)
}

fn vulkan_loader_roots_from_env() -> Vec<PathBuf> {
    ["VULKAN_SDK", "VK_SDK_PATH"]
        .into_iter()
        .filter_map(std::env::var_os)
        .flat_map(|value| std::env::split_paths(&value).collect::<Vec<_>>())
        .collect()
}

fn vulkan_standard_loader_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    #[cfg(windows)]
    {
        if let Some(drive) = std::env::var_os("SystemDrive").map(PathBuf::from) {
            let vulkan_base = drive.join("VulkanSDK");
            let mut versioned_roots = Vec::new();
            if let Ok(entries) = fs::read_dir(&vulkan_base) {
                versioned_roots.extend(
                    entries
                        .filter_map(Result::ok)
                        .map(|entry| entry.path())
                        .filter(|path| path.is_dir()),
                );
            }
            versioned_roots.sort_by(|left, right| {
                sdk_version_key(right)
                    .cmp(&sdk_version_key(left))
                    .then_with(|| right.cmp(left))
            });
            roots.extend(versioned_roots);
            push_existing_dir(&mut roots, vulkan_base.clone());
        }
    }
    #[cfg(not(windows))]
    {
        roots.push(PathBuf::from("/usr"));
        roots.push(PathBuf::from("/usr/local"));
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

fn dedup_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut deduped = Vec::new();
    for path in paths {
        if !deduped.iter().any(|existing| existing == &path) {
            deduped.push(path);
        }
    }
    deduped
}

fn vulkan_loader_dirs(root: &Path) -> Vec<PathBuf> {
    vec![
        root.join("Bin"),
        root.join("Bin32"),
        root.join("bin"),
        root.join("lib"),
        root.join("lib64"),
    ]
}

fn vulkan_loader_names() -> Vec<&'static str> {
    if cfg!(target_os = "windows") {
        vec!["vulkan-1.dll"]
    } else if cfg!(target_os = "macos") {
        vec!["libvulkan.dylib", "libMoltenVK.dylib"]
    } else {
        vec!["libvulkan.so.1", "libvulkan.so"]
    }
}

fn load_vulkan_entry() -> Result<Entry, String> {
    for candidate in vulkan_loader_candidates_from_host_roots() {
        if !candidate.is_file() {
            continue;
        }
        match unsafe { Entry::load_from(&candidate) } {
            Ok(entry) => return Ok(entry),
            Err(_) => continue,
        }
    }
    unsafe { Entry::load().map_err(|error| error.to_string()) }
}

struct VulkanRuntime {
    _entry: Entry,
    instance: ash::Instance,
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    queue: vk::Queue,
}

impl VulkanRuntime {
    fn new() -> Result<Self, String> {
        let entry = load_vulkan_entry()?;
        let app_name = CString::new("atlas-gpu-vulkan-run").map_err(|error| error.to_string())?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(0)
            .engine_name(&app_name)
            .engine_version(0)
            .api_version(vk::API_VERSION_1_1);
        let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe {
            entry
                .create_instance(&instance_info, None)
                .map_err(|error| format!("vkCreateInstance failed: {error:?}"))?
        };
        let (physical_device, queue_family_index) = select_compute_queue(&instance)?;
        let priorities = [1.0_f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities)];
        let features = vk::PhysicalDeviceFeatures::default();
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_features(&features);
        let device = unsafe {
            instance
                .create_device(physical_device, &device_info, None)
                .map_err(|error| format!("vkCreateDevice failed: {error:?}"))?
        };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        Ok(Self {
            _entry: entry,
            instance,
            device,
            physical_device,
            queue_family_index,
            queue,
        })
    }

    fn create_shader_module(&self, code: &[u32]) -> Result<ShaderModule<'_>, String> {
        let create_info = vk::ShaderModuleCreateInfo::default().code(code);
        let module = unsafe {
            self.device
                .create_shader_module(&create_info, None)
                .map_err(|error| format!("vkCreateShaderModule failed: {error:?}"))?
        };
        Ok(ShaderModule {
            device: &self.device,
            raw: module,
        })
    }

    fn launch(&self, code: &[u32], args: &LaunchArgs) -> Result<Vec<u64>, String> {
        let shader = self.create_shader_module(code)?;
        let buffer_bytes = OUT_VALUES_OFFSET
            .checked_add(
                args.max_matches
                    .checked_mul(std::mem::size_of::<u64>())
                    .ok_or_else(|| "output buffer size overflow".to_owned())?,
            )
            .ok_or_else(|| "output buffer size overflow".to_owned())?;
        let buffer = self.create_host_buffer(buffer_bytes)?;
        buffer.zero()?;
        let descriptor_layout = self.create_descriptor_set_layout()?;
        let pipeline_layout = self.create_pipeline_layout(descriptor_layout.raw)?;
        let pipeline = self.create_compute_pipeline(shader.raw, pipeline_layout.raw)?;
        let descriptor_pool = self.create_descriptor_pool()?;
        let descriptor_set =
            self.allocate_descriptor_set(descriptor_pool.raw, descriptor_layout.raw, buffer.raw);
        let command_pool = self.create_command_pool()?;
        let command_buffer = self.allocate_command_buffer(command_pool.raw)?;

        self.record_dispatch(
            command_buffer,
            pipeline.raw,
            pipeline_layout.raw,
            descriptor_set?,
            buffer.raw,
            buffer_bytes,
            args,
        )?;
        self.submit_and_wait(command_buffer)?;
        buffer.read_matches(args.max_matches)
    }

    fn create_host_buffer(&self, bytes: usize) -> Result<HostBuffer<'_>, String> {
        let size = u64::try_from(bytes).map_err(|_| "buffer size exceeds Vulkan u64".to_owned())?;
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe {
            self.device
                .create_buffer(&buffer_info, None)
                .map_err(|error| format!("vkCreateBuffer failed: {error:?}"))?
        };
        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let memory_type_index = self.find_memory_type(
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);
        let memory = unsafe {
            self.device
                .allocate_memory(&allocate_info, None)
                .map_err(|error| format!("vkAllocateMemory failed: {error:?}"))?
        };
        unsafe {
            self.device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(|error| format!("vkBindBufferMemory failed: {error:?}"))?;
        }
        Ok(HostBuffer {
            device: &self.device,
            raw: buffer,
            memory,
            bytes,
        })
    }

    fn find_memory_type(
        &self,
        type_bits: u32,
        required: vk::MemoryPropertyFlags,
    ) -> Result<u32, String> {
        let properties = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        (0..properties.memory_type_count)
            .find(|index| {
                let supported = (type_bits & (1_u32 << index)) != 0;
                let flags = properties.memory_types[*index as usize].property_flags;
                supported && flags.contains(required)
            })
            .ok_or_else(|| "no host-visible coherent Vulkan storage memory type".to_owned())
    }

    fn create_descriptor_set_layout(&self) -> Result<DescriptorSetLayout<'_>, String> {
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let raw = unsafe {
            self.device
                .create_descriptor_set_layout(&create_info, None)
                .map_err(|error| format!("vkCreateDescriptorSetLayout failed: {error:?}"))?
        };
        Ok(DescriptorSetLayout {
            device: &self.device,
            raw,
        })
    }

    fn create_pipeline_layout(
        &self,
        descriptor_layout: vk::DescriptorSetLayout,
    ) -> Result<PipelineLayout<'_>, String> {
        let set_layouts = [descriptor_layout];
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(24)];
        let create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);
        let raw = unsafe {
            self.device
                .create_pipeline_layout(&create_info, None)
                .map_err(|error| format!("vkCreatePipelineLayout failed: {error:?}"))?
        };
        Ok(PipelineLayout {
            device: &self.device,
            raw,
        })
    }

    fn create_compute_pipeline(
        &self,
        shader: vk::ShaderModule,
        layout: vk::PipelineLayout,
    ) -> Result<Pipeline<'_>, String> {
        let entry_name = CString::new("main").map_err(|error| error.to_string())?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(&entry_name);
        let create_info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout)];
        let raw = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &create_info, None)
                .map_err(|(_, error)| format!("vkCreateComputePipelines failed: {error:?}"))?
        }[0];
        Ok(Pipeline {
            device: &self.device,
            raw,
        })
    }

    fn create_descriptor_pool(&self) -> Result<DescriptorPool<'_>, String> {
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)];
        let create_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let raw = unsafe {
            self.device
                .create_descriptor_pool(&create_info, None)
                .map_err(|error| format!("vkCreateDescriptorPool failed: {error:?}"))?
        };
        Ok(DescriptorPool {
            device: &self.device,
            raw,
        })
    }

    fn allocate_descriptor_set(
        &self,
        pool: vk::DescriptorPool,
        layout: vk::DescriptorSetLayout,
        buffer: vk::Buffer,
    ) -> Result<vk::DescriptorSet, String> {
        let layouts = [layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        let descriptor_set = unsafe {
            self.device
                .allocate_descriptor_sets(&allocate_info)
                .map_err(|error| format!("vkAllocateDescriptorSets failed: {error:?}"))?
        }[0];
        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let writes = [vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_info)];
        unsafe {
            self.device.update_descriptor_sets(&writes, &[]);
        }
        Ok(descriptor_set)
    }

    fn create_command_pool(&self) -> Result<CommandPool<'_>, String> {
        let create_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(self.queue_family_index);
        let raw = unsafe {
            self.device
                .create_command_pool(&create_info, None)
                .map_err(|error| format!("vkCreateCommandPool failed: {error:?}"))?
        };
        Ok(CommandPool {
            device: &self.device,
            raw,
        })
    }

    fn allocate_command_buffer(&self, pool: vk::CommandPool) -> Result<vk::CommandBuffer, String> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        Ok(unsafe {
            self.device
                .allocate_command_buffers(&allocate_info)
                .map_err(|error| format!("vkAllocateCommandBuffers failed: {error:?}"))?
        }[0])
    }

    #[allow(clippy::too_many_arguments)]
    fn record_dispatch(
        &self,
        command_buffer: vk::CommandBuffer,
        pipeline: vk::Pipeline,
        pipeline_layout: vk::PipelineLayout,
        descriptor_set: vk::DescriptorSet,
        buffer: vk::Buffer,
        buffer_bytes: usize,
        args: &LaunchArgs,
    ) -> Result<(), String> {
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .map_err(|error| format!("vkBeginCommandBuffer failed: {error:?}"))?;
            self.device
                .cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            let push_constants = push_constant_bytes(args.start, args.end, args.max_matches)?;
            self.device.cmd_push_constants(
                command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push_constants,
            );
            let group_count = args.global_size.div_ceil(VULKAN_LOCAL_SIZE);
            let group_count =
                u32::try_from(group_count).map_err(|_| "dispatch group count exceeds u32")?;
            self.device.cmd_dispatch(command_buffer, group_count, 1, 1);
            let barrier = [vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .buffer(buffer)
                .offset(0)
                .size(u64::try_from(buffer_bytes).map_err(|_| "buffer size exceeds u64")?)];
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &barrier,
                &[],
            );
            self.device
                .end_command_buffer(command_buffer)
                .map_err(|error| format!("vkEndCommandBuffer failed: {error:?}"))?;
        }
        Ok(())
    }

    fn submit_and_wait(&self, command_buffer: vk::CommandBuffer) -> Result<(), String> {
        let command_buffers = [command_buffer];
        let submits = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
        unsafe {
            self.device
                .queue_submit(self.queue, &submits, vk::Fence::null())
                .map_err(|error| format!("vkQueueSubmit failed: {error:?}"))?;
            self.device
                .queue_wait_idle(self.queue)
                .map_err(|error| format!("vkQueueWaitIdle failed: {error:?}"))?;
        }
        Ok(())
    }
}

impl Drop for VulkanRuntime {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn select_compute_queue(instance: &ash::Instance) -> Result<(vk::PhysicalDevice, u32), String> {
    let devices = unsafe {
        instance
            .enumerate_physical_devices()
            .map_err(|error| format!("vkEnumeratePhysicalDevices failed: {error:?}"))?
    };
    devices
        .into_iter()
        .find_map(|physical_device| {
            let queues =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
            queues
                .iter()
                .enumerate()
                .find(|(_, properties)| properties.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .and_then(|(index, _)| u32::try_from(index).ok())
                .map(|index| (physical_device, index))
        })
        .ok_or_else(|| "no Vulkan compute queue found".to_owned())
}

fn push_constant_bytes(start: u64, end: u64, max_matches: usize) -> Result<[u8; 24], String> {
    let max_matches =
        u32::try_from(max_matches).map_err(|_| "max-matches exceeds Vulkan uint".to_owned())?;
    let mut bytes = [0_u8; 24];
    bytes[0..8].copy_from_slice(&start.to_le_bytes());
    bytes[8..16].copy_from_slice(&end.to_le_bytes());
    bytes[16..20].copy_from_slice(&max_matches.to_le_bytes());
    Ok(bytes)
}

struct HostBuffer<'a> {
    device: &'a ash::Device,
    raw: vk::Buffer,
    memory: vk::DeviceMemory,
    bytes: usize,
}

impl HostBuffer<'_> {
    fn zero(&self) -> Result<(), String> {
        unsafe {
            let ptr = self.map()?;
            ptr::write_bytes(ptr.cast::<u8>(), 0, self.bytes);
            self.device.unmap_memory(self.memory);
        }
        Ok(())
    }

    fn read_matches(&self, max_matches: usize) -> Result<Vec<u64>, String> {
        unsafe {
            let ptr = self.map()?.cast::<u8>();
            let retained = ptr.cast::<u32>().read_unaligned();
            let retained = usize::try_from(retained)
                .unwrap_or(max_matches)
                .min(max_matches);
            let mut matches = Vec::with_capacity(retained);
            for index in 0..retained {
                let offset = OUT_VALUES_OFFSET + index * std::mem::size_of::<u64>();
                matches.push(ptr.add(offset).cast::<u64>().read_unaligned());
            }
            self.device.unmap_memory(self.memory);
            matches.sort_unstable();
            Ok(matches)
        }
    }

    unsafe fn map(&self) -> Result<*mut std::ffi::c_void, String> {
        let size = u64::try_from(self.bytes).map_err(|_| "buffer size exceeds u64".to_owned())?;
        self.device
            .map_memory(self.memory, 0, size, vk::MemoryMapFlags::empty())
            .map_err(|error| format!("vkMapMemory failed: {error:?}"))
    }
}

impl Drop for HostBuffer<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.raw, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

struct ShaderModule<'a> {
    device: &'a ash::Device,
    raw: vk::ShaderModule,
}

impl Drop for ShaderModule<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_shader_module(self.raw, None);
        }
    }
}

struct DescriptorSetLayout<'a> {
    device: &'a ash::Device,
    raw: vk::DescriptorSetLayout,
}

impl Drop for DescriptorSetLayout<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_set_layout(self.raw, None);
        }
    }
}

struct PipelineLayout<'a> {
    device: &'a ash::Device,
    raw: vk::PipelineLayout,
}

impl Drop for PipelineLayout<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline_layout(self.raw, None);
        }
    }
}

struct Pipeline<'a> {
    device: &'a ash::Device,
    raw: vk::Pipeline,
}

impl Drop for Pipeline<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.raw, None);
        }
    }
}

struct DescriptorPool<'a> {
    device: &'a ash::Device,
    raw: vk::DescriptorPool,
}

impl Drop for DescriptorPool<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_pool(self.raw, None);
        }
    }
}

struct CommandPool<'a> {
    device: &'a ash::Device,
    raw: vk::CommandPool,
}

impl Drop for CommandPool<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.raw, None);
        }
    }
}
