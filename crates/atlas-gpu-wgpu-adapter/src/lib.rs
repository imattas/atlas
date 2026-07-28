//! WGPU/WebGPU launch adapter.

use std::fs;
use std::sync::mpsc;

const OUTPUT_HEADER_U32S: usize = 2;
const CANDIDATE_U32S: usize = 2;

/// Parsed WGPU launch protocol arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchArgs {
    /// Generated WGSL artifact.
    pub artifact: String,
    /// Inclusive search start.
    pub start: u64,
    /// Exclusive search end.
    pub end: u64,
    /// Maximum number of retained matches.
    pub max_matches: usize,
    /// Global invocation count.
    pub global_size: usize,
    /// Workgroup size encoded in the generated WGSL.
    pub local_size: usize,
}

/// Device launch output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchOutput {
    /// Retained device-reported matches.
    pub matches: Vec<u64>,
    /// Total device-side match count.
    pub match_count: usize,
}

/// Adapter CLI command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterCommand {
    /// Report runtime/device features understood by this adapter.
    Features,
    /// Validate a WGSL source file without launching a search.
    CompileCheck {
        /// Generated WGSL source path.
        source: String,
        /// Optional checked WGSL output path.
        output: Option<String>,
    },
    /// Launch a WGPU search.
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
                return Err("missing compile-check WGSL source".to_owned());
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
    /// Returns an error when required arguments are missing, malformed, or invalid.
    pub fn parse(args: &[String]) -> Result<Self, String> {
        let Some(artifact) = args.first() else {
            return Err("missing WGSL artifact".to_owned());
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
            return Err("max-matches exceeds WGPU uint".to_owned());
        }
        let global_size = parse_usize_flag(args, "--global-size")?;
        let local_size = parse_usize_flag(args, "--local-size")?;
        if global_size == 0 || local_size == 0 {
            return Err("global-size and local-size must be nonzero".to_owned());
        }
        if global_size < usize::try_from(end.saturating_sub(start)).unwrap_or(usize::MAX) {
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

/// Launches one parsed WGPU request.
pub trait Launcher {
    /// Reports runtime/device features available to generated kernels.
    ///
    /// # Errors
    ///
    /// Returns an error when feature discovery fails.
    fn features(&self) -> Result<Vec<String>, String>;

    /// Validates generated WGSL.
    ///
    /// # Errors
    ///
    /// Returns an error when source cannot be read or parsed as WGSL.
    fn compile_check(&self, source: &str, output: Option<&str>) -> Result<(), String>;

    /// Runs the launch and returns device-reported matches.
    ///
    /// # Errors
    ///
    /// Returns an error when the WGPU device path is unavailable or fails.
    fn launch(&self, args: &LaunchArgs) -> Result<LaunchOutput, String>;
}

/// WGPU-backed launcher.
#[derive(Debug, Clone, Copy, Default)]
pub struct WgpuLauncher;

impl Launcher for WgpuLauncher {
    fn features(&self) -> Result<Vec<String>, String> {
        pollster::block_on(check_wgpu_runtime())?;
        Ok(vec!["launchAbiU32".to_owned()])
    }

    fn compile_check(&self, source: &str, output: Option<&str>) -> Result<(), String> {
        let source_text = fs::read_to_string(source)
            .map_err(|error| format!("cannot read WGSL source {source}: {error}"))?;
        validate_wgsl(&source_text)?;
        if let Some(output) = output {
            write_checked_wgsl(output, &source_text)?;
        }
        Ok(())
    }

    fn launch(&self, args: &LaunchArgs) -> Result<LaunchOutput, String> {
        pollster::block_on(launch_wgpu(args))
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
            let output = launcher.launch(&launch_args)?;
            Ok(format_launch_output(&output))
        }
    }
}

fn validate_wgsl(source: &str) -> Result<(), String> {
    naga::front::wgsl::parse_str(source).map_err(|error| format!("invalid WGSL: {error}"))?;
    Ok(())
}

/// Validates that generated WGSL encodes the launch shape requested by the host.
///
/// # Errors
///
/// Returns an error when the WGSL is invalid, has no workgroup size annotation,
/// or the encoded local size differs from the launch protocol local size.
pub fn validate_wgsl_launch_shape(source: &str, local_size: usize) -> Result<(), String> {
    validate_wgsl(source)?;
    let encoded_local_size = extract_wgsl_workgroup_size_x(source)?;
    if encoded_local_size != local_size {
        return Err(format!(
            "WGPU local-size mismatch: WGSL encodes {encoded_local_size}, launch requested {local_size}"
        ));
    }
    Ok(())
}

async fn launch_wgpu(args: &LaunchArgs) -> Result<LaunchOutput, String> {
    let source = fs::read_to_string(&args.artifact)
        .map_err(|error| format!("cannot read WGSL artifact {}: {error}", args.artifact))?;
    validate_wgsl_launch_shape(&source, args.local_size)?;

    let (device, queue) = create_wgpu_device().await?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("atlas-search-wgsl"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let output_bytes = u64::try_from((OUTPUT_HEADER_U32S + args.max_matches * CANDIDATE_U32S) * 4)
        .map_err(|_| "output buffer size exceeds WGPU address space".to_owned())?;
    let params = encode_params(args)?;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("atlas-search-output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("atlas-search-params"),
        size: u64::try_from(params.len())
            .map_err(|_| "params buffer size exceeds WGPU address space".to_owned())?,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("atlas-search-readback"),
        size: output_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    queue.write_buffer(
        &output_buffer,
        0,
        &vec![
            0;
            usize::try_from(output_bytes)
                .map_err(|_| "output buffer is too large to initialize".to_owned())?
        ],
    );
    queue.write_buffer(&params_buffer, 0, &params);

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("atlas-search-bind-group-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("atlas-search-pipeline-layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("atlas-search-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("atlas_search"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("atlas-search-bind-group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let workgroups = args.global_size.div_ceil(args.local_size);
    let x_workgroups =
        u32::try_from(workgroups).map_err(|_| "workgroup count exceeds WGPU uint".to_owned())?;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("atlas-search-command-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("atlas-search-compute-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(x_workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_bytes);
    let submission_index = queue.submit(Some(encoder.finish()));

    read_launch_output(
        &device,
        &readback_buffer,
        output_bytes,
        args.max_matches,
        submission_index,
    )
}

async fn check_wgpu_runtime() -> Result<(), String> {
    let (_device, _queue) = create_wgpu_device().await?;
    Ok(())
}

async fn create_wgpu_device() -> Result<(wgpu::Device, wgpu::Queue), String> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        })
        .await
        .map_err(|error| format!("no compatible WGPU adapter: {error}"))?;
    adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .map_err(|error| format!("cannot create WGPU device: {error}"))
}

fn encode_params(args: &LaunchArgs) -> Result<Vec<u8>, String> {
    let start_lo = args.start as u32;
    let start_hi = (args.start >> 32) as u32;
    let end_lo = args.end as u32;
    let end_hi = (args.end >> 32) as u32;
    let max_matches =
        u32::try_from(args.max_matches).map_err(|_| "max-matches exceeds WGPU uint".to_owned())?;
    let values = [start_lo, start_hi, end_lo, end_hi, max_matches, 0, 0, 0];
    Ok(values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect())
}

fn read_launch_output(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    output_bytes: u64,
    max_matches: usize,
    submission_index: wgpu::SubmissionIndex,
) -> Result<LaunchOutput, String> {
    let slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result.map_err(|error| error.to_string()));
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: None,
        })
        .map_err(|error| format!("WGPU poll failed: {error}"))?;
    receiver
        .recv()
        .map_err(|error| format!("WGPU map callback failed: {error}"))?
        .map_err(|error| format!("WGPU readback map failed: {error}"))?;

    let bytes = slice
        .get_mapped_range()
        .map_err(|error| format!("WGPU readback view failed: {error}"))?;
    let result = parse_output_bytes(&bytes, output_bytes, max_matches);
    drop(bytes);
    buffer.unmap();
    result
}

fn parse_output_bytes(
    bytes: &[u8],
    output_bytes: u64,
    max_matches: usize,
) -> Result<LaunchOutput, String> {
    let minimum_bytes = OUTPUT_HEADER_U32S * 4;
    if bytes.len() < minimum_bytes || u64::try_from(bytes.len()).unwrap_or(u64::MAX) != output_bytes
    {
        return Err("WGPU readback buffer has unexpected size".to_owned());
    }
    let match_count = read_u32(bytes, 0)? as usize;
    let retained = match_count.min(max_matches);
    let mut matches = Vec::with_capacity(retained);
    for index in 0..retained {
        let base = (OUTPUT_HEADER_U32S + index * CANDIDATE_U32S) * 4;
        let lo = u64::from(read_u32(bytes, base)?);
        let hi = u64::from(read_u32(bytes, base + 4)?);
        matches.push(lo | (hi << 32));
    }
    Ok(LaunchOutput {
        matches,
        match_count,
    })
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| "WGPU readback offset overflowed".to_owned())?;
    let chunk = bytes
        .get(offset..end)
        .ok_or_else(|| "WGPU readback buffer is truncated".to_owned())?;
    Ok(u32::from_le_bytes(
        chunk
            .try_into()
            .map_err(|_| "invalid u32 readback chunk".to_owned())?,
    ))
}

fn extract_wgsl_workgroup_size_x(source: &str) -> Result<usize, String> {
    let Some(annotation_start) = source.find("@workgroup_size") else {
        return Err("WGPU WGSL artifact is missing @workgroup_size".to_owned());
    };
    let after_annotation = &source[annotation_start + "@workgroup_size".len()..];
    let open_paren = after_annotation
        .find('(')
        .ok_or_else(|| "WGPU WGSL @workgroup_size is missing '('".to_owned())?;
    let after_paren = after_annotation[open_paren + 1..].trim_start();
    let digits = after_paren
        .chars()
        .take_while(|character| character.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        return Err("WGPU WGSL @workgroup_size has no x dimension".to_owned());
    }
    digits
        .parse()
        .map_err(|_| "WGPU WGSL @workgroup_size x dimension is invalid".to_owned())
}

fn format_features(features: &[String]) -> String {
    let mut text = "hardware=WGPU adapter\n".to_owned();
    text.push_str(
        &features
            .iter()
            .map(|feature| format!("feature={feature}\n"))
            .collect::<String>(),
    );
    text
}

fn format_launch_output(output: &LaunchOutput) -> String {
    let mut text = format!("match_count={}\n", output.match_count);
    text.push_str(
        &output
            .matches
            .iter()
            .map(|candidate| format!("match={candidate}\n"))
            .collect::<String>(),
    );
    text
}

fn optional_output_path(args: &[String]) -> Result<Option<String>, String> {
    let Some(index) = args.iter().position(|arg| arg == "-o") else {
        return Ok(None);
    };
    args.get(index + 1)
        .cloned()
        .map(Some)
        .ok_or_else(|| "missing -o output path".to_owned())
}

fn parse_u64_flag(args: &[String], flag: &str) -> Result<u64, String> {
    flag_value(args, flag)?
        .parse()
        .map_err(|_| format!("{flag} must be an unsigned integer"))
}

fn parse_usize_flag(args: &[String], flag: &str) -> Result<usize, String> {
    flag_value(args, flag)?
        .parse()
        .map_err(|_| format!("{flag} must be an unsigned integer"))
}

fn flag_value<'a>(args: &'a [String], flag: &str) -> Result<&'a str, String> {
    let index = args
        .iter()
        .position(|arg| arg == flag)
        .ok_or_else(|| format!("missing {flag}"))?;
    args.get(index + 1)
        .map(String::as_str)
        .ok_or_else(|| format!("missing {flag} value"))
}

fn write_checked_wgsl(path: &str, source: &str) -> Result<(), String> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    fs::write(path, source).map_err(|error| error.to_string())
}
