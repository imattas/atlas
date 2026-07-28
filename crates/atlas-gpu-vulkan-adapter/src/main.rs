//! Vulkan adapter CLI entrypoint.

use atlas_gpu_vulkan_adapter::{run_cli, VulkanSpirvLauncher};

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match run_cli(&args, &VulkanSpirvLauncher) {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
