//! WGPU adapter CLI entrypoint.

use atlas_gpu_wgpu_adapter::{run_cli, WgpuLauncher};

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match run_cli(&args, &WgpuLauncher) {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
