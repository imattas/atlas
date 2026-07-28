//! HIP adapter CLI entrypoint.

use atlas_gpu_hip_adapter::{run_cli, HipModuleLauncher};

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match run_cli(&args, &HipModuleLauncher) {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
