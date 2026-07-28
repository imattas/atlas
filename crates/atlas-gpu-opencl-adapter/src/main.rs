//! `atlas-gpu-opencl-run` command entry point.

use atlas_gpu_opencl_adapter::{run_cli, OpenClLauncher};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run_cli(&args, &OpenClLauncher) {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    }
}
