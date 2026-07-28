//! CUDA adapter CLI entrypoint.

use atlas_gpu_cuda_adapter::{run_cli, CudaPtxLauncher};

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match run_cli(&args, &CudaPtxLauncher) {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
