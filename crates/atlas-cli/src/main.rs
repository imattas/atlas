//! `atlas` command entry point.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match atlas_cli::run(&args) {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    }
}
