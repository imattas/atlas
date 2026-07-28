//! Native search microbenchmark used by external comparison scripts.

use std::time::Instant;

use atlas_scheduler::CancellationToken;
use atlas_search_ir::{SearchDomain, SearchOp, SearchProgram};
use atlas_search_native::NativeSearcher;

struct BenchCase {
    name: &'static str,
    program: SearchProgram,
    domain: SearchDomain,
    iterations: u32,
}

fn main() {
    let cases = [
        BenchCase {
            name: "xor_width20",
            program: SearchProgram::new(
                20,
                vec![SearchOp::XorEq {
                    mask: 0xaaaaa,
                    target: 0xfffff,
                }],
            )
            .expect("valid xor case"),
            domain: SearchDomain::new(0, 1 << 20),
            iterations: 2_000,
        },
        BenchCase {
            name: "add_width20",
            program: SearchProgram::new(
                20,
                vec![SearchOp::AddEq {
                    addend: 1,
                    target: 424_242,
                }],
            )
            .expect("valid add case"),
            domain: SearchDomain::new(0, 1 << 20),
            iterations: 2_000,
        },
        BenchCase {
            name: "checksum_width20",
            program: SearchProgram::new(
                20,
                vec![SearchOp::ChecksumEq {
                    modulus: 997,
                    target: 313,
                }],
            )
            .expect("valid checksum case"),
            domain: SearchDomain::new(0, 1 << 20),
            iterations: 20,
        },
    ];

    println!("[");
    for (index, case) in cases.iter().enumerate() {
        let token = CancellationToken::new();
        let start = Instant::now();
        let mut count = 0_usize;
        let mut evaluated = 0_u64;
        let mut closed_form = false;
        for _ in 0..case.iterations {
            let result = NativeSearcher::search_with_stats(&case.program, case.domain, &token);
            count = result.matches.len();
            evaluated = result.candidates_evaluated;
            closed_form = result.used_closed_form;
        }
        let elapsed_ns = start.elapsed().as_nanos();
        let mean_ns = elapsed_ns / u128::from(case.iterations);
        println!(
            "  {{\"name\":\"{}\",\"engine\":\"atlas-native\",\"iterations\":{},\"mean_ns\":{},\"matches\":{},\"candidates_evaluated\":{},\"used_closed_form\":{}}}{}",
            case.name,
            case.iterations,
            mean_ns,
            count,
            evaluated,
            closed_form,
            if index + 1 == cases.len() { "" } else { "," }
        );
    }
    println!("]");
}
