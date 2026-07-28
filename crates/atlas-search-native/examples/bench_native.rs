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
    let cases = bench_cases();

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

fn bench_cases() -> [BenchCase; 6] {
    [
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
        BenchCase {
            name: "rotxor_width24",
            program: SearchProgram::new(
                24,
                vec![SearchOp::RotateXorEq {
                    rotate_left: 7,
                    mask: 0xA5_A5_A5,
                    target: 0x12_34_56,
                }],
            )
            .expect("valid rotate-xor case"),
            domain: SearchDomain::new(0, 1 << 24),
            iterations: 2_000,
        },
        BenchCase {
            name: "muladd_width24",
            program: SearchProgram::new(
                24,
                vec![SearchOp::MulAddEq {
                    multiplier: 65_537,
                    addend: 0x1337,
                    target: 0xC0_FF_EE,
                }],
            )
            .expect("valid multiply-add case"),
            domain: SearchDomain::new(0, 1 << 24),
            iterations: 2_000,
        },
        BenchCase {
            name: "serial_bytes_width32",
            program: SearchProgram::new(
                32,
                vec![
                    SearchOp::ByteEq {
                        byte_index: 0,
                        value: b'C',
                    },
                    SearchOp::ByteEq {
                        byte_index: 1,
                        value: b'T',
                    },
                    SearchOp::ByteEq {
                        byte_index: 2,
                        value: b'F',
                    },
                ],
            )
            .expect("valid fixed-byte serial case"),
            domain: SearchDomain::new(0, 1 << 32),
            iterations: 200,
        },
    ]
}
