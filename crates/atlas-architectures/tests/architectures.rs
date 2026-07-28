//! Architecture differential-style tests.

use atlas_architectures::{evaluate_instruction, semantics, ByteOrder, InstructionSemantics};
use atlas_program::Architecture;

#[test]
fn supports_x86_32_arm64_and_wasm_pointer_semantics() {
    let x86 = semantics(Architecture::X86_32).unwrap();
    let arm = semantics(Architecture::Arm64).unwrap();
    let wasm = semantics(Architecture::WebAssembly).unwrap();

    assert_eq!(x86.pointer_width, 32);
    assert_eq!(arm.pointer_width, 64);
    assert_eq!(wasm.pointer_width, 32);
    assert_eq!(wasm.byte_order, ByteOrder::Little);
    assert!(x86.registers.contains(&"eax".to_owned()));
    assert!(arm.registers.contains(&"x0".to_owned()));
    assert!(wasm.registers.contains(&"stack".to_owned()));
}

#[test]
fn reports_precise_unsupported_architecture_diagnostic() {
    let diagnostic = semantics(Architecture::X86_64).unwrap_err();

    assert_eq!(diagnostic.location, "architecture");
    assert!(diagnostic.message.contains("unsupported"));
}

#[test]
fn arithmetic_branch_and_load_store_match_reference_fixtures() {
    assert_eq!(
        evaluate_instruction(InstructionSemantics::Add {
            width: 32,
            lhs: u64::from(u32::MAX),
            rhs: 1,
        }),
        0
    );
    assert_eq!(
        evaluate_instruction(InstructionSemantics::Branch { condition: true }),
        1
    );
    assert_eq!(
        evaluate_instruction(InstructionSemantics::LoadStore {
            width: 32,
            offset: 8,
        }),
        (32_u64 << 32) | 8
    );
}
