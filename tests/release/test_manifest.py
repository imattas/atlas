import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "release"))

from write_manifest import validate_manifest


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "release" / "manifest.toml"


def write_manifest_variant(data: str) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, encoding="utf-8")
    with handle:
        handle.write(data)
    return Path(handle.name)


class ReleaseManifestTest(unittest.TestCase):
    def test_checked_in_manifest_is_valid(self) -> None:
        self.assertEqual(validate_manifest(MANIFEST), [])

    def test_rejects_absent_required_suites_and_skipped_tests(self) -> None:
        text = MANIFEST.read_text(encoding="utf-8")
        missing = write_manifest_variant(text.replace('name = "core"', 'name = "core-missing"', 1))
        skipped = write_manifest_variant(text.replace('status = "passed"', 'status = "skipped"', 1))

        self.assertIn("all required suites must be present", validate_manifest(missing))
        self.assertTrue(any("is not passed" in error for error in validate_manifest(skipped)))

    def test_rejects_unsigned_artifacts_and_missing_architecture_evidence(self) -> None:
        text = MANIFEST.read_text(encoding="utf-8")
        unsigned = write_manifest_variant(
            "\n".join(
                'signature = "short"' if line.startswith("signature = ") else line
                for line in text.splitlines()
            )
        )
        missing_architecture = write_manifest_variant(
            text.replace('path = "tests/fixtures/architectures/wasm.toml"', 'path = "tests/fixtures/architectures/missing.toml"', 1)
        )

        self.assertIn("signature is required", validate_manifest(unsigned))
        self.assertIn("architecture evidence is incomplete", validate_manifest(missing_architecture))

    def test_rejects_benchmark_claims_without_hardware_or_sample_metadata(self) -> None:
        text = MANIFEST.read_text(encoding="utf-8")
        no_hardware = write_manifest_variant(text.replace('hardware = "sample-corpus-cpu"', 'hardware = ""', 1))
        no_sample = write_manifest_variant(
            text.replace('sample_metadata = "bounded authorized fixtures"', 'sample_metadata = ""', 1)
        )

        self.assertTrue(any("hardware metadata missing" in error for error in validate_manifest(no_hardware)))
        self.assertTrue(any("sample metadata missing" in error for error in validate_manifest(no_sample)))

    def test_manifest_requires_all_gpu_adapter_artifacts(self) -> None:
        text = MANIFEST.read_text(encoding="utf-8")
        required_paths = {
            "crates/atlas-gpu-opencl-adapter/src/lib.rs",
            "crates/atlas-gpu-opencl-adapter/src/main.rs",
            "crates/atlas-gpu-cuda-adapter/src/lib.rs",
            "crates/atlas-gpu-cuda-adapter/src/main.rs",
            "crates/atlas-gpu-hip-adapter/src/lib.rs",
            "crates/atlas-gpu-hip-adapter/src/main.rs",
            "crates/atlas-gpu-vulkan-adapter/src/lib.rs",
            "crates/atlas-gpu-vulkan-adapter/src/main.rs",
            "crates/atlas-gpu-wgpu-adapter/src/lib.rs",
            "crates/atlas-gpu-wgpu-adapter/src/main.rs",
        }

        for path in required_paths:
            self.assertIn(f'path = "{path}"', text)

        missing_cuda = write_manifest_variant(
            text.replace('path = "crates/atlas-gpu-cuda-adapter/src/lib.rs"', 'path = "missing/cuda.rs"', 1)
        )
        self.assertIn("GPU adapter evidence is incomplete", validate_manifest(missing_cuda))

    def test_hardware_acceleration_docs_cover_all_gpu_backends(self) -> None:
        text = (ROOT / "docs" / "hardware-acceleration.md").read_text(encoding="utf-8")
        required_tokens = [
            "OpenCL",
            "Vulkan",
            "WGPU",
            "CUDA",
            "HIP",
            "gpu/wgpu/atlas_search.wgsl",
            "atlas-gpu-wgpu-run",
            "generated_wgpu_kernel_runs_on_device_and_preserves_full_candidates",
        ]

        for token in required_tokens:
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_manifest_includes_track3_device_benchmark_evidence(self) -> None:
        text = MANIFEST.read_text(encoding="utf-8")

        self.assertIn('path = "benchmarks/track3/manifest.toml"', text)
        missing_track3_benchmark = write_manifest_variant(
            text.replace(
                'path = "benchmarks/track3/manifest.toml"',
                'path = "benchmarks/track3/missing.toml"',
                1,
            )
        )

        self.assertIn(
            "Track 3 benchmark evidence is incomplete",
            validate_manifest(missing_track3_benchmark),
        )


if __name__ == "__main__":
    unittest.main()
