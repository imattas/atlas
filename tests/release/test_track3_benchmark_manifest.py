import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRACK3_BENCHMARK = ROOT / "benchmarks" / "track3" / "manifest.toml"


class Track3BenchmarkManifestTest(unittest.TestCase):
    def test_track3_benchmark_records_real_gpu_targets_not_only_fallback(self):
        manifest = tomllib.loads(TRACK3_BENCHMARK.read_text(encoding="utf-8"))

        cases = {case["id"]: case for case in manifest["case"]}
        targets = set(cases["bounded-scalar-simd-gpu-equivalence"]["targets"])

        self.assertIn("scalar", targets)
        self.assertIn("simd", targets)
        self.assertIn("gpu-device", targets)
        self.assertNotIn("gpu-fallback", targets)

    def test_track3_benchmark_samples_include_device_backend_and_non_placeholder_timings(self):
        manifest = tomllib.loads(TRACK3_BENCHMARK.read_text(encoding="utf-8"))

        for sample in manifest["sample"]:
            with self.subTest(sample=sample["id"]):
                self.assertRegex(
                    sample["hardware"],
                    r"(OpenCL|Vulkan|HIP|CUDA|GPU|Radeon|NVIDIA|AMD|Intel)",
                )
                self.assertIn(sample["gpu_backend"], {"OpenCL", "Vulkan", "HIP", "CUDA"})
                timing_fields = [
                    "scalar_ms",
                    "simd_ms",
                    "gpu_compile_ms",
                    "gpu_device_ms",
                    "distributed_overhead_ms",
                ]
                for field in timing_fields:
                    self.assertGreaterEqual(len(sample[field]), 3)
                    self.assertTrue(all(value >= 0 for value in sample[field]))
                timing_vectors = tuple(tuple(sample[field]) for field in timing_fields)
                self.assertNotEqual(len(set(timing_vectors)), 1)


if __name__ == "__main__":
    unittest.main()
