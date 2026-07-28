import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class VerifyScriptTests(unittest.TestCase):
    def test_distributed_profiles_require_all_gpu_adapters_on_windows_and_unix(self):
        required_gpu_adapter_paths = [
            "crates/atlas-gpu-opencl-adapter/src/lib.rs",
            "crates/atlas-gpu-opencl-adapter/src/main.rs",
            "crates/atlas-gpu-cuda-adapter/src/lib.rs",
            "crates/atlas-gpu-cuda-adapter/src/main.rs",
            "crates/atlas-gpu-hip-adapter/src/lib.rs",
            "crates/atlas-gpu-hip-adapter/src/main.rs",
            "crates/atlas-gpu-vulkan-adapter/src/lib.rs",
            "crates/atlas-gpu-vulkan-adapter/src/main.rs",
        ]
        scripts = [
            ROOT / "scripts" / "verify.ps1",
            ROOT / "scripts" / "verify.sh",
        ]

        for script in scripts:
            with self.subTest(script=script.name):
                text = script.read_text(encoding="utf-8")
                for required_path in required_gpu_adapter_paths:
                    self.assertIn(required_path, text)

    def test_hardware_profile_runs_all_real_device_gpu_tests_on_windows_and_unix(self):
        hardware_test_cases = [
            ("atlas-gpu-opencl-adapter", "generated_opencl_kernel_runs_on_device_and_preserves_full_candidates"),
            ("atlas-gpu-cuda-adapter", "generated_cuda_kernel_runs_on_device_and_preserves_full_candidates"),
            ("atlas-gpu-hip-adapter", "generated_hip_kernel_runs_on_device_and_preserves_full_candidates"),
            ("atlas-gpu-vulkan-adapter", "generated_vulkan_kernel_runs_on_device_and_preserves_full_candidates"),
            ("atlas-gpu-vulkan-adapter", "generated_vulkan_64_bit_kernel_runs_on_device"),
        ]

        scripts = [
            ROOT / "scripts" / "verify.ps1",
            ROOT / "scripts" / "verify.sh",
        ]

        for script in scripts:
            with self.subTest(script=script.name):
                text = script.read_text(encoding="utf-8")
                self.assertIn("hardware", text)
                for package, test_name in hardware_test_cases:
                    self.assertIn(package, text)
                    self.assertIn(test_name, text)
                self.assertIn("--ignored", text)


if __name__ == "__main__":
    unittest.main()
