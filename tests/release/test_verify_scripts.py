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

    def test_hardware_profile_emits_doctor_diagnostics_before_device_tests(self):
        scripts = [
            ROOT / "scripts" / "verify.ps1",
            ROOT / "scripts" / "verify.sh",
        ]

        for script in scripts:
            with self.subTest(script=script.name):
                text = script.read_text(encoding="utf-8")
                doctor_index = text.find("cargo run -q -p atlas-cli -- doctor")
                first_hardware_test_index = text.find(
                    "generated_opencl_kernel_runs_on_device_and_preserves_full_candidates"
                )
                self.assertNotEqual(-1, doctor_index)
                self.assertNotEqual(-1, first_hardware_test_index)
                self.assertLess(doctor_index, first_hardware_test_index)

    def test_hardware_profile_records_forced_gpu_benchmark_before_device_tests(self):
        scripts = [
            ROOT / "scripts" / "verify.ps1",
            ROOT / "scripts" / "verify.sh",
        ]

        for script in scripts:
            with self.subTest(script=script.name):
                text = script.read_text(encoding="utf-8")
                hardware_block_index = text.find("GPU doctor diagnostics")
                doctor_index = text.find("cargo run -q -p atlas-cli -- doctor", hardware_block_index)
                benchmark_index = text.find("Forced-GPU benchmark", hardware_block_index)
                force_gpu_index = text.find("--force-gpu", benchmark_index)
                first_hardware_test_index = text.find(
                    "generated_opencl_kernel_runs_on_device_and_preserves_full_candidates"
                )
                self.assertNotEqual(-1, hardware_block_index)
                self.assertNotEqual(-1, doctor_index)
                self.assertNotEqual(-1, benchmark_index)
                self.assertNotEqual(-1, force_gpu_index)
                self.assertNotEqual(-1, first_hardware_test_index)
                self.assertLess(doctor_index, benchmark_index)
                self.assertLess(benchmark_index, first_hardware_test_index)

    def test_hardware_profile_records_per_sdk_forced_gpu_benchmarks(self):
        scripts = [
            ROOT / "scripts" / "verify.ps1",
            ROOT / "scripts" / "verify.sh",
        ]

        for script in scripts:
            with self.subTest(script=script.name):
                text = script.read_text(encoding="utf-8")
                self.assertIn("--gpu-sdk", text)
                for sdk in ["opencl", "vulkan", "cuda", "hip"]:
                    self.assertIn(sdk, text)

    def test_hardware_profile_validates_forced_gpu_benchmark_backend_identity(self):
        expectations = {
            "verify.ps1": [
                "Invoke-ForcedGpuBenchmark",
                "actual_gpu_sdk",
                "DeviceValidated",
                "expected actual_gpu_sdk",
            ],
            "verify.sh": [
                "run_forced_gpu_benchmark",
                "actual_gpu_sdk",
                "DeviceValidated",
                "expected actual_gpu_sdk",
            ],
        }

        for script_name, required_tokens in expectations.items():
            with self.subTest(script=script_name):
                text = (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
                for token in required_tokens:
                    self.assertIn(token, text)

    def test_hardware_profile_attempts_every_backend_before_reporting_failure(self):
        expectations = {
            "verify.ps1": ["Invoke-HardwareStep", "$HardwareFailures", "$HardwareFailures.Count"],
            "verify.sh": ["run_hardware_step", "hardware_failures", "set +e"],
        }

        for script_name, required_tokens in expectations.items():
            with self.subTest(script=script_name):
                text = (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
                for token in required_tokens:
                    self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
