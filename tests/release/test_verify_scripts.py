import unittest
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class VerifyScriptTests(unittest.TestCase):
    def test_unix_shell_verify_scripts_use_lf_line_endings(self):
        for script in [ROOT / "scripts" / "verify.sh"]:
            with self.subTest(script=script.name):
                self.assertNotIn(b"\r\n", script.read_bytes())

    def test_git_attributes_preserve_unix_shell_line_endings(self):
        attributes_path = ROOT / ".gitattributes"
        self.assertTrue(attributes_path.exists(), ".gitattributes is required")
        attributes = attributes_path.read_text(encoding="utf-8")
        self.assertIn("*.sh text eol=lf", attributes)

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
            "crates/atlas-gpu-wgpu-adapter/src/lib.rs",
            "crates/atlas-gpu-wgpu-adapter/src/main.rs",
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
            ("atlas-gpu-opencl-adapter", "generated_opencl_dense_kernel_retains_full_device_buffer"),
            ("atlas-gpu-opencl-adapter", "generated_opencl_64_bit_kernel_runs_on_device"),
            ("atlas-gpu-cuda-adapter", "generated_cuda_kernel_runs_on_device_and_preserves_full_candidates"),
            ("atlas-gpu-cuda-adapter", "generated_cuda_dense_kernel_retains_full_device_buffer"),
            ("atlas-gpu-cuda-adapter", "generated_cuda_64_bit_kernel_runs_on_device"),
            ("atlas-gpu-hip-adapter", "generated_hip_kernel_runs_on_device_and_preserves_full_candidates"),
            ("atlas-gpu-hip-adapter", "generated_hip_dense_kernel_retains_full_device_buffer"),
            ("atlas-gpu-hip-adapter", "generated_hip_64_bit_kernel_runs_on_device"),
            ("atlas-gpu-vulkan-adapter", "generated_vulkan_kernel_runs_on_device_and_preserves_full_candidates"),
            ("atlas-gpu-vulkan-adapter", "generated_vulkan_dense_kernel_retains_full_device_buffer"),
            ("atlas-gpu-vulkan-adapter", "generated_vulkan_64_bit_kernel_runs_on_device"),
            ("atlas-gpu-wgpu-adapter", "generated_wgpu_kernel_runs_on_device_and_preserves_full_candidates"),
            ("atlas-gpu-wgpu-adapter", "generated_wgpu_dense_kernel_retains_full_device_buffer"),
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
                doctor_index = text.find("run -q -p atlas-cli -- doctor")
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
                doctor_index = text.find("run -q -p atlas-cli -- doctor", hardware_block_index)
                benchmark_index = text.find("Forced-GPU benchmark", hardware_block_index)
                force_gpu_index = text.find("--force-gpu")
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

    def test_hardware_profile_records_placement_selected_gpu_benchmark_before_forced_gpu(self):
        expectations = {
            "verify.ps1": [
                "Invoke-PlacementSelectedGpuBenchmark",
                "Placement-selected GPU benchmark",
                '"--end", "1000000"',
                "requested_gpu_sdk",
                "actual_gpu_sdk",
                "DeviceValidated",
            ],
            "verify.sh": [
                "run_placement_selected_gpu_benchmark",
                "Placement-selected GPU benchmark",
                "--end 1000000",
                "requested_gpu_sdk",
                "actual_gpu_sdk",
                "DeviceValidated",
            ],
        }

        for script_name, required_tokens in expectations.items():
            with self.subTest(script=script_name):
                text = (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
                for token in required_tokens:
                    self.assertIn(token, text)
                placement_index = text.find("Placement-selected GPU benchmark")
                forced_index = text.find("Forced-GPU benchmark")
                self.assertNotEqual(-1, placement_index)
                self.assertNotEqual(-1, forced_index)
                self.assertLess(placement_index, forced_index)

    def test_hardware_profile_records_warm_cache_placement_benchmark(self):
        expectations = {
            "verify.ps1": [
                "Invoke-WarmCachePlacementGpuBenchmark",
                "Warm-cache placement GPU benchmark",
                "--force-gpu",
                "Warm-cache auto-placement GPU benchmark",
                "requested_gpu_sdk",
                "actual_gpu_sdk",
                "DeviceValidated",
            ],
            "verify.sh": [
                "run_warm_cache_placement_gpu_benchmark",
                "Warm-cache placement GPU benchmark",
                "--force-gpu",
                "Warm-cache auto-placement GPU benchmark",
                "requested_gpu_sdk",
                "actual_gpu_sdk",
                "DeviceValidated",
            ],
        }

        for script_name, required_tokens in expectations.items():
            with self.subTest(script=script_name):
                text = (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
                for token in required_tokens:
                    self.assertIn(token, text)
                forced_warm_index = text.find("Warm-cache placement GPU benchmark")
                auto_warm_index = text.find("Warm-cache auto-placement GPU benchmark")
                first_backend_index = text.find("Forced-GPU OpenCL benchmark")
                self.assertNotEqual(-1, forced_warm_index)
                self.assertNotEqual(-1, auto_warm_index)
                self.assertNotEqual(-1, first_backend_index)
                self.assertLess(forced_warm_index, auto_warm_index)
                self.assertLess(auto_warm_index, first_backend_index)

    def test_unix_verify_script_falls_back_to_windows_cargo_exe(self):
        text = (ROOT / "scripts" / "verify.sh").read_text(encoding="utf-8")
        for token in [
            "resolve_cargo_command",
            "cargo.exe",
            "cargo.exe --version",
            "cargo_cmd",
            '"$cargo_cmd" run -q -p atlas-cli -- doctor',
        ]:
            self.assertIn(token, text)

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

    def test_hardware_profile_records_64_bit_forced_gpu_benchmarks(self):
        scripts = [
            ROOT / "scripts" / "verify.ps1",
            ROOT / "scripts" / "verify.sh",
        ]

        for script in scripts:
            with self.subTest(script=script.name):
                text = script.read_text(encoding="utf-8")
                self.assertIn("xor64", text)
                self.assertIn("0x8000000000000000", text)
                self.assertIn("0x8000000000000002", text)

    def test_hardware_profile_validates_forced_gpu_benchmark_backend_identity(self):
        expectations = {
            "verify.ps1": [
                "Invoke-ForcedGpuBenchmark",
                'Invoke-ForcedGpuBenchmark "Forced-GPU benchmark" $null $null',
                "actual_gpu_sdk",
                "DeviceValidated",
                "expected actual_gpu_sdk",
                "driver exit 0",
                "driver launches",
                "launch abi",
            ],
            "verify.sh": [
                "run_forced_gpu_benchmark",
                'run_forced_gpu_benchmark "Forced-GPU benchmark" "" ""',
                "actual_gpu_sdk",
                "DeviceValidated",
                "expected actual_gpu_sdk",
                "driver exit 0",
                "driver launches",
                "launch abi",
            ],
        }

        for script_name, required_tokens in expectations.items():
            with self.subTest(script=script_name):
                text = (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
                for token in required_tokens:
                    self.assertIn(token, text)

    def test_hardware_profile_skips_unavailable_real_device_backends_from_doctor_probe(self):
        expectations = {
            "verify.ps1": [
                "Get-GpuFeatureProbeOk",
                "Skip-HardwareStep",
                "gpu_feature_probes",
                "CUDA real-device search",
            ],
            "verify.sh": [
                "gpu_feature_probe_ok",
                "skip_hardware_step",
                "gpu_feature_probes",
                "CUDA real-device search",
            ],
        }

        for script_name, required_tokens in expectations.items():
            with self.subTest(script=script_name):
                text = (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
                for token in required_tokens:
                    self.assertIn(token, text)

    def test_hardware_profile_requires_adapter_launch_abi_features(self):
        expectations = {
            "verify.ps1": [
                "Assert-GpuFeatureProbeHasLaunchAbi",
                "launchAbiU32",
                "launchAbiU64",
            ],
            "verify.sh": [
                "assert_gpu_feature_probes_have_launch_abi",
                "launchAbiU32",
                "launchAbiU64",
            ],
        }

        for script_name, required_tokens in expectations.items():
            with self.subTest(script=script_name):
                text = (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
                for token in required_tokens:
                    self.assertIn(token, text)

    def test_hardware_doctor_validator_rejects_empty_and_incomplete_probe_reports(self):
        validator = ROOT / "scripts" / "verify_hardware_doctor.py"
        cases = [
            (
                '{"schema_major":1,"kind":"doctor","gpu_feature_probes":[]}',
                1,
                "at least one GPU feature probe",
            ),
            (
                '{"schema_major":1,"kind":"doctor","gpu_feature_probes":[{"name":"OpenCL","ok":true,"features":["int64"]}]}',
                1,
                "missing launchAbiU32",
            ),
            (
                '{"schema_major":1,"kind":"doctor","gpu_feature_probes":[{"name":"OpenCL","ok":true,"features":["int64","launchAbiU32","launchAbiU64"]}]}',
                1,
                "does not have an available adapter binary",
            ),
            (
                '{"schema_major":1,"kind":"doctor","adapter_binaries":[{"name":"OpenCL","available":true}],"gpu_feature_probes":[{"name":"OpenCL","ok":true,"features":["int64","launchAbiU32","launchAbiU64"]}]}',
                0,
                "",
            ),
        ]

        for payload, expected_status, expected_error in cases:
            with self.subTest(payload=payload):
                result = subprocess.run(
                    ["python", str(validator), "--require-launch-abi"],
                    input=payload,
                    text=True,
                    capture_output=True,
                    cwd=ROOT,
                    check=False,
                )

                self.assertEqual(result.returncode, expected_status)
                if expected_error:
                    self.assertIn(expected_error, result.stderr)

    def test_hardware_profile_gates_int64_gpu_checks_on_backend_features(self):
        expectations = {
            "verify.ps1": [
                "Get-GpuFeatureProbeHasFeature",
                "Get-AnyGpuFeatureProbeHasInt64",
                '"int64"',
                '"shaderInt64"',
                "OpenCL int64 feature unavailable",
                "Vulkan shaderInt64 feature unavailable",
                "CUDA int64 feature unavailable",
                "HIP int64 feature unavailable",
            ],
            "verify.sh": [
                "gpu_feature_probe_has_feature",
                "gpu_any_feature_probe_has_int64",
                "int64",
                "shaderInt64",
                "OpenCL int64 feature unavailable",
                "Vulkan shaderInt64 feature unavailable",
                "CUDA int64 feature unavailable",
                "HIP int64 feature unavailable",
            ],
        }

        for script_name, required_tokens in expectations.items():
            with self.subTest(script=script_name):
                text = (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
                for token in required_tokens:
                    self.assertIn(token, text)

    def test_hardware_profile_uses_repeated_benchmark_samples(self):
        expectations = {
            "verify.ps1": ["--samples", "$BenchmarkSamples", "sample_count"],
            "verify.sh": ["--samples", "benchmark_samples", "sample_count"],
        }

        for script_name, required_tokens in expectations.items():
            with self.subTest(script=script_name):
                text = (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
                for token in required_tokens:
                    self.assertIn(token, text)

    def test_hardware_profile_exercises_dense_retained_gpu_buffers(self):
        expectations = {
            "verify.ps1": [
                "Forced-GPU dense benchmark",
                "dense",
                "1500",
                "MinRetainedMatches",
                "max_matches",
            ],
            "verify.sh": [
                "Forced-GPU dense benchmark",
                "dense",
                "1500",
                "min_retained_matches",
                "max_matches",
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

    def test_hardware_profile_requires_native_simd_gpu_benchmark_equivalence(self):
        expectations = {
            "verify.ps1": [
                "native/SIMD benchmark mismatch",
                "native/GPU benchmark mismatch",
                "$Benchmark.simd.matches",
                "$Benchmark.native.matches",
                "$Benchmark.accelerator.matches",
                "simd_samples_ns",
            ],
            "verify.sh": [
                "native/SIMD benchmark mismatch",
                "native/GPU benchmark mismatch",
                'document["simd"]["matches"]',
                'document["native"]["matches"]',
                'document["accelerator"]["matches"]',
                "simd_samples_ns",
            ],
        }

        for script_name, required_tokens in expectations.items():
            with self.subTest(script=script_name):
                text = (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
                for token in required_tokens:
                    self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
