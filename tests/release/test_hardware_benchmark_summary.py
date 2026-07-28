import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class HardwareBenchmarkSummaryTests(unittest.TestCase):
    def test_summarizes_dense_benchmark_without_dumping_match_arrays(self):
        benchmark = {
            "schema_major": 1,
            "kind": "benchmark",
            "fixture": "dense",
            "domain": {"start": 0, "end": 1500},
            "sample_count": 3,
            "native_samples_ns": [100, 110, 120],
            "simd_samples_ns": [80, 85, 90],
            "accelerator_samples_ns": [200, 210, 220],
            "native": {"elapsed_ns": 100, "matches": list(range(1500))},
            "simd": {"elapsed_ns": 80, "matches": list(range(1500))},
            "accelerator": {
                "elapsed_ns": 200,
                "requested_gpu_sdk": "hip",
                "actual_gpu_sdk": "HIP",
                "mode": "DeviceValidated",
                "matches": list(range(1500)),
                "launch": {
                    "global_size": 1536,
                    "local_size": 256,
                    "max_matches": 1500,
                    "output_buffer_bytes": 12000,
                },
                "telemetry": "HIP; driver exit 0; driver launches 1; launch abi u32",
            },
        }

        result = subprocess.run(
            ["python", "scripts/summarize_hardware_benchmark.py"],
            input=json.dumps(benchmark),
            text=True,
            capture_output=True,
            cwd=ROOT,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["fixture"], "dense")
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["native_samples_ns"], [100, 110, 120])
        self.assertEqual(summary["simd_samples_ns"], [80, 85, 90])
        self.assertEqual(summary["accelerator_samples_ns"], [200, 210, 220])
        self.assertEqual(summary["simd"]["elapsed_ns"], 80)
        self.assertEqual(summary["simd"]["match_count"], 1500)
        self.assertEqual(summary["accelerator"]["mode"], "DeviceValidated")
        self.assertEqual(summary["accelerator"]["actual_gpu_sdk"], "HIP")
        self.assertEqual(summary["accelerator"]["match_count"], 1500)
        self.assertEqual(summary["accelerator"]["first_matches"], [0, 1, 2, 3, 4])
        self.assertEqual(summary["accelerator"]["last_matches"], [1495, 1496, 1497, 1498, 1499])
        self.assertNotIn("matches", summary["accelerator"])
        self.assertLess(len(result.stdout), 1000)

    def test_hardware_verify_scripts_emit_compact_benchmark_summaries(self):
        expectations = {
            "verify.ps1": ["scripts/summarize_hardware_benchmark.py", "--input"],
            "verify.sh": [
                "scripts/summarize_hardware_benchmark.py",
                "--input",
                "export -f print_hardware_benchmark_summary",
            ],
        }

        for script_name, tokens in expectations.items():
            with self.subTest(script=script_name):
                text = (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
                for token in tokens:
                    self.assertIn(token, text)

    def test_summarizer_reads_benchmark_json_from_input_file(self):
        benchmark_file = ROOT / "target" / "test-hardware-benchmark-summary.json"
        benchmark_file.parent.mkdir(exist_ok=True)
        benchmark_file.write_text(
            json.dumps(
                {
                    "schema_major": 1,
                    "kind": "benchmark",
                    "fixture": "xor",
                    "domain": {"start": 0, "end": 4},
                    "sample_count": 1,
                    "native": {"elapsed_ns": 10, "matches": [1, 3]},
                    "accelerator": {
                        "elapsed_ns": 20,
                        "actual_gpu_sdk": "OpenCL",
                        "mode": "DeviceValidated",
                        "matches": [1, 3],
                        "launch": {"global_size": 256},
                        "telemetry": "OpenCL; driver exit 0",
                    },
                }
            ),
            encoding="utf-8",
        )

        result = subprocess.run(
            ["python", "scripts/summarize_hardware_benchmark.py", "--input", str(benchmark_file)],
            text=True,
            capture_output=True,
            cwd=ROOT,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["accelerator"]["match_count"], 2)

    def test_summarizer_accepts_powershell_utf8_bom_files(self):
        benchmark_file = ROOT / "target" / "test-hardware-benchmark-summary-bom.json"
        benchmark_file.parent.mkdir(exist_ok=True)
        benchmark_file.write_bytes(
            b"\xef\xbb\xbf"
            + json.dumps(
                {
                    "schema_major": 1,
                    "kind": "benchmark",
                    "fixture": "xor",
                    "domain": {"start": 0, "end": 4},
                    "sample_count": 1,
                    "native": {"elapsed_ns": 10, "matches": [1, 3]},
                    "accelerator": {
                        "elapsed_ns": 20,
                        "actual_gpu_sdk": "OpenCL",
                        "mode": "DeviceValidated",
                        "matches": [1, 3],
                        "launch": {"global_size": 256},
                        "telemetry": "OpenCL; driver exit 0",
                    },
                }
            ).encode("utf-8")
        )

        result = subprocess.run(
            ["python", "scripts/summarize_hardware_benchmark.py", "--input", str(benchmark_file)],
            text=True,
            capture_output=True,
            cwd=ROOT,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["accelerator"]["match_count"], 2)


if __name__ == "__main__":
    unittest.main()
