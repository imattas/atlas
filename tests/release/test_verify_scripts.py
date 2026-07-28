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


if __name__ == "__main__":
    unittest.main()
