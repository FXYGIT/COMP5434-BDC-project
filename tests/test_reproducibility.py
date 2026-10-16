"""复现工具的行为测试，不依赖真实训练任务。"""

import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

import numpy as np

from project.common import reproducibility


class ReproducibilityTests(TestCase):
    def test_python_and_numpy_seed(self):
        reproducibility.set_random_seed(5434)
        expected_python = [random.random(), random.random()]
        expected_numpy = np.random.random(3)
        reproducibility.set_random_seed(5434)
        self.assertEqual([random.random(), random.random()], expected_python)
        np.testing.assert_array_equal(np.random.random(3), expected_numpy)

    def test_manifest_preserves_metadata_and_unicode_paths(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            output = Path(directory) / "run"
            metadata = {
                "task": "task1",
                "params": {"description": "天气分类"},
                "file": Path("data/train.csv"),
            }
            path = reproducibility.write_run_manifest(output, metadata)
            result = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(path, output / "manifest.json")
            self.assertEqual(result["task"], "task1")
            self.assertEqual(result["params"], metadata["params"])
            self.assertEqual(result["file"], str(metadata["file"]))
            self.assertIn("created_at", result)
            self.assertNotIn("created_at", metadata)


if __name__ == "__main__":
    main()
