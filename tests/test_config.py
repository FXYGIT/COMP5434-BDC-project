"""配置读取的正常流程测试。"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

from project.common.config import load_config


class ConfigTests(TestCase):
    def test_load_relative_paths_and_parameters(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            base = Path(directory).resolve()
            path = base / "task1.json"
            path.write_text(
                json.dumps(
                    {
                        "task": "task1",
                        "train_path": "../data/task1/train.csv",
                        "test_path": "../data/task1/test.csv",
                        "sample_path": "../data/task1/sample_submission.csv",
                        "output_dir": "../runs/task1",
                        "seed": 5434,
                        "params": {"n_trees": 100, "missing_indicators": True},
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.task, "task1")
            self.assertEqual(config.train_path, (base / "../data/task1/train.csv").resolve())
            self.assertEqual(config.test_path, (base / "../data/task1/test.csv").resolve())
            self.assertEqual(
                config.sample_path,
                (base / "../data/task1/sample_submission.csv").resolve(),
            )
            self.assertEqual(config.output_dir, (base / "../runs/task1").resolve())
            self.assertEqual(config.seed, 5434)
            self.assertEqual(config.params, {"n_trees": 100, "missing_indicators": True})
            self.assertFalse(config.output_dir.exists())

    def test_load_absolute_paths(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            base = Path(directory).resolve()
            path = base / "task2.json"
            path.write_text(
                json.dumps(
                    {
                        "task": "task2",
                        "train_path": str(base / "train.csv"),
                        "test_path": str(base / "test.csv"),
                        "sample_path": str(base / "sample.csv"),
                        "output_dir": str(base / "output"),
                        "seed": 42,
                        "params": {"spark_master": "local[*]"},
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.task, "task2")
            self.assertEqual(config.train_path, base / "train.csv")
            self.assertEqual(config.output_dir, base / "output")


if __name__ == "__main__":
    main()
