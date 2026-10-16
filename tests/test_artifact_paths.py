"""验证正式模型与按时间命名的运行目录分离。"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

import numpy as np

from project.common.artifacts import ArtifactStore
from project.common.config import load_config
from project.task1 import pipeline as task1
from project.task2 import pipeline as task2


class ArtifactPathTests(TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        directory = self.root / "configs"
        directory.mkdir()
        path = directory / "task1.json"
        path.write_text(
            json.dumps(
                {
                    "task": "task1",
                    "train_path": "../data/train.csv",
                    "test_path": "../data/test.csv",
                    "sample_path": "../data/sample.csv",
                    "output_dir": "../runs/task1",
                    "model_dir": "../models/task1/forest",
                    "seed": 5434,
                    "params": {},
                }
            ),
            encoding="utf-8",
        )
        self.config = load_config(path)

    def test_config_resolves_independent_model_directory(self):
        self.assertEqual(self.config.model_dir, self.root / "models/task1/forest")
        self.assertFalse(self.config.model_dir.exists())

    def test_model_save_and_load_outside_run_directory(self):
        run = self.root / "runs/train"
        model = self.root / "models/task1/forest"
        store = ArtifactStore(run)
        bundle = store.save_bundle({"weights": np.array([1.0, 2.0])}, model_dir=model)
        self.assertEqual(bundle.directory, model)
        self.assertFalse((run / "model").exists())
        state = ArtifactStore(self.root / "runs/predict").load_bundle(bundle)
        np.testing.assert_array_equal(state["weights"], np.array([1.0, 2.0]))
        self.assertEqual(store.write_results({"score": 1}), run / "metrics.json")

    def test_timestamp_run_names_preserve_model_directory(self):
        for pipeline in [task1, task2]:
            with self.subTest(pipeline=pipeline.__name__):
                prepared = pipeline._prepare_run(self.config, "experiment")
                self.assertRegex(
                    prepared.output_dir.name,
                    r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{6}_experiment$",
                )
                self.assertEqual(prepared.model_dir, self.config.model_dir)
                self.assertEqual(prepared.output_dir.parent, self.config.output_dir)
                manifest = json.loads(
                    (prepared.output_dir / "manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(manifest["model_dir"], str(self.config.model_dir))


if __name__ == "__main__":
    main()
