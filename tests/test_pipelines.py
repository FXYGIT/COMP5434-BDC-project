"""Task 1 流程编排测试；Task 2 本地集成覆盖见 test_task2_local.py。"""

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import DEFAULT, patch

from project.common.contracts import EvaluationResult, ModelBundle, TaskConfig
from project.task1 import pipeline as task1


@contextmanager
def dependencies(module, names):
    """隔离未实现的依赖，不替代被测 pipeline 函数。"""
    with patch.multiple(module, create=True, **dict.fromkeys(names, DEFAULT)) as mocks:
        yield mocks


COMMON = [
    "set_random_seed",
    "ArtifactStore",
    "BenchmarkRecorder",
    "capture_hardware",
    "write_run_manifest",
]


class PipelineTests(TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = TaskConfig(
            "task1",
            self.root / "train.csv",
            self.root / "test.csv",
            self.root / "sample.csv",
            self.root / "runs",
            5434,
            {"valid_fraction": 0.2, "split_strategy": "stable_hash_group"},
            model_dir=self.root / "models",
        )
        self.bundle = ModelBundle(self.root / "model", self.root / "model/manifest.json")

    def test_task1_experiment_delegates_training(self):
        expected = EvaluationResult("macro_f1", 0.8, {}, [])
        with dependencies(task1, COMMON + ["evaluate_candidate"]) as dep:
            dep["evaluate_candidate"].return_value = expected
            result = task1.run_experiment(self.config)
            self.assertIs(result, expected)
            actual_config = dep["evaluate_candidate"].call_args.args[0]
            self.assertEqual(actual_config.output_dir.parent, self.config.output_dir)
            dep["set_random_seed"].assert_called_once_with(5434)
            dep["ArtifactStore"].return_value.write_results.assert_called_once()

    def test_task1_final_delegates_full_training(self):
        with dependencies(task1, COMMON + ["fit_final"]) as dep:
            dep["fit_final"].return_value = self.bundle
            self.assertIs(task1.train_final(self.config), self.bundle)
            dep["fit_final"].assert_called_once()

    def test_task1_prediction_restores_without_fitting(self):
        names = COMMON + [
            "WeatherDataset",
            "WeatherPreprocessor",
            "RandomForestClassifier",
            "export_weather_submission",
        ]
        with dependencies(task1, names) as dep:
            dep["ArtifactStore"].return_value.load_bundle.return_value = {
                "params": {},
                "preprocessor": {"feature": "state"},
                "model": {"tree": "state"},
            }
            dep["WeatherDataset"].return_value.load.return_value = (
                [],
                [{"id": "x"}, {"id": "y"}],
            )
            dep["RandomForestClassifier"].return_value.predict.return_value = [1, 4]
            dep["export_weather_submission"].return_value = self.root / "submission.csv"
            output = task1.predict_submission(self.config, self.bundle)
            self.assertEqual(output, self.root / "submission.csv")
            dep["WeatherPreprocessor"].return_value.load_state_dict.assert_called_once_with(
                {"feature": "state"}
            )
            dep["RandomForestClassifier"].return_value.load_state_dict.assert_called_once_with(
                {"tree": "state"}
            )
            dep["WeatherPreprocessor"].return_value.fit.assert_not_called()
            dep["RandomForestClassifier"].return_value.fit.assert_not_called()
            batch = dep["export_weather_submission"].call_args.args[0]
            self.assertEqual(list(batch.ids), ["x", "y"])
            self.assertEqual(list(batch.values), [1, 4])


if __name__ == "__main__":
    main()
