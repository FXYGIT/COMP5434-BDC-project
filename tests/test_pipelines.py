"""流程编排测试：依赖模块仍为骨架，用替身验证调用和数据流。"""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import DEFAULT, MagicMock, patch

from project.common.contracts import EvaluationResult, ModelBundle, TaskConfig
from project.task1 import pipeline as task1
from project.task2 import pipeline as task2


@contextmanager
def dependencies(module, names):
    """隔离未实现的依赖，不替代被测 pipeline 函数。"""
    with patch.multiple(module, create=True, **dict.fromkeys(names, DEFAULT)) as mocks:
        yield mocks


def frame(name):
    result = MagicMock(name=name)
    result.cache.return_value = result
    return result


COMMON = [
    "set_random_seed",
    "ArtifactStore",
    "BenchmarkRecorder",
    "capture_hardware",
    "write_run_manifest",
]
TASK2 = COMMON + [
    "create_session",
    "close_session",
    "describe_cluster",
    "read_reviews",
    "audit_reviews",
    "split_reviews",
    "ReviewFeaturePipeline",
    "DistributedRatingRegressor",
    "evaluate_ratings",
    "evaluate_slices",
    "export_rating_submission",
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

    def test_task2_experiment_uses_training_only_and_materializes(self):
        config = replace(self.config, task="task2")
        raw, train, valid = frame("raw"), frame("train"), frame("valid")
        train_x, valid_x, predictions, scored = (
            frame("train_x"),
            frame("valid_x"),
            frame("pred"),
            frame("scored"),
        )
        with dependencies(task2, TASK2) as dep:
            dep["read_reviews"].return_value = raw
            dep["split_reviews"].return_value = train, valid
            feature = dep["ReviewFeaturePipeline"].return_value
            feature.transform_training.return_value = train_x
            feature.transform.return_value = valid_x
            model = dep["DistributedRatingRegressor"].return_value
            model.fit.return_value = []
            model.predict.return_value = predictions
            predictions.join.return_value = scored
            dep["evaluate_ratings"].return_value = {"rmse": 0.5, "mae": 0.3}
            dep["evaluate_slices"].return_value = {"cold_rmse": 0.7}
            result = task2.run_experiment(config)
            self.assertEqual(result.primary_value, 0.5)
            feature.fit.assert_called_once_with(train)
            feature.transform_training.assert_called_once_with(train)
            feature.transform.assert_called_once_with(valid)
            model.fit.assert_called_once_with(train_x, valid_x)
            predictions.join.assert_called_once_with(valid, on="id", how="inner")
            train_x.count.assert_called_once()
            valid_x.count.assert_called_once()
            scored.count.assert_called_once()
            for cached in [train_x, valid_x, scored]:
                cached.unpersist.assert_called_once()
            dep["close_session"].assert_called_once_with(dep["create_session"].return_value)

    def test_task2_final_saves_features_and_model(self):
        config = replace(self.config, task="task2")
        train, train_x = frame("train"), frame("train_x")
        with dependencies(task2, TASK2) as dep:
            dep["read_reviews"].return_value = train
            dep["ReviewFeaturePipeline"].return_value.transform_training.return_value = train_x
            dep["DistributedRatingRegressor"].return_value.fit.return_value = []
            dep["ArtifactStore"].return_value.save_bundle.return_value = self.bundle
            self.assertIs(task2.train_final(config), self.bundle)
            dep["DistributedRatingRegressor"].return_value.fit.assert_called_once_with(train_x)
            dep["ReviewFeaturePipeline"].return_value.save.assert_called_once_with(
                self.bundle.directory / "features"
            )
            dep["DistributedRatingRegressor"].return_value.save.assert_called_once_with(
                self.bundle.directory / "regressor"
            )
            train_x.unpersist.assert_called_once()
            dep["close_session"].assert_called_once()

    def test_task2_prediction_restores_without_fitting(self):
        config = replace(self.config, task="task2")
        test, predictions = frame("test"), frame("predictions")
        with dependencies(task2, TASK2) as dep:
            dep["ArtifactStore"].return_value.load_bundle.return_value = {
                "params": {"saved": True},
                "features_directory": "features",
                "model_directory": "regressor",
            }
            dep["read_reviews"].return_value = test
            dep["DistributedRatingRegressor"].return_value.predict.return_value = predictions
            dep["export_rating_submission"].return_value = self.root / "submission.csv"
            output = task2.predict_submission(config, self.bundle)
            self.assertEqual(output, self.root / "submission.csv")
            dep["read_reviews"].assert_called_once_with(
                dep["create_session"].return_value, config.test_path, labeled=False
            )
            dep["ReviewFeaturePipeline"].return_value.load.assert_called_once_with(
                self.bundle.directory / "features"
            )
            dep["DistributedRatingRegressor"].return_value.load.assert_called_once_with(
                self.bundle.directory / "regressor"
            )
            dep["ReviewFeaturePipeline"].return_value.fit.assert_not_called()
            dep["DistributedRatingRegressor"].return_value.fit.assert_not_called()
            predictions.count.assert_called_once()
            predictions.unpersist.assert_called_once()
            dep["close_session"].assert_called_once()

    def test_task2_training_failure_closes_session_and_unpersists(self):
        config = replace(self.config, task="task2")
        train_x = frame("train_x")
        with dependencies(task2, TASK2) as dep:
            dep["ReviewFeaturePipeline"].return_value.transform_training.return_value = train_x
            dep["DistributedRatingRegressor"].return_value.fit.side_effect = RuntimeError(
                "training failed"
            )
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                task2.train_final(config)
            train_x.unpersist.assert_called_once()
            dep["close_session"].assert_called_once()


if __name__ == "__main__":
    main()
