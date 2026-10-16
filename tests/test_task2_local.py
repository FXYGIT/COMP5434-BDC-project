"""本地 Task 2 的集成测试：验证划分、状态恢复及所有实验入口。"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
import pandas as pd

from project import experiments
from project.common.contracts import TaskConfig
from project.task2 import pipeline
from project.task2.data import read_reviews, split_reviews
from project.task2.features import ReviewFeaturePipeline
from project.task2.optimization import merge_gradients, partition_gradient, update_parameters


class Task2LocalTests(TestCase):
    """使用带多行文本的小数据集，检查真实本地流程而不是 Spark 替身。"""

    def setUp(self):
        """创建训练、测试和乱序样例；低维特征使测试可快速重复。"""
        self.temp = TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rows = [
            {
                "id": str(i),
                "title": f"review {i}",
                "comment": f'good, "food"\nitem {i}',
                "user_id": f"u{i % 7}",
                "prod_id": f"p{i % 9}",
                "parent_prod_id": f"p{i % 9}",
                "time": 1600000000000 + i,
                "votes": i % 3,
                "purchased": i % 2 == 0,
                "rating": i % 5 + 1,
            }
            for i in range(80)
        ]
        self.frame = pd.DataFrame(rows)
        self.frame.to_csv(self.root / "train.csv", index=False)
        self.frame.iloc[:6].drop(columns="rating").to_csv(self.root / "test.csv", index=False)
        pd.DataFrame({"id": [5, 4, 3, 2, 1, 0], "rating": [3] * 6}).to_csv(
            self.root / "sample.csv", index=False
        )
        self.config = TaskConfig(
            "task2",
            self.root / "train.csv",
            self.root / "test.csv",
            self.root / "sample.csv",
            self.root / "runs",
            5434,
            {
                "text_dimension_per_field": 64,
                "epochs": 2,
                "batches_per_epoch": 2,
                "valid_fraction": 0.25,
                "split_strategy": "stable_hash_group",
                "use_idf": True,
                "clip_predictions": True,
            },
            self.root / "model",
        )

    def test_experiment_and_saved_prediction_without_spark(self):
        """实验真实训练；新对象恢复后无需训练文件，输出覆盖并对齐全部测试 ID。"""
        with patch(
            "project.task2.spark_runtime.create_session", side_effect=AssertionError("Spark")
        ):
            result = pipeline.run_experiment(self.config)
            self.assertTrue(np.isfinite(result.primary_value))
            self.assertEqual(result.primary_name, "rmse")
            self.assertEqual(len(result.epochs), 2)
            bundle = pipeline.train_final(self.config)
            (self.root / "train.csv").unlink()
            output = pipeline.predict_submission(self.config, bundle)
            first = pd.read_csv(output)
            second = pd.read_csv(pipeline.predict_submission(self.config, bundle))
        self.assertEqual(first.id.tolist(), [5, 4, 3, 2, 1, 0])
        self.assertTrue(first.rating.between(1, 5).all())
        pd.testing.assert_frame_equal(first, second)

    def test_experiment_commands_and_report(self):
        """基线、消融和真实本地并行转换全部可运行，报告保留各方案与来源。"""
        baselines = experiments.run_baselines(self.config)
        self.assertEqual(len(baselines), 3)
        ablations = experiments.run_ablations(self.config)
        self.assertGreaterEqual(len(ablations), 5)
        scaling = experiments.run_scaling_study(self.config, [1, 2])
        self.assertEqual(len(scaling), 2)
        self.assertAlmostEqual(scaling[0]["primary_value"], scaling[1]["primary_value"])
        output = experiments.export_report_tables(
            list(self.config.output_dir.iterdir()), self.root / "tables"
        )
        self.assertEqual(
            len(pd.read_csv(output / "results.csv")), len(baselines) + len(ablations) + len(scaling)
        )
        for directory in self.config.output_dir.iterdir():
            metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["backend"], "local")

    def test_group_split_replays_without_duplicate_leakage(self):
        """重复文本组不会跨集合，打乱输入行顺序也不改变验证 ID 集合。"""
        duplicate = self.frame.iloc[[0]].assign(id="duplicate", rating=5)
        frame = pd.concat([self.frame, duplicate], ignore_index=True)
        train, valid = split_reviews(frame, "stable_hash_group", 0.25, 5434)
        _, shuffled_valid = split_reviews(
            frame.sample(frac=1, random_state=2), "stable_hash_group", 0.25, 5434
        )
        self.assertEqual(set(valid.id), set(shuffled_valid.id))
        self.assertEqual(len(train) + len(valid), len(frame))
        self.assertFalse(set(train.id) & set(valid.id))
        self.assertEqual("0" in set(valid.id), "duplicate" in set(valid.id))

    def test_features_ignore_labels_and_restore_exactly(self):
        """验证评分变化不改变特征或训练状态；未知实体和空文本可转换、可重放。"""
        train = self.frame.iloc[:60]
        valid = self.frame.iloc[60:].assign(user_id="unknown", comment="")
        features = ReviewFeaturePipeline(self.config.params)
        features.fit(train)
        original_idf = features.idfs["title"].idf_.copy()
        first = features.transform(valid)
        second = features.transform(valid.assign(rating=99))
        np.testing.assert_allclose(first.matrix.toarray(), second.matrix.toarray())
        np.testing.assert_array_equal(original_idf, features.idfs["title"].idf_)
        features.save(self.root / "features")
        restored = ReviewFeaturePipeline({})
        restored.load(self.root / "features")
        np.testing.assert_allclose(
            first.matrix.toarray(), restored.transform(valid).matrix.toarray()
        )

    def test_read_limit_never_truncates_test_data(self):
        """训练上限显式生效，测试读取仍覆盖全部行，带换行的评论不会被拆行。"""
        self.assertEqual(len(read_reviews(self.config.train_path, True, 3)), 3)
        test = read_reviews(self.config.test_path, False, 3)
        self.assertEqual(len(test), 6)
        self.assertIn("\n", test.iloc[0]["comment"])

    def test_gradient_reference_aggregation_and_update(self):
        """手写参考梯度按样本数聚合，L2 只加入一次，输入权重保持不变。"""
        weights = [0.2, -0.1]
        first = partition_gradient(iter([({0: 1.0, 1: 2.0}, 1.0)]), weights, 0.3)
        second = partition_gradient(iter([({1: 3.0}, 2.0)]), weights, 0.3)
        stats = merge_gradients(first, second)
        np.testing.assert_allclose([stats.gradient_sum[0], stats.gradient_sum[1]], [-0.7, -7.4])
        self.assertAlmostEqual(stats.squared_error_sum, 4.49)
        updated, bias = update_parameters(weights, 0.3, stats, 0.1, 0.5)
        np.testing.assert_allclose(updated, [0.225, 0.275])
        self.assertAlmostEqual(bias, 0.435)
        self.assertEqual(weights, [0.2, -0.1])


if __name__ == "__main__":
    main()
