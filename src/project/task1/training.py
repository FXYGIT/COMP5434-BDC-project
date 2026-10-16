"""验证用于模型选择，最终模型在全量训练数据重训。"""

from __future__ import annotations

import json
from dataclasses import asdict
from time import perf_counter
from typing import Any, cast

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier

from project.common.artifacts import ArtifactStore
from project.common.contracts import EvaluationResult, ModelBundle, SplitIds, TaskConfig
from project.task1.data import WeatherDataset
from project.task1.features import WeatherPreprocessor
from project.task1.model import RandomForestClassifier


def _options(config: TaskConfig) -> dict[str, object]:
    options = dict(config.params)
    options.setdefault("n_estimators", options.get("n_trees", 100))
    options.setdefault("min_samples_leaf", options.get("min_leaf_samples", 1))
    return options


def _candidate(options: dict[str, object], seed: int) -> Any:
    # MVP 直接调用 sklearn；后续用自定义多数类统计、CART 和森林替换。
    if options.get("model") == "majority":
        return DummyClassifier(strategy="most_frequent")
    if options.get("model") == "single_tree":
        return DecisionTreeClassifier(
            max_depth=options.get("max_depth"),
            min_samples_leaf=options["min_samples_leaf"],
            class_weight=options.get("class_weight"),
            random_state=seed,
        )
    return RandomForestClassifier(options, seed)


def _load_train(config: TaskConfig) -> pd.DataFrame:
    # 验证和训练均不需要读取测试数据。
    frame = pd.read_csv(config.train_path)
    frame["id"] = frame["id"].astype(str)
    return frame


def evaluate_candidate(config: TaskConfig) -> EvaluationResult:
    """固定并保存划分，仅在训练折拟合，返回真实四分类验证指标。"""
    frame = _load_train(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    split_path = config.output_dir / "split.json"
    if split_path.exists():
        saved = json.loads(split_path.read_text(encoding="utf-8"))
        split = SplitIds(**saved)
    else:
        dataset = WeatherDataset(config.train_path, config.test_path)
        split = dataset.split(
            frame.to_dict("records"),
            float(cast(float, config.params.get("valid_fraction", 0.2))),
            config.seed,
        )
        split_path.write_text(json.dumps(asdict(split), indent=2), encoding="utf-8")
    train = frame[frame["id"].isin(split.train_ids)]
    valid = frame[frame["id"].isin(split.valid_ids)]
    options = _options(config)
    preprocessor = WeatherPreprocessor(options)
    started = perf_counter()
    preprocessor.fit(train.to_dict("records"))
    x_train = preprocessor.transform(train.to_dict("records"))
    model = _candidate(options, config.seed)
    model.fit(x_train, train["label"].tolist())
    training_seconds = perf_counter() - started
    predicted = model.predict(preprocessor.transform(valid.to_dict("records")))
    truth = valid["label"].tolist()
    labels = [1, 2, 3, 4]
    diagnostics = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "training_seconds": training_seconds,
    }
    for label, score in zip(
        labels,
        f1_score(truth, predicted, labels=labels, average=None, zero_division=0),
        strict=False,
    ):
        diagnostics[f"f1_class_{label}"] = float(score)
    matrix = confusion_matrix(truth, predicted, labels=labels)
    for i, actual in enumerate(labels):
        for j, prediction in enumerate(labels):
            diagnostics[f"confusion_{actual}_{prediction}"] = float(matrix[i, j])
    return EvaluationResult(
        "macro_f1",
        float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
        diagnostics,
        [],
    )


def fit_final(config: TaskConfig) -> ModelBundle:
    """全量训练并保存预测入口可直接恢复的特征和模型状态。"""
    frame = _load_train(config)
    rows = frame.to_dict("records")
    options = _options(config)
    preprocessor = WeatherPreprocessor(options)
    preprocessor.fit(rows)
    model = RandomForestClassifier(options, config.seed)
    model.fit(preprocessor.transform(rows), frame["label"].tolist())
    return ArtifactStore(config.output_dir).save_bundle(
        {
            "params": options,
            "seed": config.seed,
            "preprocessor": preprocessor.state_dict(),
            "model": model.state_dict(),
        },
        model_dir=config.model_dir,
    )
