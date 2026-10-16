"""Task 2 本地 MVP 编排：读取、划分、特征、训练、评估和模型保存恢复。

所有运行明确记录 backend=local，当前不创建 SparkSession。未来切换 Spark 时，
需在计时边界内 cache/count 物化特征，并在 finally 中 unpersist/关闭会话；
不能把惰性转换的构建时间当成训练或推理总耗时。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import cast

from project.common.artifacts import ArtifactStore
from project.common.benchmark import capture_hardware
from project.common.contracts import EvaluationResult, ModelBundle, TaskConfig
from project.common.reproducibility import set_random_seed, write_run_manifest
from project.task2.data import audit_reviews, read_reviews, split_reviews
from project.task2.evaluation import evaluate_ratings, evaluate_slices
from project.task2.features import ReviewFeaturePipeline
from project.task2.model import DistributedRatingRegressor
from project.task2.submission import export_rating_submission


def _prepare_run(config: TaskConfig, stage: str) -> TaskConfig:
    """创建北京时间命名的独立运行目录，返回替换 output_dir 后的新配置。

    不修改调用者配置；设置随机种子，写入路径、参数、阶段和 local 后端信息。
    max_train_rows 记录本次训练范围，None 表示全量，不把前 N 行结果称为全量实验。
    """
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d_%H-%M-%S-%f")
    run_dir = config.output_dir / f"{timestamp}_{stage}"
    run_dir.mkdir(parents=True)
    set_random_seed(config.seed)
    write_run_manifest(
        run_dir,
        {
            "stage": stage,
            "task": config.task,
            "backend": "local",
            "seed": config.seed,
            "params": dict(config.params),
            "train_path": str(config.train_path),
            "test_path": str(config.test_path),
            "sample_path": str(config.sample_path),
            "output_dir": str(run_dir),
            "model_dir": str(config.model_dir),
            "max_train_rows": config.params.get("max_train_rows"),
        },
    )
    return replace(config, output_dir=run_dir)


def evaluate_candidate(config: TaskConfig) -> EvaluationResult:
    """从零训练一个候选方案并返回验证 RMSE、切片诊断和真实逐轮日志。

    config.output_dir 是已创建的运行目录；首次保存 split.json，其后各方案直接
    复用 ID。只在训练折 fit 特征，验证表仅 transform；测试文件完全不读取。
    单独记录特征转换耗时，供本地 n_jobs benchmark 比较相同工作量。
    """
    started = perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    raw = read_reviews(
        config.train_path,
        labeled=True,
        max_rows=cast(int | None, config.params.get("max_train_rows")),
    )
    split_path = config.output_dir / "split.json"
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        train = raw[raw["id"].isin(split["train_ids"])]
        valid = raw[raw["id"].isin(split["valid_ids"])]
    else:
        train, valid = split_reviews(
            raw,
            str(config.params.get("split_strategy", "stable_hash_group")),
            float(cast(float, config.params.get("valid_fraction", 0.2))),
            config.seed,
        )
        split_path.write_text(
            json.dumps({"train_ids": train["id"].tolist(), "valid_ids": valid["id"].tolist()}),
            encoding="utf-8",
        )
    features = ReviewFeaturePipeline(config.params)
    features.fit(train)
    transform_started = perf_counter()
    train_features = features.transform_training(train)
    valid_features = features.transform(valid)
    transform_seconds = perf_counter() - transform_started
    preprocessing_seconds = perf_counter() - started
    training_started = perf_counter()
    model = DistributedRatingRegressor(config.params, config.seed)
    epochs = model.fit(train_features, valid_features)
    training_seconds = perf_counter() - training_started
    # 按 ID 接回评分及实体信息，不假设 DataFrame 的索引或行序恰好相同。
    scored = model.predict(valid_features).merge(valid, on="id", how="inner")
    metrics = evaluate_ratings(scored)
    diagnostics = {
        "mae": metrics["mae"],
        **evaluate_slices(scored, train),
        "train_rows": float(len(train)),
        "valid_rows": float(len(valid)),
        "selected_epoch": float(model.selected_epoch),
        "feature_transform_seconds": transform_seconds,
        "preprocessing_seconds": preprocessing_seconds,
        "training_seconds": training_seconds,
        "experiment_total_seconds": perf_counter() - started,
    }
    return EvaluationResult("rmse", metrics["rmse"], diagnostics, epochs)


def run_experiment(config: TaskConfig) -> EvaluationResult:
    """创建运行记录、训练临时模型并保存验证结果；不保存正式模型。

    输入为解析后的 TaskConfig；输出为 EvaluationResult，并在运行目录写入
    manifest.json、split.json、metrics.json。硬件信息与实测指标一起记录。
    """
    config = _prepare_run(config, "experiment")
    result = evaluate_candidate(config)
    ArtifactStore(config.output_dir).write_results(
        {
            "stage": "experiment",
            "backend": "local",
            "result": asdict(result),
            "max_train_rows": config.params.get("max_train_rows"),
            "hardware": capture_hardware(),
        }
    )
    return result


def train_final(config: TaskConfig) -> ModelBundle:
    """在配置选定的全部训练记录上重新拟合特征与模型，并保存独立模型包。

    max_train_rows=None 才表示完整 CSV；MVP 默认前 50000 条。无验证集、不启用
    验证早停，执行配置 epochs；不会自动采用历史实验的最佳 epoch，需手动锁定。
    bundle 状态保存 params/features_directory/model_directory，特征和模型各自
    落盘；运行记录保存实际行数、逐轮日志、硬件及训练/保存时间。
    """
    config = _prepare_run(config, "train-final")
    started = perf_counter()
    train = read_reviews(
        config.train_path,
        labeled=True,
        max_rows=cast(int | None, config.params.get("max_train_rows")),
    )
    features = ReviewFeaturePipeline(config.params)
    features.fit(train)
    train_features = features.transform_training(train)
    model = DistributedRatingRegressor(config.params, config.seed)
    epochs = model.fit(train_features)
    offline_seconds = perf_counter() - started
    save_started = perf_counter()
    store = ArtifactStore(config.output_dir)
    bundle = store.save_bundle(
        {
            "task": "task2",
            "backend": "local",
            "params": dict(config.params),
            "seed": config.seed,
            "train_rows": len(train),
            "features_directory": "features",
            "model_directory": "regressor",
        },
        model_dir=config.model_dir,
    )
    features.save(bundle.directory / "features")
    model.save(bundle.directory / "regressor")
    store.write_results(
        {
            "stage": "train-final",
            "backend": "local",
            "model_directory": str(bundle.directory),
            "audit": audit_reviews(train, labeled=True),
            "max_train_rows": config.params.get("max_train_rows"),
            "epochs": [asdict(epoch) for epoch in epochs],
            "hardware": capture_hardware(),
            "timings": {"offline_total": offline_seconds, "save": perf_counter() - save_started},
        }
    )
    return bundle


def predict_submission(config: TaskConfig, bundle: ModelBundle) -> Path:
    """从模型包恢复特征和回归器，预测全部测试记录并导出单个评分 CSV。

    不读取训练表、不调用 fit。模型选项以包内保存值为准，当前配置只提供路径。
    返回本次 submission.csv 路径，并保存数据审计、推理和导出耗时；max_train_rows
    不限制测试行数。新进程加载同一模型包应得到相同预测。
    """
    config = _prepare_run(config, "predict")
    store = ArtifactStore(config.output_dir)
    started = perf_counter()
    state = store.load_bundle(bundle)
    options = cast(Mapping[str, object], state["params"])
    features = ReviewFeaturePipeline(options)
    features.load(bundle.directory / str(state["features_directory"]))
    model = DistributedRatingRegressor(options, config.seed)
    model.load(bundle.directory / str(state["model_directory"]))
    test = read_reviews(config.test_path, labeled=False)
    predictions = model.predict(features.transform(test))
    inference_seconds = perf_counter() - started
    export_started = perf_counter()
    output = export_rating_submission(
        predictions, config.sample_path, config.output_dir / "submission.csv"
    )
    store.write_results(
        {
            "stage": "predict",
            "backend": "local",
            "submission": str(output),
            "audit": audit_reviews(test, labeled=False),
            "hardware": capture_hardware(),
            "timings": {
                "inference_total": inference_seconds,
                "export": perf_counter() - export_started,
            },
        }
    )
    return output
