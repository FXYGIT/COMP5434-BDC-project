"""评论评分的 Spark 流程编排；依赖模块负责各自的算法实现。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from project.common.artifacts import ArtifactStore
from project.common.benchmark import BenchmarkRecorder, capture_hardware
from project.common.contracts import EvaluationResult, ModelBundle, TaskConfig
from project.common.reproducibility import (
    set_random_seed,
    write_run_manifest,
)
from project.task2.data import audit_reviews, read_reviews, split_reviews
from project.task2.evaluation import evaluate_ratings, evaluate_slices
from project.task2.features import ReviewFeaturePipeline
from project.task2.model import DistributedRatingRegressor
from project.task2.spark_runtime import close_session, create_session, describe_cluster
from project.task2.submission import export_rating_submission

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


def _prepare_run(config: TaskConfig, stage: str) -> TaskConfig:
    """创建独立运行目录并记录配置；不修改原配置对象。"""
    run_dir = config.output_dir / f"{stage}-{uuid4().hex[:12]}"
    run_dir.mkdir(parents=True)
    run_config = replace(config, output_dir=run_dir)
    set_random_seed(config.seed)
    write_run_manifest(
        run_dir,
        {
            "stage": stage,
            "task": config.task,
            "seed": config.seed,
            "params": dict(config.params),
            "train_path": str(config.train_path),
            "test_path": str(config.test_path),
            "sample_path": str(config.sample_path),
            "output_dir": str(run_dir),
        },
    )
    return run_config


def run_experiment(config: TaskConfig) -> EvaluationResult:
    """训练折拟合特征与模型，在验证折评估 RMSE，返回实验结果。

    保存划分 ID、数据审计、epoch 日志、硬件和阶段耗时。
    训练/验证特征及最终评分表缓存后 count，避免只测量 Spark 惰性转换。
    preprocessing 包含读取、划分、特征拟合和物化；training 包含模型内部验证；
    experiment_total 还包含最终评估，不把它称为最终模型的离线时间。
    """
    config = _prepare_run(config, "experiment")
    store = ArtifactStore(config.output_dir)
    timer = BenchmarkRecorder()
    spark = create_session(config.params)
    cached: list[DataFrame] = []

    try:
        hardware = {"machine": capture_hardware(), "spark": describe_cluster(spark)}
        timer.start("experiment_total")
        timer.start("preprocessing")
        raw = read_reviews(spark, config.train_path, labeled=True)
        audit = audit_reviews(raw, labeled=True)
        train, valid = split_reviews(
            raw,
            str(config.params["split_strategy"]),
            float(cast(float, config.params["valid_fraction"])),
            config.seed,
        )
        train.select("id").write.mode("error").parquet(str(config.output_dir / "split/train"))
        valid.select("id").write.mode("error").parquet(str(config.output_dir / "split/valid"))

        features = ReviewFeaturePipeline(config.params)
        features.fit(train)
        train_features = features.transform_training(train).cache()
        cached.append(train_features)
        train_features.count()
        valid_features = features.transform(valid).cache()
        cached.append(valid_features)
        valid_features.count()
        timer.stop("preprocessing")

        timer.start("training")
        model = DistributedRatingRegressor(config.params, config.seed)
        epochs = model.fit(train_features, valid_features)
        timer.stop("training")

        timer.start("evaluation")
        # 模型只返回 id,prediction；按 ID 接回真实评分及用户/商品信息。
        predictions = model.predict(valid_features).join(valid, on="id", how="inner").cache()
        cached.append(predictions)
        predictions.count()
        metrics = evaluate_ratings(predictions)
        slices = evaluate_slices(predictions, train)
        timer.stop("evaluation")
        timer.stop("experiment_total")

        diagnostics = {key: value for key, value in metrics.items() if key != "rmse"}
        diagnostics.update(slices)
        result = EvaluationResult("rmse", metrics["rmse"], diagnostics, epochs)
        store.write_results(
            {
                "stage": "experiment",
                "result": asdict(result),
                "audit": audit,
                "hardware": hardware,
                "timings": timer.summary(),
            }
        )
        return result
    finally:
        # 使用同一个会话处理所有步骤；失败也释放已缓存数据和 Spark。
        try:
            for frame in cached:
                frame.unpersist()
        finally:
            close_session(spark)


def train_final(config: TaskConfig) -> ModelBundle:
    """在完整训练集重建特征并训练最终模型，保存可用于预测的模型包。

    使用配置中锁定的 epoch，不再切分验证集；fit(train_features) 不启用验证早停。
    bundle 状态保存 params、features_directory 和 model_directory；后两者
    相对 bundle.directory。实际状态分别由 features.save 和 model.save 保存。
    offline_total 包含读取、审计、特征处理与训练，不含 Spark 启动及模型落盘。
    """
    config = _prepare_run(config, "train-final")
    store = ArtifactStore(config.output_dir)
    timer = BenchmarkRecorder()
    spark = create_session(config.params)
    cached: list[DataFrame] = []

    try:
        hardware = {"machine": capture_hardware(), "spark": describe_cluster(spark)}
        timer.start("offline_total")
        timer.start("preprocessing")
        train = read_reviews(spark, config.train_path, labeled=True)
        audit = audit_reviews(train, labeled=True)
        features = ReviewFeaturePipeline(config.params)
        features.fit(train)
        train_features = features.transform_training(train).cache()
        cached.append(train_features)
        train_features.count()
        timer.stop("preprocessing")

        timer.start("training")
        model = DistributedRatingRegressor(config.params, config.seed)
        epochs = model.fit(train_features)
        timer.stop("training")
        timer.stop("offline_total")

        timer.start("save")
        bundle = store.save_bundle(
            {
                "task": "task2",
                "params": dict(config.params),
                "features_directory": "features",
                "model_directory": "regressor",
            }
        )
        features.save(bundle.directory / "features")
        model.save(bundle.directory / "regressor")
        timer.stop("save")

        store.write_results(
            {
                "stage": "train-final",
                "model_directory": str(bundle.directory),
                "audit": audit,
                "epochs": [asdict(epoch) for epoch in epochs],
                "hardware": hardware,
                "timings": timer.summary(),
            }
        )
        return bundle
    finally:
        try:
            for frame in cached:
                frame.unpersist()
        finally:
            close_session(spark)


def predict_submission(config: TaskConfig, bundle: ModelBundle) -> Path:
    """恢复模型和特征状态，预测全部测试记录并导出单个评分 CSV。

    不读取训练数据，不调用 fit。模型和特征使用训练时保存的 params；
    当前 config.params 仅用于创建此次 Spark 运行环境。
    inference_total 包含状态加载、测试读取、特征转换与完整预测物化；
    不含 Spark 启动，CSV 导出另计；返回提交 CSV 路径。
    """
    config = _prepare_run(config, "predict")
    store = ArtifactStore(config.output_dir)
    timer = BenchmarkRecorder()
    spark = create_session(config.params)
    cached: list[DataFrame] = []

    try:
        hardware = {"machine": capture_hardware(), "spark": describe_cluster(spark)}
        timer.start("inference_total")
        state = store.load_bundle(bundle)
        options = cast(Mapping[str, object], state["params"])
        features = ReviewFeaturePipeline(options)
        features.load(bundle.directory / str(state["features_directory"]))
        model = DistributedRatingRegressor(options, config.seed)
        model.load(bundle.directory / str(state["model_directory"]))

        test = read_reviews(spark, config.test_path, labeled=False)
        audit = audit_reviews(test, labeled=False)
        test_features = features.transform(test)
        predictions = model.predict(test_features).cache()
        cached.append(predictions)
        predictions.count()
        timer.stop("inference_total")

        timer.start("export")
        output = export_rating_submission(
            spark,
            predictions,
            config.sample_path,
            config.output_dir / "submission.csv",
        )
        timer.stop("export")
        store.write_results(
            {
                "stage": "predict",
                "submission": str(output),
                "audit": audit,
                "hardware": hardware,
                "timings": timer.summary(),
            }
        )
        return output
    finally:
        try:
            for frame in cached:
                frame.unpersist()
        finally:
            close_session(spark)
