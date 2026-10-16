"""天气分类的流程入口；算法、数据处理和保存逻辑委托各自模块。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast
from uuid import uuid4

from project.common.artifacts import ArtifactStore
from project.common.benchmark import BenchmarkRecorder, capture_hardware
from project.common.contracts import (
    EvaluationResult,
    ModelBundle,
    PredictionBatch,
    TaskConfig,
)
from project.common.reproducibility import (
    set_random_seed,
    write_run_manifest,
)
from project.task1.data import WeatherDataset
from project.task1.features import WeatherPreprocessor
from project.task1.model import RandomForestClassifier
from project.task1.submission import export_weather_submission
from project.task1.training import evaluate_candidate, fit_final


def _prepare_run(config: TaskConfig, stage: str) -> TaskConfig:
    """创建独立运行目录，记录配置，返回本次运行配置。

    不修改调用者的 config；后续组件统一使用新 output_dir。
    """
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
    """运行一次候选方案的验证实验，并保存结果、硬件和耗时。

    evaluate_candidate 负责读取/审计数据、固定划分、训练折拟合特征、
    训练模型和验证评估，pipeline 不重复实现这些步骤。
    返回原始 EvaluationResult；experiment_total 包含该完整验证流程。
    """
    config = _prepare_run(config, "experiment")
    store = ArtifactStore(config.output_dir)
    hardware = capture_hardware()
    timer = BenchmarkRecorder()

    timer.start("experiment_total")
    result = evaluate_candidate(config)
    timer.stop("experiment_total")

    store.write_results(
        {
            "stage": "experiment",
            "result": asdict(result),
            "hardware": hardware,
            "timings": timer.summary(),
        }
    )
    return result


def train_final(config: TaskConfig) -> ModelBundle:
    """委托 fit_final 使用全量训练数据重训，返回已保存的模型包。

    fit_final 应使用锁定的超参数，不再切分验证集或重新调参。
    模型包状态约定：params 为训练配置，preprocessor 为特征状态，
    model 为森林状态；对应 predict_submission 的恢复逻辑。
    offline_total 覆盖 fit_final 全流程，包含其模型保存耗时。
    """
    config = _prepare_run(config, "train-final")
    store = ArtifactStore(config.output_dir)
    hardware = capture_hardware()
    timer = BenchmarkRecorder()

    timer.start("offline_total")
    bundle = fit_final(config)
    timer.stop("offline_total")

    store.write_results(
        {
            "stage": "train-final",
            "model_directory": str(bundle.directory),
            "hardware": hardware,
            "timings": timer.summary(),
        }
    )
    return bundle


def predict_submission(config: TaskConfig, bundle: ModelBundle) -> Path:
    """恢复已有特征和森林状态，预测测试集，返回提交 CSV 路径。

    不调用 fit；模型选项使用 bundle 内的 params，而非当前配置的模型参数。
    现有 WeatherDataset.load 同时读取 train/test，故仍需要训练文件，
    但返回的训练记录不参与预测。待数据模块提供独立测试读取接口后可替换。
    inference_total 包含模型恢复、读取与预测；CSV 导出另计。
    """
    config = _prepare_run(config, "predict")
    store = ArtifactStore(config.output_dir)
    hardware = capture_hardware()
    timer = BenchmarkRecorder()

    timer.start("inference_total")
    state = store.load_bundle(bundle)
    options = cast(Mapping[str, object], state["params"])

    preprocessor = WeatherPreprocessor(options)
    preprocessor.load_state_dict(cast(Mapping[str, object], state["preprocessor"]))
    model = RandomForestClassifier(options, config.seed)
    model.load_state_dict(cast(Mapping[str, object], state["model"]))

    dataset = WeatherDataset(config.train_path, config.test_path)
    _, test = dataset.load()
    audit = dataset.audit(test, labeled=False)
    features = preprocessor.transform(test)
    predictions = PredictionBatch(
        ids=[str(row["id"]) for row in test],
        values=model.predict(features),
    )
    timer.stop("inference_total")

    timer.start("export")
    output = export_weather_submission(
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
