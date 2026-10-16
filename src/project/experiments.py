"""固定划分的基线、消融、参数比较和效率实验；不自动向 Kaggle 提交。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd

from project.common.artifacts import ArtifactStore
from project.common.contracts import TaskConfig
from project.task1.pipeline import _prepare_run
from project.task1.training import evaluate_candidate
from project.task2 import pipeline as task2


def run_baselines(config: TaskConfig) -> Sequence[Mapping[str, object]]:
    """Task 1 实现多数类、简单自定义单树；Task 2 实现训练全局均值、文本模型；基线和主模型共用划分及计时口径。

    参数：
        config (TaskConfig)：已解析并通过字段校验的任务配置
    返回：
        Sequence[Mapping[str, object]]：按类型标注返回结构化结果；字段含义及顺序见本函数职责和公共数据契约。
    实现边界：
        只实现本接口声明的职责；训练/验证隔离、ID 对齐及状态恢复遵循架构文档。
        不允许静默丢弃测试记录、把未测数据填零或生成虚构评估结果。
    当前状态：
        Task 1/2 MVP 使用本地第三方模型和 pandas 汇总；Spark 并行实验待接入。"""
    if config.task == "task2":
        return _run_variants(
            config,
            [
                (
                    "global_mean",
                    {
                        "model": "global_mean",
                        "use_title": False,
                        "use_comment": False,
                        "use_metadata": False,
                    },
                ),
                ("text_only", {"use_metadata": False}),
                ("text_metadata", {}),
            ],
            "baseline",
        )
    return _run_variants(
        config,
        [
            ("majority", {"model": "majority"}),
            ("single_tree", {"model": "single_tree"}),
            ("random_forest", {}),
        ],
        "baseline",
    )


def run_ablations(config: TaskConfig) -> Sequence[Mapping[str, object]]:
    """Task 1 对比类别权重/bootstrap/森林与单树；Task 2 去除 title、comment、元数据或 IDF；每次只改变一个因素。

    参数：
        config (TaskConfig)：已解析并通过字段校验的任务配置
    返回：
        Sequence[Mapping[str, object]]：按类型标注返回结构化结果；字段含义及顺序见本函数职责和公共数据契约。
    实现边界：
        只实现本接口声明的职责；训练/验证隔离、ID 对齐及状态恢复遵循架构文档。
        不允许静默丢弃测试记录、把未测数据填零或生成虚构评估结果。
    当前状态：
        Task 1/2 MVP 使用本地第三方模型和 pandas 汇总；Spark 并行实验待接入。"""
    if config.task == "task2":
        return _run_variants(
            config,
            [
                ("full", {}),
                ("no_title", {"use_title": False}),
                ("no_comment", {"use_comment": False}),
                ("no_metadata", {"use_metadata": False}),
                ("no_idf", {"use_idf": False}),
                ("no_clipping", {"clip_predictions": False}),
            ],
            "ablate",
        )
    return _run_variants(
        config,
        [
            ("full", {}),
            ("no_class_weight", {"class_weight": None}),
            # 不再用 missing_indicators 做消融：本数据没有缺失值，7 个掩码列恒为 0，
            # 而模型又会把整列恒定的特征从候选池里剔除，于是 full 和 no_missing_indicators
            # 拿到的是同一份候选池、建出完全相同的树，差异恒为 0，这一臂没有信息量。
            ("no_bootstrap", {"bootstrap": False}),
            ("single_tree", {"model": "single_tree"}),
        ],
        "ablate",
    )


def run_scaling_study(
    config: TaskConfig, parallelisms: Sequence[int]
) -> Sequence[Mapping[str, object]]:
    """固定数据、模型和缓存口径，只改并行度；报告耗时、加速比和指标变化；不得把不同工作量直接比较。

    task1 改的是手写森林的按树线程数（n_jobs），task2 改的是 Spark 并行度。

    参数：
        config (TaskConfig)：已解析并通过字段校验的任务配置
        parallelisms (Sequence[int])：效率实验所用的正整数并行度列表
    返回：
        Sequence[Mapping[str, object]]：按类型标注返回结构化结果；字段含义及顺序见本函数职责和公共数据契约。
    实现边界：
        只实现本接口声明的职责；训练/验证隔离、ID 对齐及状态恢复遵循架构文档。
        不允许静默丢弃测试记录、把未测数据填零或生成虚构评估结果。
    当前状态：
        Task 1/2 MVP 使用本地第三方模型和 pandas 汇总；Spark 并行实验待接入。"""
    if config.task == "task2":
        # 本地 MVP 测量实际 joblib 特征转换线程；固定 epoch 并禁用早停。
        return _run_variants(
            config,
            [
                (f"n_jobs_{workers}", {"n_jobs": workers, "early_stopping_patience": 0})
                for workers in parallelisms
            ],
            "benchmark",
        )
    # Task 1 用 sklearn 的 n_jobs 比较 CPU 并行度，工作量及划分保持一致。
    results = _run_variants(
        config,
        [(f"n_jobs_{workers}", {"n_jobs": workers}) for workers in parallelisms],
        "benchmark",
    )
    return results


def export_report_tables(run_directories: Sequence[Path], output_dir: Path) -> Path:
    """读取实测结果输出可追溯的报告表格；缺失数据标为未测，禁止生成虚构分数或耗时。

    参数：
        run_directories (Sequence[Path])：已完成实验的输出目录列表，表格必须追踪到这些运行
        output_dir (Path)：输出目录；不同实验使用独立 run ID，避免覆盖
    返回：
        Path：输出文件或目录的路径，成功返回前应完成相关校验。
    实现边界：
        只实现本接口声明的职责；训练/验证隔离、ID 对齐及状态恢复遵循架构文档。
        不允许静默丢弃测试记录、把未测数据填零或生成虚构评估结果。
    当前状态：
        Task 1/2 MVP 使用本地第三方模型和 pandas 汇总；Spark 并行实验待接入。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for directory in run_directories:
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        for result in metrics.get("results", [metrics]):
            rows.append({"run_directory": str(directory), "stage": metrics["stage"], **result})
    pd.json_normalize(rows).to_csv(output_dir / "results.csv", index=False)
    return output_dir


def _run_variants(
    config: TaskConfig,
    variants: Sequence[tuple[str, Mapping[str, object]]],
    stage: str,
) -> list[dict[str, Any]]:
    """在同一划分上运行配置变体，保存可汇总的结果并返回逐方案字典。

    config 指定基础参数；variants 为名称与参数覆盖列表；stage 指定运行阶段。
    Task 2 单独记录 local 后端及数据上限。benchmark 加速比以列表首项为参照，
    Task 1 测训练时间，Task 2 测本地特征转换时间，不声称测量了 Spark 加速。
    """
    # 共用随机种子和同一份 split.json，确保对比的是相同验证记录。
    prepare = task2._prepare_run if config.task == "task2" else _prepare_run
    evaluate = task2.evaluate_candidate if config.task == "task2" else evaluate_candidate
    run_config = prepare(config, stage)
    results: list[dict[str, Any]] = []
    for name, changes in variants:
        candidate = replace(run_config, params={**config.params, **changes})
        result = evaluate(candidate)
        results.append({"variant": name, **asdict(result)})
    if stage == "benchmark":
        timing = "feature_transform_seconds" if config.task == "task2" else "training_seconds"
        reference = results[0]["diagnostics"][timing]
        for row in results:
            row["diagnostics"]["speedup"] = reference / row["diagnostics"][timing]
    ArtifactStore(run_config.output_dir).write_results(
        {
            "stage": stage,
            "backend": "local",
            "results": results,
            "max_train_rows": config.params.get("max_train_rows"),
        }
    )
    return results
