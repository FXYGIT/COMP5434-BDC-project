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
        Task 1 MVP 使用第三方模型和 pandas 汇总；Task 2 实验待实现。"""
    if config.task == "task2":
        raise NotImplementedError("Task 2 实验待实现")
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
    """Task 1 对比类别权重/缺失指示/森林与单树；Task 2 去除 title、comment、元数据或 IDF；每次只改变一个因素。

    参数：
        config (TaskConfig)：已解析并通过字段校验的任务配置
    返回：
        Sequence[Mapping[str, object]]：按类型标注返回结构化结果；字段含义及顺序见本函数职责和公共数据契约。
    实现边界：
        只实现本接口声明的职责；训练/验证隔离、ID 对齐及状态恢复遵循架构文档。
        不允许静默丢弃测试记录、把未测数据填零或生成虚构评估结果。
    当前状态：
        Task 1 MVP 使用第三方模型和 pandas 汇总；Task 2 实验待实现。"""
    if config.task == "task2":
        raise NotImplementedError("Task 2 实验待实现")
    return _run_variants(
        config,
        [
            ("full", {}),
            ("no_class_weight", {"class_weight": None}),
            ("no_missing_indicators", {"missing_indicators": False}),
            ("single_tree", {"model": "single_tree"}),
        ],
        "ablate",
    )


def run_scaling_study(
    config: TaskConfig, parallelisms: Sequence[int]
) -> Sequence[Mapping[str, object]]:
    """固定数据、模型、epoch 和缓存口径，改变 Spark 并行度；报告耗时、加速比和指标变化；不得把不同工作量直接比较。

    参数：
        config (TaskConfig)：已解析并通过字段校验的任务配置
        parallelisms (Sequence[int])：效率实验所用的正整数并行度列表
    返回：
        Sequence[Mapping[str, object]]：按类型标注返回结构化结果；字段含义及顺序见本函数职责和公共数据契约。
    实现边界：
        只实现本接口声明的职责；训练/验证隔离、ID 对齐及状态恢复遵循架构文档。
        不允许静默丢弃测试记录、把未测数据填零或生成虚构评估结果。
    当前状态：
        Task 1 MVP 使用第三方模型和 pandas 汇总；Task 2 实验待实现。"""
    if config.task == "task2":
        raise NotImplementedError("Task 2 Spark 并行实验待实现")
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
        Task 1 MVP 使用第三方模型和 pandas 汇总；Task 2 实验待实现。"""
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
    # 共用随机种子和同一份 split.json，确保对比的是相同验证记录。
    run_config = _prepare_run(config, stage)
    results: list[dict[str, Any]] = []
    for name, changes in variants:
        candidate = replace(run_config, params={**config.params, **changes})
        result = evaluate_candidate(candidate)
        results.append({"variant": name, **asdict(result)})
    if stage == "benchmark":
        reference = results[0]["diagnostics"]["training_seconds"]
        for row in results:
            row["diagnostics"]["speedup"] = reference / row["diagnostics"]["training_seconds"]
    ArtifactStore(run_config.output_dir).write_results({"stage": stage, "results": results})
    return results
