"""固定划分的基线、消融、参数比较和效率实验；不自动向 Kaggle 提交。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from project.common.contracts import TaskConfig


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
        骨架占位；调用即抛出 NotImplementedError，不执行实际业务。"""
    raise NotImplementedError("仅定义接口，尚未实现")


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
        骨架占位；调用即抛出 NotImplementedError，不执行实际业务。"""
    raise NotImplementedError("仅定义接口，尚未实现")


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
        骨架占位；调用即抛出 NotImplementedError，不执行实际业务。"""
    raise NotImplementedError("仅定义接口，尚未实现")


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
        骨架占位；调用即抛出 NotImplementedError，不执行实际业务。"""
    raise NotImplementedError("仅定义接口，尚未实现")
