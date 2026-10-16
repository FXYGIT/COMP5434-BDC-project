"""自定义同步分布式小批次梯度算法的纯计算契约。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from project.common.contracts import SparseVector


@dataclass(frozen=True)
class GradientStats:
    """分区数据损失及梯度的和；正则化在全局更新时仅添加一次。"""

    gradient_sum: Mapping[int, float]
    bias_gradient_sum: float
    squared_error_sum: float
    count: int


def partition_gradient(
    rows: Iterable[tuple[SparseVector, float]], weights: Sequence[float], bias: float
) -> GradientStats:
    """在 executor 内流式计算线性评分、平方损失及稀疏梯度和；不 list 化整分区，不在 executor 修改 driver 参数。

    参数：
        rows (Iterable[tuple[SparseVector, float]])：输入记录；保持 ID 对齐，按当前函数契约决定是否含标签
        weights (Sequence[float])：当前全局模型权重，仅以只读形式传给 executor
        bias (float)：当前线性模型偏置
    返回：
        GradientStats：梯度和、偏置梯度和、平方误差和及样本数。
    实现边界：
        只实现本接口声明的职责；训练/验证隔离、ID 对齐及状态恢复遵循架构文档。
        不允许静默丢弃测试记录、把未测数据填零或生成虚构评估结果。
    当前状态：
        骨架占位；调用即抛出 NotImplementedError，不执行实际业务。"""
    raise NotImplementedError("仅定义接口，尚未实现")


def merge_gradients(left: GradientStats, right: GradientStats) -> GradientStats:
    """按特征索引合并梯度、误差和样本数；空分区为中性元；禁止对分区均值直接取平均。

    参数：
        left (GradientStats)：第一个分区/子树的梯度汇总统计
        right (GradientStats)：第二个分区/子树的梯度汇总统计
    返回：
        GradientStats：梯度和、偏置梯度和、平方误差和及样本数。
    实现边界：
        只实现本接口声明的职责；训练/验证隔离、ID 对齐及状态恢复遵循架构文档。
        不允许静默丢弃测试记录、把未测数据填零或生成虚构评估结果。
    当前状态：
        骨架占位；调用即抛出 NotImplementedError，不执行实际业务。"""
    raise NotImplementedError("仅定义接口，尚未实现")


def update_parameters(
    weights: Sequence[float],
    bias: float,
    stats: GradientStats,
    learning_rate: float,
    l2: float,
) -> tuple[Sequence[float], float]:
    """按全局样本数归一化数据梯度后加一次 L2，更新权重与偏置；检查样本数、NaN、溢出及维数。

    参数：
        weights (Sequence[float])：当前全局模型权重，仅以只读形式传给 executor
        bias (float)：当前线性模型偏置
        stats (GradientStats)：全局梯度和、误差和与样本数；更新前检查样本计数
        learning_rate (float)：本次参数更新使用的正学习率
        l2 (float)：非负 L2 正则化系数，只在全局更新阶段计入一次
    返回：
        tuple[Sequence[float], float]：按类型标注返回结构化结果；字段含义及顺序见本函数职责和公共数据契约。
    实现边界：
        只实现本接口声明的职责；训练/验证隔离、ID 对齐及状态恢复遵循架构文档。
        不允许静默丢弃测试记录、把未测数据填零或生成虚构评估结果。
    当前状态：
        骨架占位；调用即抛出 NotImplementedError，不执行实际业务。"""
    raise NotImplementedError("仅定义接口，尚未实现")
