"""仅收集聚合统计的 RMSE 及冷启动诊断。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


def evaluate_ratings(predictions: DataFrame) -> Mapping[str, float]:
    """输入 id,rating,prediction；用分布式 sum squared error 和 count 计算 RMSE/MAE；检查 join 后数量和非有限预测；不 collect 行。

    参数：
        predictions (DataFrame)：带 ID 的预测；评估接口还要求真实 rating，参见原始职责说明
    返回：
        Mapping[str, float]：按类型标注返回结构化结果；字段含义及顺序见本函数职责和公共数据契约。
    实现边界：
        只实现本接口声明的职责；训练/验证隔离、ID 对齐及状态恢复遵循架构文档。
        不允许静默丢弃测试记录、把未测数据填零或生成虚构评估结果。
    当前状态：
        骨架占位；调用即抛出 NotImplementedError，不执行实际业务。"""
    raise NotImplementedError("仅定义接口，尚未实现")


def evaluate_slices(predictions: DataFrame, train: DataFrame) -> Mapping[str, float]:
    """对已见/未见用户、商品、评分区间和文本长度分组评估，报告每组样本数；用户/商品元信息通过 ID join 获取。

    参数：
        predictions (DataFrame)：带 ID 的预测；评估接口还要求真实 rating，参见原始职责说明
        train (DataFrame)：当前训练折或锁定配置后的全量训练表，不能包含验证记录
    返回：
        Mapping[str, float]：按类型标注返回结构化结果；字段含义及顺序见本函数职责和公共数据契约。
    实现边界：
        只实现本接口声明的职责；训练/验证隔离、ID 对齐及状态恢复遵循架构文档。
        不允许静默丢弃测试记录、把未测数据填零或生成虚构评估结果。
    当前状态：
        骨架占位；调用即抛出 NotImplementedError，不执行实际业务。"""
    raise NotImplementedError("仅定义接口，尚未实现")
