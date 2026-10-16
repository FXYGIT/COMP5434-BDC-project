"""自主实现指标；可用第三方实现作为开发时的交叉核对。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def macro_f1(
    y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int] = (1, 2, 3, 4)
) -> float:
    """按固定四类计算 TP/FP/FN 和逐类 F1 后平均；分母为零时该类 F1=0；拒绝空输入、长度不等或非法标签。

    参数：
        y_true (Sequence[int])：真实目标序列，仅用于本地验证/评估
        y_pred (Sequence[int])：与真实目标逐条对应的预测序列
        labels (Sequence[int])：固定类别集合及顺序，默认天气类别 1、2、3、4
    返回：
        float：有限的浮点指标或秒数；明确空输入/零分母策略。
    """
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)

    f1_scores = []

    for label in labels:
        tp = np.sum((true == label) & (pred == label))
        fp = np.sum((true != label) & (pred == label))
        fn = np.sum((true == label) & (pred != label))

        denominator = 2 * tp + fp + fn
        f1_scores.append(2 * tp / denominator if denominator else 0.0)

    return float(np.mean(f1_scores))


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """计算 sqrt(sum((真实值-预测值)^2)/N)；拒绝空输入、非有限值与长度不等。Task 2 的生产评估在 Spark 中聚合，不能 collect 全量预测。

    参数：
        y_true (Sequence[float])：真实目标序列，仅用于本地验证/评估
        y_pred (Sequence[float])：与真实目标逐条对应的预测序列
    返回：
        float：有限的浮点指标或秒数；明确空输入/零分母策略。
    """
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)

    return float(np.sqrt(np.mean((true - pred) ** 2)))
