"""本地 RMSE/MAE 和误差切片；未来改为 Spark 聚合而非收集全量预测。"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def evaluate_ratings(predictions: pd.DataFrame) -> Mapping[str, float]:
    """从含 rating,prediction 的评分表计算 RMSE/MAE，返回浮点指标字典。

    上层已按 ID 接回真实评分；本函数不改变评分、不裁剪、不参与拟合。当前直接
    使用 sklearn，手写时分别累计平方误差、绝对误差和行数后统一归一化。
    """
    return {
        "rmse": float(root_mean_squared_error(predictions["rating"], predictions["prediction"])),
        "mae": float(mean_absolute_error(predictions["rating"], predictions["prediction"])),
    }


def evaluate_slices(predictions: pd.DataFrame, train: pd.DataFrame) -> Mapping[str, float]:
    """输出已见/未见用户和商品、评分档位、正文长度分组的样本数与 RMSE。

    predictions 需含 rating,prediction,user_id,prod_id,comment；train 仅用于判断
    实体是否已见，不能包含验证记录。空组只报告 count=0，不捏造 RMSE=0。
    后续 Spark 可通过训练实体表 join 和 groupBy 累积同样统计，避免 collect 全表。
    """
    groups = {}
    for column, name in (("user_id", "user"), ("prod_id", "product")):
        seen = predictions[column].isin(train[column])
        groups[f"seen_{name}"] = seen
        groups[f"unseen_{name}"] = ~seen
    for rating in range(1, 6):
        groups[f"rating_{rating}"] = predictions["rating"] == rating
    short = predictions["comment"].fillna("").str.len() < 100
    groups["short_text"] = short
    groups["long_text"] = ~short
    result: dict[str, float] = {}
    for name, mask in groups.items():
        subset = predictions.loc[mask]
        result[f"{name}_count"] = float(len(subset))
        if len(subset):
            result[f"{name}_rmse"] = evaluate_ratings(subset)["rmse"]
    return result
