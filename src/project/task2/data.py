"""本地评论 CSV 读取与可重放划分；后续可替换为 Spark schema/聚合实现。

CSV 中的逗号、引号和多行文本交由 pandas 解析。MVP 信任输入，不增加额外
校验；只在显式 max_rows 配置存在时截取训练记录，测试集始终完整读取。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd


def read_reviews(path: Path, labeled: bool, max_rows: int | None = None) -> pd.DataFrame:
    """读取评论表，保留字符串 ID，并把缺失标题/正文视为空文本。

    参数：path 为 CSV 路径；labeled 区分训练和测试；max_rows 仅限制训练行数，
    None 表示读取全部训练数据。返回 pandas DataFrame，列名与原 CSV 一致。
    后续 Spark 实现需使用显式 schema、multiline 和 escaped quote，并记录坏行。
    """
    frame = pd.read_csv(
        path,
        dtype={"id": str, "user_id": str, "prod_id": str, "parent_prod_id": str},
        nrows=max_rows if labeled else None,
    )
    frame[["title", "comment"]] = frame[["title", "comment"]].fillna("")
    return frame


def audit_reviews(frame: pd.DataFrame, labeled: bool) -> Mapping[str, object]:
    """返回当前实际读取行数、缺失统计及可选评分分布，不拒绝或删除记录。

    参数：frame 为原始评论表，labeled=True 时统计 rating。返回可写入 JSON 的
    小型字典；行数反映本次数据范围，不把采样结果称为全量。未来改为分布式聚合。
    """
    result: dict[str, object] = {
        "rows": len(frame),
        "columns": list(frame.columns),
        "missing": frame.isna().sum().to_dict(),
    }
    if labeled:
        result["rating_distribution"] = frame["rating"].value_counts().to_dict()
    return result


def split_reviews(
    frame: pd.DataFrame, strategy: str, valid_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回互斥的训练/验证表；固定种子与输入可重放，各方案共用同一划分。

    stable_hash_group 对完全相同的 title/comment 文本组做稳定哈希，不使用 rating，
    同一评论组不会跨集合；stable_hash 对 ID 哈希；time 按毫秒时间字段排序切分。
    valid_fraction 是哈希划分的期望比例（实际比例可略有差异）。不修改输入表。
    后续 Spark 实现可用等价稳定哈希或时间排序，保存 ID 及分组规则供重放。
    """
    if strategy == "time":
        ordered = frame.sort_values("time", kind="stable")
        boundary = int(len(ordered) * (1 - valid_fraction))
        return ordered.iloc[:boundary].copy(), ordered.iloc[boundary:].copy()
    columns = ["title", "comment"] if strategy == "stable_hash_group" else ["id"]
    keys = frame[columns].fillna("").astype(str).copy()
    keys["_split_seed"] = str(seed)
    hashed = pd.util.hash_pandas_object(keys, index=False)
    valid = hashed % 1_000_000 < valid_fraction * 1_000_000
    return frame.loc[~valid].copy(), frame.loc[valid].copy()
