"""天气数据读取、质量检查和固定的分层划分。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from project.common.contracts import SplitIds

# 标签的合法取值，共四类天气；分层划分和最终提交都以它为准。
LABELS = (1, 2, 3, 4)


def _profile(series: pd.Series) -> dict[str, object]:
    """统计单个属性：数据类型、缺失个数、不同取值个数。

    整列都能当数字看时按数值属性处理，额外给出最小值和最大值；
    否则按类别属性处理，列出全部取值，方便人工核对编码范围。
    """
    profile: dict[str, object] = {
        "dtype": str(series.dtype),
        "missing": int(series.isna().sum()),
        "unique": int(series.nunique()),
    }
    numbers = pd.to_numeric(series, errors="coerce")
    if numbers.notna().all():
        profile["min"] = float(numbers.min())
        profile["max"] = float(numbers.max())
    else:
        profile["values"] = sorted(str(value) for value in series.dropna().unique())
    return profile


class WeatherDataset:
    """保留原始 ID 和列名；特征中不得出现 ID 和 label。"""

    def __init__(self, train_path: Path, test_path: Path) -> None:
        self.train_path = train_path
        self.test_path = test_path

    def load(
        self,
    ) -> tuple[Sequence[Mapping[str, object]], Sequence[Mapping[str, object]]]:
        """读取带标签训练表和无标签测试表；保留缺失值，不静默删除任何一行。

        参数：
            无显式业务参数；对象方法使用初始化时声明的状态。
        返回：
            tuple[Sequence[Mapping[str, object]], Sequence[Mapping[str, object]]]：训练记录在前、测试记录在后。
        """
        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)

        return train.to_dict("records"), test.to_dict("records")

    def audit(self, rows: Sequence[Mapping[str, object]], labeled: bool) -> Mapping[str, object]:
        """体检一张表并返回报告，只读不改，报告可直接写进 metrics.json。

        检查项：行数、id 是否缺失或重复、每个属性的数据类型/缺失/取值范围、
        完全重复的行；labeled 为 True 时再检查 label 是否存在、是否都落在
        LABELS 以及各类别数量。发现的问题汇总在 problems 里，用英文短标签，
        空列表表示没发现问题。

        参数：
            rows (Sequence[Mapping[str, object]])：输入记录；保持 ID 对齐
            labeled (bool)：训练表传 True，测试表传 False
        返回：
            Mapping[str, object]：体检报告，键为 rows/columns/duplicate_rows/missing_cells/
            attributes，训练表额外含 label_distribution，另有 problems 问题清单。
        """
        frame = pd.DataFrame(rows)
        problems: list[str] = []

        duplicate_rows = int(frame.duplicated().sum())
        if duplicate_rows:
            problems.append(f"duplicate rows: {duplicate_rows}")

        if "id" not in frame.columns:
            problems.append("column 'id' is missing")
        else:
            missing_ids = int(frame["id"].isna().sum())
            if missing_ids:
                problems.append(f"column 'id' has {missing_ids} missing values")
            if not frame["id"].is_unique:
                duplicate_ids = int(frame["id"].duplicated().sum())
                problems.append(f"column 'id' has {duplicate_ids} duplicate values")

        attributes = [name for name in frame.columns if name not in ("id", "label")]

        report: dict[str, object] = {
            "rows": int(len(frame)),
            "columns": [str(name) for name in frame.columns],
            "duplicate_rows": duplicate_rows,
            "missing_cells": int(frame.isna().sum().sum()),
            "attributes": {str(name): _profile(frame[name]) for name in attributes},
        }

        if labeled:
            if "label" not in frame.columns:
                problems.append("column 'label' is missing")
            else:
                labels = frame["label"]
                missing_labels = int(labels.isna().sum())
                if missing_labels:
                    problems.append(f"column 'label' has {missing_labels} missing values")
                seen = labels.dropna().unique().tolist()
                unexpected = sorted(set(seen) - set(LABELS), key=str)
                if unexpected:
                    problems.append(f"column 'label' has unexpected values: {unexpected}")
                distribution = {
                    str(value): int(count) for value, count in labels.value_counts().items()
                }
                report["label_distribution"] = dict(sorted(distribution.items()))

        report["problems"] = problems
        return report

    def split(
        self, rows: Sequence[Mapping[str, object]], valid_fraction: float, seed: int
    ) -> SplitIds:
        """按标签分层切分训练/验证集，返回互斥且可重放的 ID。

        本数据不存在重复特征记录，按行分层即可；若以后出现重复记录，
        需要改成按组划分，避免同一条记录同时落进训练集和验证集。

        参数：
            rows (Sequence[Mapping[str, object]])：训练表记录；必须含 id 和 label
            valid_fraction (float)：验证集比例，取值需在 0 到 1 之间
            seed (int)：固定随机种子，保证同一份数据每次切分结果一致
        返回：
            SplitIds：训练 ID 和验证 ID，两者互斥。
        """
        if not 0 < valid_fraction < 1:
            raise ValueError(f"valid_fraction must be between 0 and 1, got {valid_fraction}")

        frame = pd.DataFrame(rows)
        train_ids, valid_ids = train_test_split(
            frame["id"],
            test_size=valid_fraction,
            random_state=seed,
            stratify=frame["label"],
        )

        return SplitIds(
            train_ids=list(train_ids),
            valid_ids=list(valid_ids),
        )
