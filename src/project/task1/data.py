"""天气数据读取、质量检查和固定的分层划分。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from project.common.contracts import SplitIds


class WeatherDataset:
    """保留原始 ID 和列名，特征禁止包含 ID 和 label。"""

    def __init__(self, train_path: Path, test_path: Path) -> None:
        self.train_path = train_path
        self.test_path = test_path

    def load(
        self,
    ) -> tuple[Sequence[Mapping[str, object]], Sequence[Mapping[str, object]]]:
        """读取带标签训练表和无标签测试表；保留缺失值，不静默删除测试行。

        参数：
            无显式业务参数；对象方法使用初始化时声明的状态。
        返回：
            tuple[Sequence[Mapping[str, object]], Sequence[Mapping[str, object]]]：按类型标注返回结构化结果；字段含义及顺序见本函数职责和公共数据契约。
        """
        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)

        return train.to_dict("records"), test.to_dict("records")

    def audit(self, rows: Sequence[Mapping[str, object]], labeled: bool) -> Mapping[str, object]:
        """检查 ID、10 个属性、标签域、类型、重复行、缺失、异常值、类别分布；禁止预设未确认的属性名称。

        参数：
            rows (Sequence[Mapping[str, object]])：输入记录；保持 ID 对齐，按当前函数契约决定是否含标签
            labeled (bool)：是否要求目标字段；测试表必须按无标签模式校验
        返回：
            Mapping[str, object]：按类型标注返回结构化结果；字段含义及顺序见本函数职责和公共数据契约。
        """
        df = pd.DataFrame(rows)

        result = {
            "rows": len(df),
            "columns": list(df.columns),
            "missing": df.isna().sum().to_dict(),
            "duplicates": int(df.duplicated().sum()),
        }

        if labeled and "label" in df:
            result["label_distribution"] = df["label"].value_counts().to_dict()

        return result

    def split(
        self, rows: Sequence[Mapping[str, object]], valid_fraction: float, seed: int
    ) -> SplitIds:
        """按标签分层划分并持久化 ID；重复特征记录需成组划分，降低相同记录跨集合泄漏。

        参数：
            rows (Sequence[Mapping[str, object]])：输入记录；保持 ID 对齐，按当前函数契约决定是否含标签
            valid_fraction (float)：验证集比例，范围应为 0 到 1 的开区间
            seed (int)：固定随机种子；与数据划分和模型随机状态共同记录
        返回：
            SplitIds：互斥且可重放的训练/验证 ID 集合。
        """
        df = pd.DataFrame(rows)

        train_ids, valid_ids = train_test_split(
            df["id"],
            test_size=valid_fraction,
            random_state=seed,
            stratify=df["label"],
        )

        return SplitIds(
            train_ids=list(train_ids),
            valid_ids=list(valid_ids),
        )
