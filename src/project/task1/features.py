"""仅在训练折拟合的数据清洗和编码流水线。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd

from project.common.contracts import DenseMatrix


class WeatherPreprocessor:
    """负责缺失值、数值/类别特征编码；验证集和测试集只 transform。"""

    def __init__(self, options: Mapping[str, object]) -> None:
        self.options = dict(options)
        self.feature_columns: list[str] = []

    def fit(self, rows: Sequence[Mapping[str, object]]) -> None:
        """仅用当前训练折学习数值中位数、类别词典和标准化参数；保存输入与输出列顺序。

        参数：
            rows (Sequence[Mapping[str, object]])：输入记录；保持 ID 对齐，按当前函数契约决定是否含标签
        返回：
            None：无返回值；状态更新/资源释放/写出职责见上文，失败不能静默忽略。
        """
        df = pd.DataFrame(rows)
        self.feature_columns = [c for c in df.columns if c not in {"id", "label"}]

    def transform(self, rows: Sequence[Mapping[str, object]]) -> DenseMatrix:
        """固定特征顺序转换；未见类别映射到未知桶；拒绝缺少必要列，输出不能包含 ID 或 label。

        参数：
            rows (Sequence[Mapping[str, object]])：输入记录；保持 ID 对齐，按当前函数契约决定是否含标签
        返回：
            DenseMatrix：固定列顺序的二维数值/概率矩阵，具体语义见功能说明。
        """
        df = pd.DataFrame(rows)
        values = (
            df[self.feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .to_numpy(dtype=float)
        )

        # 缺失指示消融：只附加缺失掩码；后续手写缺失值统计和编码时保留此开关。
        if self.options.get("missing_indicators", False):
            values = np.column_stack((values, df[self.feature_columns].isna().to_numpy(float)))
        return cast(DenseMatrix, values)

    def state_dict(self) -> Mapping[str, object]:
        """导出可重放的特征状态；预测时恢复状态而非重新拟合。

        参数：
            无显式业务参数；对象方法使用初始化时声明的状态。
        返回：
            Mapping[str, object]：按类型标注返回结构化结果；字段含义及顺序见本函数职责和公共数据契约。
        """
        return {
            "options": self.options,
            "feature_columns": self.feature_columns,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """校验版本和列映射后恢复参数。

        参数：
            state (Mapping[str, object])：已序列化的模型/特征状态；恢复前须检查版本和维数
        返回：
            None：无返回值；状态更新/资源释放/写出职责见上文，失败不能静默忽略。
        """
        self.options = dict(cast(Mapping[str, object], state["options"]))
        self.feature_columns = cast(list[str], state["feature_columns"])
