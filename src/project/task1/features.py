"""仅在训练折拟合的数据清洗和编码流水线。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd

from project.common.contracts import DenseMatrix


def _is_numeric(values: pd.Series) -> bool:
    """整列（除开本来就缺失的位置）都能当数字看，就算数值属性。"""
    converted = pd.to_numeric(values, errors="coerce")
    return bool((converted.notna() | values.isna()).all())


def _category_codes(values: pd.Series, categories: Sequence[str]) -> np.ndarray:
    """把类别列转成整数编码。

    训练时见过的类别按 categories 的顺序取 0、1、2……；
    没见过的类别（包括缺失）统一取 len(categories)，与所有已知类别都不同。
    """
    position = {name: index for index, name in enumerate(categories)}
    unknown = len(categories)
    return np.array([position.get(str(value), unknown) for value in values], dtype=float)


class WeatherPreprocessor:
    """负责数值/类别特征编码；数值列缺失按训练折中位数填充；验证集和测试集只 transform。"""

    def __init__(self, options: Mapping[str, object]) -> None:
        self.options = dict(options)
        self.feature_columns: list[str] = []
        self.category_columns: dict[str, list[str]] = {}
        self.numeric_fill: dict[str, float] = {}

    def fit(self, rows: Sequence[Mapping[str, object]]) -> None:
        """只用当前训练折决定两件事：有哪些特征列，每个类别列有哪些取值。

        类别清单排序后保存，保证同一份训练折每次得到完全相同的编码。
        数值列记录训练折上的中位数，缺失值统一用它填充。

        参数：
            rows (Sequence[Mapping[str, object]])：训练折记录；保持 ID 对齐
        返回：
            None：只更新对象状态，不返回结果。
        """
        frame = pd.DataFrame(rows)
        self.feature_columns = [name for name in frame.columns if name not in {"id", "label"}]
        self.category_columns = {}
        self.numeric_fill = {}
        for name in self.feature_columns:
            if not _is_numeric(frame[name]):
                seen = frame[name].dropna().unique()
                self.category_columns[name] = sorted(str(value) for value in seen)
            else:
                numbers = pd.to_numeric(frame[name], errors="coerce")
                # 整列都缺失时中位数也是 NaN，填了等于没填；退回 0 保证填进去的确实是个数。
                if numbers.notna().any():
                    self.numeric_fill[name] = float(numbers.median())
                else:
                    self.numeric_fill[name] = 0.0

    def transform(self, rows: Sequence[Mapping[str, object]]) -> DenseMatrix:
        """按 fit 记录的顺序逐列转换，每列输出一个数值列；结果不含 ID 和 label。

        类别列用 fit 时的类别清单编码，未见类别（含缺失）归到未知编码；
        数值列直接转成浮点数，缺失用 fit 时算好的中位数填充。
        开启 missing_indicators 时在末尾为数值列附加缺失掩码；类别列不附加，
        因为它们的缺失已经落在未知编码里了。
        列名与训练时不一致会直接报错，不静默跳过。

        参数：
            rows (Sequence[Mapping[str, object]])：任意一批记录；列名需与训练时一致
        返回：
            DenseMatrix：固定列顺序的二维浮点矩阵。
        """
        frame = pd.DataFrame(rows)
        columns: list[np.ndarray] = []
        for name in self.feature_columns:
            if name in self.category_columns:
                columns.append(_category_codes(frame[name], self.category_columns[name]))
            else:
                numbers = pd.to_numeric(frame[name], errors="coerce")
                columns.append(numbers.fillna(self.numeric_fill[name]).to_numpy(dtype=float))
        values = np.column_stack(columns)

        # 缺失指示消融：只给数值列附加缺失掩码。类别列的缺失已经由未知编码桶表示，
        # 再挂一列等于重复统计同一件事，还白白多占每次分裂的列名额。
        if self.options.get("missing_indicators", False):
            numeric_columns = [
                name for name in self.feature_columns if name not in self.category_columns
            ]
            values = np.column_stack((values, frame[numeric_columns].isna().to_numpy(float)))
        return cast(DenseMatrix, values)

    def state_dict(self) -> Mapping[str, object]:
        """导出可重放的特征状态；预测时恢复状态而不是重新拟合。

        参数：
            无显式业务参数；对象方法使用初始化时声明的状态。
        返回：
            Mapping[str, object]：列顺序、类别清单和中位数填充值，可直接 pickle 保存。
        """
        return {
            "options": self.options,
            "feature_columns": self.feature_columns,
            "category_columns": self.category_columns,
            "numeric_fill": self.numeric_fill,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """恢复训练时保存的列顺序、类别清单和中位数填充值，不重新拟合。

        参数：
            state (Mapping[str, object])：state_dict 保存的内容
        返回：
            None：只更新对象状态。
        """
        self.options = dict(cast(Mapping[str, object], state["options"]))
        self.feature_columns = cast(list[str], state["feature_columns"])
        self.category_columns = cast(dict[str, list[str]], state["category_columns"])
        self.numeric_fill = cast(dict[str, float], state["numeric_fill"])
