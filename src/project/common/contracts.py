"""跨模块的数据契约；核心逻辑不得以裸字典传递未知结构。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

type DenseMatrix = Sequence[Sequence[float]]
type SparseVector = Mapping[int, float]


@dataclass(frozen=True)
class TaskConfig:
    """配置：任务、数据路径、输出根目录、随机种子及算法专属参数。

    params 必须由配置验证器检查键名、类型及范围，禁止静默忽略拼写错误。
    """

    task: str
    train_path: Path
    test_path: Path
    sample_path: Path
    output_dir: Path
    seed: int
    params: Mapping[str, object]


@dataclass(frozen=True)
class SplitIds:
    """持久化训练/验证 ID；必须互斥且不遗漏输入记录；用于重放实验。"""

    train_ids: Sequence[str]
    valid_ids: Sequence[str]


@dataclass(frozen=True)
class PredictionBatch:
    """预测与 ID 一一对应；分类值为 1..4，评分值为有限实数。

    Task 2 默认保留连续评分（允许回归），是否裁剪至 [1,5] 由验证实验决定。
    """

    ids: Sequence[str]
    values: Sequence[float]


@dataclass(frozen=True)
class EpochRecord:
    """一个 epoch 的实测信息；不可用填充的零值替代缺失测量。"""

    epoch: int
    seconds: float
    train_loss: float
    validation_metric: float | None


@dataclass(frozen=True)
class EvaluationResult:
    """实验结果：主指标、诊断指标及训练日志。"""

    primary_name: str
    primary_value: float
    diagnostics: Mapping[str, float]
    epochs: Sequence[EpochRecord]


@dataclass(frozen=True)
class ModelBundle:
    """模型包路径：参数、预处理状态、schema、特征顺序及版本必须一起保存。"""

    directory: Path
    manifest_path: Path
