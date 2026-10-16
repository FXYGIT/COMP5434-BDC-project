"""Task 2 的大数据提交导出：不把全量测试预测收集到 driver 内存。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


def export_rating_submission(
    spark: SparkSession, predictions: DataFrame, sample_path: Path, output_path: Path
) -> Path:
    """按样例 ID 检查一对一覆盖并 join，显式构造样例顺序；通过有序迭代流式写单一 id,rating CSV（或外部合并），不使用 collect/toPandas/coalesce(1) 作为默认大规模路径；原子落盘后校验行数。

    参数：
        spark (SparkSession)：已创建且未关闭的 SparkSession
        predictions (DataFrame)：带 ID 的预测；评估接口还要求真实 rating，参见原始职责说明
        sample_path (Path)：sample_submission.csv；输出列名、ID 覆盖与顺序的依据
        output_path (Path)：最终输出文件路径；校验完成后原子落盘
    返回：
        Path：输出文件或目录的路径，成功返回前应完成相关校验。
    实现边界：
        只实现本接口声明的职责；训练/验证隔离、ID 对齐及状态恢复遵循架构文档。
        不允许静默丢弃测试记录、把未测数据填零或生成虚构评估结果。
    当前状态：
        骨架占位；调用即抛出 NotImplementedError，不执行实际业务。"""
    raise NotImplementedError("仅定义接口，尚未实现")
