"""小数据提交契约；Task 2 实现独立的流式/分布式导出。"""

from __future__ import annotations

from pathlib import Path

from project.common.contracts import PredictionBatch


def validate_predictions(predictions: PredictionBatch, task: str) -> None:
    """检查 ID 唯一、预测数量、有限数值；Task 1 必须是整数 1..4；不把 Task 2 回归输出强制转成整数。

    参数：
        predictions (PredictionBatch)：带 ID 的预测；评估接口还要求真实 rating，参见原始职责说明
        task (str)：任务标识 task1 或 task2，用于选择标签与提交契约
    返回：
        None：无返回值；状态更新/资源释放/写出职责见上文，失败不能静默忽略。
    实现边界：
        只实现本接口声明的职责；训练/验证隔离、ID 对齐及状态恢复遵循架构文档。
        不允许静默丢弃测试记录、把未测数据填零或生成虚构评估结果。
    当前状态：
        骨架占位；调用即抛出 NotImplementedError，不执行实际业务。"""
    raise NotImplementedError("仅定义接口，尚未实现")


def export_submission(
    predictions: PredictionBatch, sample_path: Path, output_path: Path, task: str
) -> Path:
    """以 sample_submission.csv 的 ID 顺序输出精确列名；按 ID 对齐而非假设行序一致；拒绝缺失、多余、重复 ID；不输出索引列。

    参数：
        predictions (PredictionBatch)：带 ID 的预测；评估接口还要求真实 rating，参见原始职责说明
        sample_path (Path)：sample_submission.csv；输出列名、ID 覆盖与顺序的依据
        output_path (Path)：最终输出文件路径；校验完成后原子落盘
        task (str)：任务标识 task1 或 task2，用于选择标签与提交契约
    返回：
        Path：输出文件或目录的路径，成功返回前应完成相关校验。
    实现边界：
        只实现本接口声明的职责；训练/验证隔离、ID 对齐及状态恢复遵循架构文档。
        不允许静默丢弃测试记录、把未测数据填零或生成虚构评估结果。
    当前状态：
        骨架占位；调用即抛出 NotImplementedError，不执行实际业务。"""
    raise NotImplementedError("仅定义接口，尚未实现")
