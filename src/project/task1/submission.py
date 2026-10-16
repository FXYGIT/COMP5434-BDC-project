"""Task 1 提交出口：复用 common/submission 的校验及文件写出契约。"""

from __future__ import annotations

from pathlib import Path

from project.common.contracts import PredictionBatch


def export_weather_submission(
    predictions: PredictionBatch, sample_path: Path, output_path: Path
) -> Path:
    """生成天气分类预测文件。

    功能：作为 Task 1 领域入口，未来调用公共校验和导出方法，固定 task=task1。
    参数：predictions 为 ID 与预测标签；sample_path 为提交样例；output_path 为目标文件。
    返回：成功写出的 CSV 路径。
    约束：标签必须为整数 1..4；ID 与样例一一对应，按样例顺序输出 id,label。
    错误：重复、缺失、多余 ID、非法标签或无法写入时明确失败，不输出部分有效文件。
    """
    raise NotImplementedError("仅定义接口，尚未实现")
