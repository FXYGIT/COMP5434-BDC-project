"""本地 Task 2 CSV 导出；未来大规模预测可替换为 Spark 有序迭代写出。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_rating_submission(
    predictions: pd.DataFrame, sample_path: Path, output_path: Path
) -> Path:
    """按样例 ID 顺序输出 id,rating，保持连续预测评分且不写 DataFrame 索引。

    参数：predictions 含 id,prediction；sample_path 为 sampleSubmission.csv；
    output_path 为目标 CSV。输入可信，直接按字符串 ID 对齐，不添加额外校验。
    自动创建父目录，返回输出路径；后续 Spark 实现需保留顺序并流式导出全部记录。
    """
    sample = pd.read_csv(sample_path, dtype={"id": str})
    values = predictions.assign(id=predictions["id"].astype(str)).set_index("id")["prediction"]
    sample["rating"] = sample["id"].map(values)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)
    return output_path
