"""Task 1 CSV 导出。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from project.common.contracts import PredictionBatch


def export_weather_submission(
    predictions: PredictionBatch, sample_path: Path, output_path: Path
) -> Path:
    """按样例 ID 顺序写出 id,label。"""
    sample = pd.read_csv(sample_path, dtype={"id": str})
    labels = dict(zip(map(str, predictions.ids), predictions.values, strict=False))
    sample["label"] = sample["id"].map(labels).astype(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)
    return output_path
