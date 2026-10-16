"""随机种子与实验运行清单。"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def set_random_seed(seed: int) -> None:
    """设置 Python random 和 NumPy 全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)


def write_run_manifest(output_dir: Path, metadata: Mapping[str, object]) -> Path:
    """将运行信息保存为 manifest.json，返回文件路径。

    metadata 由调用者提供，可包含任务、配置、种子和输出路径；Path 转为字符串。
    添加 UTC 创建时间，不修改原 metadata。先写临时文件再替换目标文件。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(metadata)
    manifest["created_at"] = datetime.now(UTC).isoformat()
    content = json.dumps(manifest, ensure_ascii=False, indent=2, default=str)

    path = output_dir / "manifest.json"
    temporary_path = output_dir / "manifest.json.tmp"
    temporary_path.write_text(content + "\n", encoding="utf-8")
    temporary_path.replace(path)
    return path
