"""读取 JSON 配置，将配置文件中的路径转换为绝对路径。"""

from __future__ import annotations

import json
from pathlib import Path

from project.common.contracts import TaskConfig


def load_config(path: Path) -> TaskConfig:
    path = path.resolve()
    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    base_dir = path.parent
    config = TaskConfig(
        task=data["task"],
        train_path=(base_dir / data["train_path"]).resolve(),
        test_path=(base_dir / data["test_path"]).resolve(),
        sample_path=(base_dir / data["sample_path"]).resolve(),
        output_dir=(base_dir / data["output_dir"]).resolve(),
        model_dir=(base_dir / data["model_dir"]).resolve(),
        seed=data["seed"],
        params=data["params"],
    )

    if config.task not in ("task1", "task2"):
        raise ValueError("task 必须为 task1 或 task2")

    return config
