"""保存、恢复模型状态，并输出实验结果。"""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np

from project.common.contracts import ModelBundle


def _json_default(value: object) -> object:
    """将路径、NumPy 数值和数组转为 JSON 可保存的类型。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"无法保存为 JSON：{type(value).__name__}")


def _write_json(path: Path, data: Mapping[str, object]) -> Path:
    """先写临时文件再替换目标文件；不覆盖已有结果。"""
    if path.exists():
        raise FileExistsError(path)

    content = json.dumps(dict(data), ensure_ascii=False, indent=2, default=_json_default)
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(content + "\n", encoding="utf-8")
    temporary_path.replace(path)
    return path


class ArtifactStore:
    """管理一次运行的产物：model/ 模型包和 metrics.json 实验结果。"""

    def __init__(self, run_dir: Path) -> None:
        """创建运行目录；允许使用 pipeline 已经创建好的目录。"""
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_bundle(self, state: Mapping[str, object]) -> ModelBundle:
        """保存模型/特征状态并返回模型包位置。

        state.pkl 保存完整状态，支持嵌套字典及 NumPy 数组；manifest.json
        只记录格式版本和状态文件名。Task 2 的特征及回归器目录由各自模块保存。
        模型目录已存在时直接报错，避免覆盖；清单在状态写完后生成。
        """
        model_dir = self.run_dir / "model"
        model_dir.mkdir()

        state_path = model_dir / "state.pkl"
        temporary_path = model_dir / "state.pkl.tmp"
        with temporary_path.open("wb") as file:
            pickle.dump(dict(state), file, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_path.replace(state_path)

        manifest_path = _write_json(
            model_dir / "manifest.json",
            {"schema_version": 1, "state_file": "state.pkl"},
        )
        return ModelBundle(model_dir, manifest_path)

    def load_bundle(self, bundle: ModelBundle) -> Mapping[str, object]:
        """读取已有模型包的完整状态，不重新训练或拟合特征。

        从 bundle 指定位置加载，不依赖当前 store 的运行目录。
        检查保存格式版本；模型维度和特征结构由对应模块恢复时校验。
        """
        manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
        if manifest["schema_version"] != 1:
            raise ValueError("不支持的模型保存格式版本")

        state_path = bundle.directory / manifest["state_file"]
        with state_path.open("rb") as file:
            return cast(Mapping[str, object], pickle.load(file))

    def write_results(self, metrics: Mapping[str, object]) -> Path:
        """保存指标、硬件、耗时等信息到 metrics.json，返回该文件路径。

        保留传入字段，不补造实验结果；支持路径、NumPy 数值及数组。
        """
        return _write_json(self.run_dir / "metrics.json", metrics)
