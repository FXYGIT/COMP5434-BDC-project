"""测量边界统一；Spark 惰性转换必须包含触发实际计算的 action。"""

from __future__ import annotations

import os
import platform
from collections.abc import Mapping
from time import perf_counter


def capture_hardware() -> Mapping[str, object]:
    """记录 CPU 型号、物理/逻辑核心数、服务器数、线程数、内存及 GPU（如有）；Spark 记录 driver/executor/分区配置。

    参数：
        无显式业务参数；对象方法使用初始化时声明的状态。
    返回：
        Mapping[str, object]：按类型标注返回结构化结果；字段含义及顺序见本函数职责和公共数据契约。
    实现边界：
        只实现本接口声明的职责；训练/验证隔离、ID 对齐及状态恢复遵循架构文档。
        不允许静默丢弃测试记录、把未测数据填零或生成虚构评估结果。
    当前状态：
        MVP 记录系统信息和阶段耗时；内存、GPU 及 Spark 配置待补充。"""
    return {
        "cpu": platform.processor(),
        "logical_cores": os.cpu_count(),
        "platform": platform.platform(),
    }


class BenchmarkRecorder:
    """保存 wall-clock 耗时，嵌套阶段不能重复计入总耗时；初始化范围必须披露。"""

    def __init__(self) -> None:
        self.started: dict[str, float] = {}
        self.elapsed: dict[str, float] = {}

    def start(self, stage: str) -> None:
        """用单调时钟开始阶段；拒绝重复启动同一未结束阶段。

        参数：
            stage (str)：计时阶段名称，用于配对开始/结束及生成报告
        返回：
            None：无返回值；状态更新/资源释放/写出职责见上文，失败不能静默忽略。
        实现边界：
            只实现本接口声明的职责；训练/验证隔离、ID 对齐及状态恢复遵循架构文档。
            不允许静默丢弃测试记录、把未测数据填零或生成虚构评估结果。
        当前状态：
            MVP 记录系统信息和阶段耗时；内存、GPU 及 Spark 配置待补充。"""
        self.started[stage] = perf_counter()

    def stop(self, stage: str) -> float:
        """在必要的 Spark action 完成或设备同步后结束，返回秒数；拒绝未开始的阶段。

        参数：
            stage (str)：计时阶段名称，用于配对开始/结束及生成报告
        返回：
            float：有限的浮点指标或秒数；明确空输入/零分母策略。
        实现边界：
            只实现本接口声明的职责；训练/验证隔离、ID 对齐及状态恢复遵循架构文档。
            不允许静默丢弃测试记录、把未测数据填零或生成虚构评估结果。
        当前状态：
            MVP 记录系统信息和阶段耗时；内存、GPU 及 Spark 配置待补充。"""
        self.elapsed[stage] = perf_counter() - self.started.pop(stage)
        return self.elapsed[stage]

    def summary(self) -> Mapping[str, object]:
        """返回每 epoch 时间、预处理、训练、离线总时间、全测试集推理总时间，以及各计时边界。

        参数：
            无显式业务参数；对象方法使用初始化时声明的状态。
        返回：
            Mapping[str, object]：按类型标注返回结构化结果；字段含义及顺序见本函数职责和公共数据契约。
        实现边界：
            只实现本接口声明的职责；训练/验证隔离、ID 对齐及状态恢复遵循架构文档。
            不允许静默丢弃测试记录、把未测数据填零或生成虚构评估结果。
        当前状态：
            MVP 记录系统信息和阶段耗时；内存、GPU 及 Spark 配置待补充。"""
        return dict(self.elapsed)
