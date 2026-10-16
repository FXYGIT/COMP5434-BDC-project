"""验证用于模型选择，最终模型在全量训练数据重训。"""

from __future__ import annotations

from project.common.contracts import EvaluationResult, ModelBundle, TaskConfig


def evaluate_candidate(config: TaskConfig) -> EvaluationResult:
    """读取已固定的 split；仅在训练折 fit 预处理和模型；在验证折计算 Macro-F1、逐类 F1 和混淆矩阵；不访问测试标签。

    参数：
        config (TaskConfig)：已解析并通过字段校验的任务配置
    返回：
        EvaluationResult：主指标、辅助诊断及训练时间日志。
    实现边界：
        只实现本接口声明的职责；训练/验证隔离、ID 对齐及状态恢复遵循架构文档。
        不允许静默丢弃测试记录、把未测数据填零或生成虚构评估结果。
    当前状态：
        骨架占位；调用即抛出 NotImplementedError，不执行实际业务。"""
    raise NotImplementedError("仅定义接口，尚未实现")


def fit_final(config: TaskConfig) -> ModelBundle:
    """锁定验证选出的超参数；在完整 train.csv 上重新 fit 特征和模型，保存可独立预测的模型包；不根据私有榜调参。

    参数：
        config (TaskConfig)：已解析并通过字段校验的任务配置
    返回：
        ModelBundle：持久化的模型包位置与清单，不是内存模型对象。
    实现边界：
        只实现本接口声明的职责；训练/验证隔离、ID 对齐及状态恢复遵循架构文档。
        不允许静默丢弃测试记录、把未测数据填零或生成虚构评估结果。
    当前状态：
        骨架占位；调用即抛出 NotImplementedError，不执行实际业务。"""
    raise NotImplementedError("仅定义接口，尚未实现")
