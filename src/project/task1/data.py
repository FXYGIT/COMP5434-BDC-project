"""天气数据读取、质量检查和固定的分层划分。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from project.common.contracts import SplitIds


class WeatherDataset:
    """保留原始 ID 和列名，特征禁止包含 ID 和 label。"""

    def __init__(self, train_path: Path, test_path: Path) -> None:
        """记录文件路径，不在构造函数中执行昂贵加载。

        参数：
            train_path (Path)：含真实目标的训练 CSV 路径
            test_path (Path)：不含目标的测试 CSV 路径
        返回：
            None：无返回值；状态更新/资源释放/写出职责见上文，失败不能静默忽略。
        实现边界：
            只实现本接口声明的职责；训练/验证隔离、ID 对齐及状态恢复遵循架构文档。
            不允许静默丢弃测试记录、把未测数据填零或生成虚构评估结果。
        当前状态：
            骨架占位；调用即抛出 NotImplementedError，不执行实际业务。"""
        raise NotImplementedError("仅定义接口，尚未实现")

    def load(
        self,
    ) -> tuple[Sequence[Mapping[str, object]], Sequence[Mapping[str, object]]]:
        """读取带标签训练表和无标签测试表；保留缺失值，不静默删除测试行。

        参数：
            无显式业务参数；对象方法使用初始化时声明的状态。
        返回：
            tuple[Sequence[Mapping[str, object]], Sequence[Mapping[str, object]]]：按类型标注返回结构化结果；字段含义及顺序见本函数职责和公共数据契约。
        实现边界：
            只实现本接口声明的职责；训练/验证隔离、ID 对齐及状态恢复遵循架构文档。
            不允许静默丢弃测试记录、把未测数据填零或生成虚构评估结果。
        当前状态：
            骨架占位；调用即抛出 NotImplementedError，不执行实际业务。"""
        raise NotImplementedError("仅定义接口，尚未实现")

    def audit(self, rows: Sequence[Mapping[str, object]], labeled: bool) -> Mapping[str, object]:
        """检查 ID、10 个属性、标签域、类型、重复行、缺失、异常值、类别分布；禁止预设未确认的属性名称。

        参数：
            rows (Sequence[Mapping[str, object]])：输入记录；保持 ID 对齐，按当前函数契约决定是否含标签
            labeled (bool)：是否要求目标字段；测试表必须按无标签模式校验
        返回：
            Mapping[str, object]：按类型标注返回结构化结果；字段含义及顺序见本函数职责和公共数据契约。
        实现边界：
            只实现本接口声明的职责；训练/验证隔离、ID 对齐及状态恢复遵循架构文档。
            不允许静默丢弃测试记录、把未测数据填零或生成虚构评估结果。
        当前状态：
            骨架占位；调用即抛出 NotImplementedError，不执行实际业务。"""
        raise NotImplementedError("仅定义接口，尚未实现")

    def split(
        self, rows: Sequence[Mapping[str, object]], valid_fraction: float, seed: int
    ) -> SplitIds:
        """按标签分层划分并持久化 ID；重复特征记录需成组划分，降低相同记录跨集合泄漏。

        参数：
            rows (Sequence[Mapping[str, object]])：输入记录；保持 ID 对齐，按当前函数契约决定是否含标签
            valid_fraction (float)：验证集比例，范围应为 0 到 1 的开区间
            seed (int)：固定随机种子；与数据划分和模型随机状态共同记录
        返回：
            SplitIds：互斥且可重放的训练/验证 ID 集合。
        实现边界：
            只实现本接口声明的职责；训练/验证隔离、ID 对齐及状态恢复遵循架构文档。
            不允许静默丢弃测试记录、把未测数据填零或生成虚构评估结果。
        当前状态：
            骨架占位；调用即抛出 NotImplementedError，不执行实际业务。"""
        raise NotImplementedError("仅定义接口，尚未实现")
