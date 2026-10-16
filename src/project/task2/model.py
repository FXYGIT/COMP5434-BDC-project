"""本地稀疏评分回归 MVP；保留后续 Spark 同步梯度实现的类接口。"""

from __future__ import annotations

import copy
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error

from project.common.contracts import EpochRecord
from project.task2.features import ReviewFeatures


class DistributedRatingRegressor:
    """保留原类名；当前实际执行 sklearn 本地 SGD，不创建 Spark 会话。

    训练目标为评分减去训练均值后的残差，使用平方损失和 L2；全局均值基线使用
    DummyRegressor。记录真实每轮耗时/MSE/验证 RMSE，并在早停后恢复最佳模型。
    后续手写实现：广播只读权重，mapPartitions 计算稀疏梯度，treeAggregate
    按样本数合并后更新；销毁旧广播，不收集整张评论表。纯梯度接口见 optimization。
    """

    def __init__(self, options: Mapping[str, object], seed: int) -> None:
        """保存优化参数和随机种子，初始化空模型及训练日志；不执行拟合。

        options 支持 model=global_mean、epochs、batches_per_epoch、learning_rate、
        l2、early_stopping_patience 和 clip_predictions；seed 控制每轮行顺序。
        """
        self.options = dict(options)
        self.seed = seed
        self.model: Any = None
        self.mean = 0.0
        self.selected_epoch = 0
        self.history: list[EpochRecord] = []

    def fit(
        self, train: ReviewFeatures, valid: ReviewFeatures | None = None
    ) -> Sequence[EpochRecord]:
        """拟合训练 CSR 特征，返回实际执行的逐轮 EpochRecord 列表。

        train 必须带 ratings；valid 仅用于评估和早停，不拟合特征。每轮固定种子
        洗牌并分 batches_per_epoch 批调用 partial_fit；train_loss 为未裁剪预测的 MSE。
        patience=0 禁用早停并保留最后一轮（用于固定工作量 benchmark）；无验证集时
        总是执行配置的全部 epochs。基线只计算一次训练均值，记录一条真实耗时日志。
        """
        y = cast(NDArray[np.float64], train.ratings)
        self.mean = float(np.mean(y))
        self.history = []
        if self.options.get("model") == "global_mean":
            started = perf_counter()
            self.model = DummyRegressor(strategy="mean").fit(train.matrix, y)
            self.mean = 0.0  # DummyRegressor 已直接输出评分，不再叠加均值。
            loss = float(mean_squared_error(y, self._predict_values(train)))
            score = self._validation_rmse(valid) if valid is not None else None
            self.history.append(EpochRecord(1, perf_counter() - started, loss, score))
            self.selected_epoch = 1
            return self.history
        self.model = SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=float(cast(float, self.options.get("l2", 0.0001))),
            learning_rate="constant",
            eta0=float(cast(float, self.options.get("learning_rate", 0.01))),
            random_state=self.seed,
            shuffle=False,
            tol=None,
        )
        rng = np.random.default_rng(self.seed)
        epochs = int(cast(int, self.options.get("epochs", 20)))
        batches = int(cast(int, self.options.get("batches_per_epoch", 4)))
        patience = int(cast(int, self.options.get("early_stopping_patience", 3)))
        best_score, stale = float("inf"), 0
        best_model = None
        for epoch in range(1, epochs + 1):
            started = perf_counter()
            for indices in np.array_split(rng.permutation(len(y)), batches):
                if len(indices):
                    self.model.partial_fit(train.matrix[indices], y[indices] - self.mean)
            loss = float(mean_squared_error(y, self.model.predict(train.matrix) + self.mean))
            score = self._validation_rmse(valid) if valid is not None else None
            self.history.append(EpochRecord(epoch, perf_counter() - started, loss, score))
            self.selected_epoch = epoch
            if score is not None and patience > 0:
                if score < best_score:
                    best_score, stale = score, 0
                    best_model = (copy.deepcopy(self.model), epoch)
                else:
                    stale += 1
                if stale >= patience:
                    break
        if best_model is not None:
            self.model, self.selected_epoch = best_model
        return self.history

    def _predict_values(self, frame: ReviewFeatures) -> NDArray[np.float64]:
        """对 CSR 特征返回一维连续评分数组，按配置决定是否裁剪到 [1,5]。

        输入顺序保持不变；不四舍五入、不访问 ratings。训练均值只来自当前训练折，
        与拟合时的残差目标配套恢复原评分尺度。
        """
        values = np.asarray(self.model.predict(frame.matrix), dtype=float) + self.mean
        return np.clip(values, 1, 5) if self.options.get("clip_predictions", True) else values

    def _validation_rmse(self, valid: ReviewFeatures) -> float:
        """用验证评分和当前预测计算 RMSE，裁剪规则与最终导出完全一致。

        返回真实标量指标；不会更新模型参数或特征状态，供每轮早停比较使用。
        """
        return float(root_mean_squared_error(valid.ratings, self._predict_values(valid)))

    def predict(self, frame: ReviewFeatures) -> pd.DataFrame:
        """返回 id,prediction 两列，覆盖全部输入记录；评分保持浮点数。

        frame 为由匹配特征状态生成的 ReviewFeatures。调用前需要 fit 或 load；
        验证标签和元数据由上层按 ID 合并，避免模型接口混入真实评分。
        """
        return pd.DataFrame({"id": frame.ids, "prediction": self._predict_values(frame)})

    def save(self, directory: Path) -> None:
        """在模型子目录写入 model.pkl，保存估计器、均值、参数、种子和训练历史。

        返回 None；后续手写版本应显式保存权重、偏置、维数、已选 epoch 及算法版本。
        """
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "model.pkl").open("wb") as file:
            pickle.dump(self.__dict__, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, directory: Path) -> None:
        """恢复 save 生成的可信模型状态，覆盖构造参数，不读取训练数据或重新训练。

        返回 None；加载后使用与模型包配套的特征状态进行预测。
        """
        with (directory / "model.pkl").open("rb") as file:
            self.__dict__.update(pickle.load(file))
