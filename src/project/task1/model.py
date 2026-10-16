"""手写 CART 决策树与随机森林；sklearn 只用于数据划分和评估指标。

课程不指定必须用哪个算法，但作业要求算法细节由自己实现，所以这里从零写
CART 的加权 Gini、逐列候选切点和停止条件，以及森林的 bootstrap 采样、
节点级子特征抽样、按树线程并行和概率聚合。numpy 只承担数组运算。
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from project.common.contracts import DenseMatrix

# 类别顺序固定为 1、2、3、4；概率矩阵的列顺序和 state_dict 都用它。
_CLASSES = [1, 2, 3, 4]

# max_depth 为 None 时的递归深度兜底。本任务的 min_samples_leaf 已经限制了
# 实际深度，这个上限只是防止极端数据下递归过深。
_UNLIMITED_DEPTH = 500


def _as_matrix(x: DenseMatrix) -> NDArray[np.float64]:
    """把输入统一成二维 float64 数组，模型内部只处理这一种形式。

    参数：
        x (DenseMatrix)：特征矩阵；列顺序须与训练时一致
    返回：
        NDArray[np.float64]：二维浮点数组，供索引和排序使用。
    """
    return np.asarray(x, dtype=np.float64)


def _class_indices(y: Sequence[int]) -> NDArray[np.intp]:
    """把 1..4 的标签转成 0..3 的列下标，与 _CLASSES 的顺序一一对应。

    参数：
        y (Sequence[int])：训练标签
    返回：
        NDArray[np.intp]：0..3 的类别下标；出现其它取值直接报错，不静默丢弃。
    """
    labels = np.asarray(y, dtype=np.float64)
    if not np.all(np.isin(labels, _CLASSES)):
        raise ValueError(f"labels must be drawn from {_CLASSES}")
    return (labels - 1).astype(np.intp)


def _positive_int(value: object, name: str) -> int:
    """校验正整数选项；写错就报错，不悄悄用默认值糊过去。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be greater than 0, got {value}")
    return value


def _sample_weights(labels: NDArray[np.intp], class_weight: object) -> NDArray[np.float64]:
    """把 class_weight 选项展开成逐样本权重。

    None 表示不加权；"balanced" 用 n / (K * count_c)。本数据四类完全均衡，
    这个式子恒等于 1.0，也就是等价于不加权——这是数据本身的性质，不是实现取巧。

    参数：
        labels (NDArray[np.intp])：0..3 的类别下标
        class_weight (object)：None、"balanced" 或其它（其它直接报错）
    返回：
        NDArray[np.float64]：与 labels 等长的逐样本权重。
    """
    if class_weight is None:
        return np.ones(len(labels), dtype=np.float64)
    if class_weight != "balanced":
        raise ValueError(f"unsupported class_weight: {class_weight!r}")
    counts = np.asarray(np.bincount(labels, minlength=len(_CLASSES)), dtype=np.float64)
    # 分母先避开 0（np.where 的两个分支都会被求值，直接除会触发除零警告）；
    # 缺席类别的权重取 0，而 labels 里不会出现它，所以查表永远取不到这一项。
    totals = np.where(counts > 0, counts, 1.0)
    per_class = np.where(counts > 0, len(labels) / (len(_CLASSES) * totals), 0.0)
    return per_class[labels]


def _gini(counts: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """加权 Gini 不纯度 1 - Σ p_c²；按行算，调用处保证每行权重为正。

    参数：
        counts (NDArray[np.float64])：形状 (n, K) 的各类别权重和
        weights (NDArray[np.float64])：形状 (n,) 的每行总权重
    返回：
        NDArray[np.float64]：形状 (n,) 的 Gini 值。
    """
    share = counts / weights[:, None]
    return 1.0 - np.sum(share * share, axis=1)


def _argmax_labels(probabilities: NDArray[np.float64]) -> list[int]:
    """按概率取类别标签。

    并列时 np.argmax 取下标最小的，也就是标签最小的类别优先；这条规则是确定的，
    同一份概率永远给同一个标签。
    """
    return [_CLASSES[int(index)] for index in np.argmax(probabilities, axis=1)]


class _Tree:
    """一棵训练好的 CART，用一组扁平数组存节点，下标就是节点编号。

    每个节点占一个下标：feature 是该节点分裂用的列（叶子为 -1），threshold 是
    分裂阈值（走左子树的条件是 x <= threshold），left/right 是左右孩子的下标。
    叶子的 left/right 指向自己，这样预测就是「循环固定轮数、每轮只更新当前节点
    下标」：既不用递归也不用判断叶子，多走几轮也不会移动。

    probabilities 每一行是该节点的类别分布。内部节点的分布预测时用不到，留着只是
    为了让数组和节点编号对齐。
    """

    def __init__(
        self,
        feature: NDArray[np.intp],
        threshold: NDArray[np.float64],
        left: NDArray[np.intp],
        right: NDArray[np.intp],
        probabilities: NDArray[np.float64],
        depth: int,
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.probabilities = probabilities
        self.depth = depth

    def predict_proba(self, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """把所有样本一起往下推，返回固定类别顺序 [1,2,3,4] 的概率。"""
        count = matrix.shape[0]
        rows = np.arange(count)
        nodes = np.zeros(count, dtype=np.intp)
        # 叶子深度不超过 depth，所以 depth + 1 轮一定够停；叶子自指，多走也不会动。
        for _ in range(self.depth + 1):
            at_feature = matrix[rows, self.feature[nodes]]
            go_left = at_feature <= self.threshold[nodes]
            nodes = np.where(go_left, self.left[nodes], self.right[nodes])
        return self.probabilities[nodes]

    def state_dict(self) -> Mapping[str, object]:
        """导出节点数组；存的是真实树结构，不依赖任何第三方估计器。"""
        return {
            "feature": self.feature,
            "threshold": self.threshold,
            "left": self.left,
            "right": self.right,
            "probabilities": self.probabilities,
            "depth": self.depth,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> _Tree:
        """从 state_dict 的节点数组还原一棵树，不重新训练。"""
        return cls(
            feature=cast(NDArray[np.intp], state["feature"]),
            threshold=cast(NDArray[np.float64], state["threshold"]),
            left=cast(NDArray[np.intp], state["left"]),
            right=cast(NDArray[np.intp], state["right"]),
            probabilities=cast(NDArray[np.float64], state["probabilities"]),
            depth=cast(int, state["depth"]),
        )


class _TreeBuilder:
    """递归建一棵 CART；节点按创建顺序追加到列表里，列表下标就是节点编号。"""

    def __init__(
        self,
        matrix: NDArray[np.float64],
        labels: NDArray[np.intp],
        weights: NDArray[np.float64],
        options: Mapping[str, object],
        rng: np.random.Generator,
    ) -> None:
        self.matrix = matrix
        self.labels = labels
        self.weights = weights
        max_depth = options.get("max_depth")
        self.max_depth = (
            _UNLIMITED_DEPTH if max_depth is None else _positive_int(max_depth, "max_depth")
        )
        self.min_samples_leaf = _positive_int(
            options.get("min_samples_leaf", 1), "min_samples_leaf"
        )
        self.min_samples_split = _positive_int(
            options.get("min_samples_split", 2), "min_samples_split"
        )
        max_features = options.get("max_features")
        self.max_features: int | None
        if max_features is None:
            self.max_features = None
        elif isinstance(max_features, str):
            if max_features == "sqrt":
                self.max_features = max(1, int(np.sqrt(matrix.shape[1])))
            elif max_features == "log2":
                self.max_features = max(1, int(np.log2(matrix.shape[1])))
            else:
                raise ValueError(f"unsupported max_features: {max_features!r}")
        elif isinstance(max_features, bool) or not isinstance(max_features, int):
            raise ValueError(
                f"max_features must be a positive int, 'sqrt', 'log2' or None, got {max_features!r}"
            )
        elif max_features < 1:
            raise ValueError(f"max_features must be greater than 0, got {max_features}")
        else:
            self.max_features = min(max_features, matrix.shape[1])
        # 整列恒定的特征在任何一个节点上都不可能成为分裂点，建树前先从候选池里剔除，
        # 否则它们会白占 max_features 的名额。本数据 17 列里有 7 列缺失掩码恒为 0。
        self.varying_features = np.flatnonzero(matrix.min(axis=0) != matrix.max(axis=0)).astype(
            np.intp
        )
        self.rng = rng
        self.feature: list[int] = []
        self.threshold: list[float] = []
        self.left: list[int] = []
        self.right: list[int] = []
        self.probabilities: list[NDArray[np.float64]] = []
        self.deepest = 0

    def build(self, rows: NDArray[np.intp], depth: int) -> int:
        """建以 rows 为样本的子树，返回该子树根节点的下标。

        先按叶子登记（叶子的分布就是这一节点的类别分布），找到分裂点再把它改写成
        内部节点并递归两个孩子。rows 里可以有重复行——bootstrap 采样就是这样表示的。
        """
        index = self._add_leaf(rows, depth)
        split = self._best_split(rows, depth)
        if split is None:
            return index
        feature, threshold, left_rows, right_rows = split
        self.feature[index] = feature
        self.threshold[index] = threshold
        self.left[index] = self.build(left_rows, depth + 1)
        self.right[index] = self.build(right_rows, depth + 1)
        return index

    def to_tree(self) -> _Tree:
        """把列表形式的节点转成扁平数组，得到可预测的树。"""
        return _Tree(
            feature=np.asarray(self.feature, dtype=np.intp),
            threshold=np.asarray(self.threshold, dtype=np.float64),
            left=np.asarray(self.left, dtype=np.intp),
            right=np.asarray(self.right, dtype=np.intp),
            probabilities=np.asarray(self.probabilities, dtype=np.float64),
            depth=self.deepest,
        )

    def _add_leaf(self, rows: NDArray[np.intp], depth: int) -> int:
        """登记一个叶子节点：类别分布取加权频率，并把 left/right 指向自己。"""
        index = len(self.feature)
        counts = np.asarray(
            np.bincount(self.labels[rows], weights=self.weights[rows], minlength=len(_CLASSES)),
            dtype=np.float64,
        )
        total = float(counts.sum())
        # 正常情况 total > 0；只有传进来的权重全是 0 才会落到均匀分布这个兜底上。
        distribution = counts / total if total > 0 else np.full(len(_CLASSES), 1.0 / len(_CLASSES))
        self.feature.append(-1)
        self.threshold.append(0.0)
        self.left.append(index)
        self.right.append(index)
        self.probabilities.append(distribution)
        self.deepest = max(self.deepest, depth)
        return index

    def _best_split(
        self, rows: NDArray[np.intp], depth: int
    ) -> tuple[int, float, NDArray[np.intp], NDArray[np.intp]] | None:
        """在本节点的候选列里找最优切点；找不到就返回 None，调用处当叶子处理。

        返回 (列号, 阈值, 左子行下标, 右子行下标)。
        """
        if depth >= self.max_depth:
            return None
        # 两个条件都能直接判死：样本太少不够分，或者分完一侧不够 min_samples_leaf。
        if len(rows) < self.min_samples_split or len(rows) < 2 * self.min_samples_leaf:
            return None
        if len(np.unique(self.labels[rows])) < 2:
            return None

        # 候选按 (加权不纯度, 列号, 阈值, 排序后的行顺序, 切点位置) 记下来，取不纯度最小的。
        best: tuple[float, int, float, NDArray[np.intp], int] | None = None
        for feature in self._sample_features():
            found = self._best_split_for_column(int(feature), rows)
            if found is None:
                continue
            impurity, threshold, order, position = found
            if best is None or impurity < best[0]:
                best = (impurity, int(feature), threshold, order, position)
        if best is None:
            return None

        _, feature, threshold, order, position = best
        return feature, threshold, rows[order[: position + 1]], rows[order[position + 1 :]]

    def _sample_features(self) -> NDArray[np.intp]:
        """按 max_features 从候选池里随机抽一批列，不放回。

        候选池是建树前算好的「整列取值有变化」的列，所以这里不做任何扫描；本节点内
        恰好也是常数的列由 _best_split_for_column 兜底跳过。
        """
        pool = self.varying_features
        if pool.size == 0 or self.max_features is None or self.max_features >= pool.size:
            return pool
        chosen = self.rng.choice(pool, size=self.max_features, replace=False)
        return np.sort(chosen).astype(np.intp)

    def _best_split_for_column(
        self, feature: int, rows: NDArray[np.intp]
    ) -> tuple[float, float, NDArray[np.intp], int] | None:
        """在单列上找最优切点。

        做法：把这列在本节点的取值排序，用累积和得到「切在第 i 个位置之后」时左右两侧
        的各类别权重和，再算两侧加权 Gini。阈值取相邻两个不同取值的中点，这样
        x <= threshold 的判断不会落在任何一个训练值上，不存在归属歧义。
        返回 (加权不纯度, 阈值, 排序后的行顺序, 切点位置)；本列分不开则返回 None。
        """
        values = self.matrix[rows, feature]
        if values.min() == values.max():
            # 本节点上这列是常数，切不动。缺失掩码在无缺失数据上就是这种情况，直接跳过。
            return None

        order = np.argsort(values, kind="stable")
        ordered_values = values[order]
        ordered_labels = self.labels[rows][order]
        ordered_weights = self.weights[rows][order]

        # 累积加权类别计数：cumulative[i] 是前 i+1 个样本的各类别权重和。
        one_hot = np.zeros((len(rows), len(_CLASSES)), dtype=np.float64)
        one_hot[np.arange(len(rows)), ordered_labels] = ordered_weights
        cumulative = np.cumsum(one_hot, axis=0)
        total = cumulative[-1]
        left_counts = cumulative[:-1]
        right_counts = total - left_counts
        left_weight = np.sum(left_counts, axis=1)
        right_weight = np.sum(right_counts, axis=1)

        # 合法切点：两侧都够 min_samples_leaf，且相邻两个取值确实不同（否则切了也白切）。
        positions = np.arange(len(rows) - 1)
        valid = (positions + 1 >= self.min_samples_leaf) & (
            len(rows) - positions - 1 >= self.min_samples_leaf
        )
        valid &= ordered_values[:-1] < ordered_values[1:]
        valid_positions = np.flatnonzero(valid)
        if valid_positions.size == 0:
            return None

        # 最小化两侧加权不纯度之和，等价于最大化 Gini 下降（父节点那一项对所有候选相同）。
        impurity = left_weight[valid_positions] * _gini(
            left_counts[valid_positions], left_weight[valid_positions]
        ) + right_weight[valid_positions] * _gini(
            right_counts[valid_positions], right_weight[valid_positions]
        )
        # argmin 并列时取最小下标，也就是取最小阈值，结果确定。
        position = int(valid_positions[int(np.argmin(impurity))])
        threshold = float((ordered_values[position] + ordered_values[position + 1]) / 2)
        return float(np.min(impurity)), threshold, order, position


@dataclass(frozen=True)
class _TreeTask:
    """交给线程池的一棵树的训练任务。

    每棵树的随机源只由 seed + index 决定，和线程调度无关，所以森林的结果
    不随线程数变化。
    """

    index: int
    matrix: NDArray[np.float64]
    labels: NDArray[np.intp]
    weights: NDArray[np.float64]
    options: Mapping[str, object]
    seed: int
    bootstrap: bool


def _train_tree(task: _TreeTask) -> _Tree:
    """训练一棵树：先抽 bootstrap 样本，再交给 CART 建树。"""
    rng = np.random.default_rng(task.seed + task.index)
    count = task.matrix.shape[0]
    rows: NDArray[np.intp]
    if task.bootstrap:
        # 有放回抽 count 行；重复行直接用重复下标表示，不复制数据。
        rows = np.asarray(rng.integers(0, count, count, dtype=np.intp), dtype=np.intp)
    else:
        rows = np.arange(count, dtype=np.intp)
    builder = _TreeBuilder(task.matrix, task.labels, task.weights, task.options, rng)
    builder.build(rows, 0)
    return builder.to_tree()


class DecisionTreeClassifier:
    """手写 CART 单树：加权 Gini、逐列候选切点、显式停止条件和叶节点分布。

    没有实现缺失值分支：调用方保证送进来的特征矩阵已经填好缺失
    （数值列填训练折中位数，类别列归到未知编码）。
    """

    def __init__(self, options: Mapping[str, object], seed: int) -> None:
        """记录超参数和随机种子；此时还没有树，fit 或 load_state_dict 之后才有。"""
        self.options = dict(options)
        self.seed = seed
        self.tree: _Tree | None = None

    def fit(
        self,
        x: DenseMatrix,
        y: Sequence[int],
        sample_weights: Sequence[float] | None = None,
    ) -> None:
        """训练一棵四分类树。

        类别权重和外部传入的样本权重相乘后一起参与 Gini 计算。节点级子特征抽样
        用本树的随机源，所以同一个 seed 每次建出的树完全一样。

        参数：
            x (DenseMatrix)：特征矩阵；列顺序和维度必须与已保存特征 schema 一致
            y (Sequence[int])：与特征行一一对应的训练标签，取值 1..4
            sample_weights (Sequence[float] | None)：可选逐样本非负权重，与 y 等长
        返回：
            None：只更新对象状态。
        """
        matrix = _as_matrix(x)
        labels = _class_indices(y)
        weights = _sample_weights(labels, self.options.get("class_weight"))
        if sample_weights is not None:
            weights = weights * np.asarray(sample_weights, dtype=np.float64)
        builder = _TreeBuilder(
            matrix, labels, weights, self.options, np.random.default_rng(self.seed)
        )
        builder.build(np.arange(matrix.shape[0], dtype=np.intp), 0)
        self.tree = builder.to_tree()

    def predict_proba(self, x: DenseMatrix) -> DenseMatrix:
        """逐样本沿树下行，返回固定类别顺序 [1,2,3,4] 的概率。

        参数：
            x (DenseMatrix)：特征矩阵；列顺序和维度必须与训练时一致
        返回：
            DenseMatrix：形状 (n, 4) 的概率矩阵。
        """
        if self.tree is None:
            raise ValueError("model is not fitted or restored yet, cannot predict")
        return cast(DenseMatrix, self.tree.predict_proba(_as_matrix(x)))

    def predict(self, x: DenseMatrix) -> Sequence[int]:
        """取概率最大的类别，返回 1..4 的标签。"""
        probabilities = _as_matrix(self.predict_proba(x))
        return _argmax_labels(probabilities)

    def state_dict(self) -> Mapping[str, object]:
        """导出节点数组、超参数和种子，可直接 pickle 保存。"""
        if self.tree is None:
            raise ValueError("model is not fitted, nothing to save")
        return {"tree": self.tree.state_dict(), "options": self.options, "seed": self.seed}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """从保存状态恢复树，不重新训练；新实例上可直接调用。"""
        self.tree = _Tree.from_state(cast(Mapping[str, object], state["tree"]))
        self.options = dict(cast(Mapping[str, object], state["options"]))
        self.seed = cast(int, state["seed"])


class RandomForestClassifier:
    """手写随机森林：bootstrap 采样、节点级子特征抽样、按树并行和概率聚合。

    森林的结果只由 seed 决定，与线程数、与各棵树的完成顺序无关：第 i 棵树的
    随机源固定是 seed + i。
    """

    def __init__(self, options: Mapping[str, object], seed: int) -> None:
        """记录超参数和随机种子；此时还没有树，fit 或 load_state_dict 之后才有。"""
        self.options = dict(options)
        self.seed = seed
        self.trees: list[_Tree] = []

    def fit(self, x: DenseMatrix, y: Sequence[int]) -> None:
        """训练整片森林。

        n_jobs 为 None 或 1 时顺序训练，-1 时用满所有核，其它正整数用该线程数。
        线程之间共享同一份训练矩阵，不复制数据。

        参数：
            x (DenseMatrix)：特征矩阵；列顺序和维度必须与已保存特征 schema 一致
            y (Sequence[int])：与特征行一一对应的训练标签，取值 1..4
        返回：
            None：只更新对象状态。
        """
        matrix = _as_matrix(x)
        labels = _class_indices(y)
        weights = _sample_weights(labels, self.options.get("class_weight"))
        bootstrap = bool(self.options.get("bootstrap", True))
        trees = _positive_int(self.options.get("n_estimators", 100), "n_estimators")
        tasks = [
            _TreeTask(index, matrix, labels, weights, self.options, self.seed, bootstrap)
            for index in range(trees)
        ]
        n_jobs = self.options.get("n_jobs", -1)
        if n_jobs is None:
            workers = 1
        elif isinstance(n_jobs, bool) or not isinstance(n_jobs, int):
            raise ValueError(f"n_jobs must be an integer, got {n_jobs!r}")
        elif n_jobs == -1:
            workers = os.cpu_count() or 1
        elif n_jobs < 1:
            raise ValueError(f"n_jobs must be a positive integer or -1, got {n_jobs}")
        else:
            workers = n_jobs
        if workers == 1:
            self.trees = [_train_tree(task) for task in tasks]
        else:
            # map 按输入顺序返回结果，所以树的顺序与线程调度无关。
            with ThreadPoolExecutor(max_workers=workers) as pool:
                self.trees = list(pool.map(_train_tree, tasks))

    def predict_proba(self, x: DenseMatrix) -> DenseMatrix:
        """平均各棵树的四分类概率，类别列顺序固定 [1,2,3,4]。

        参数：
            x (DenseMatrix)：特征矩阵；列顺序和维度必须与训练时一致
        返回：
            DenseMatrix：形状 (n, 4) 的概率矩阵。
        """
        if not self.trees:
            raise ValueError("model is not fitted or restored yet, cannot predict")
        matrix = _as_matrix(x)
        total = np.zeros((matrix.shape[0], len(_CLASSES)), dtype=np.float64)
        for tree in self.trees:
            total += tree.predict_proba(matrix)
        return cast(DenseMatrix, total / len(self.trees))

    def predict(self, x: DenseMatrix) -> Sequence[int]:
        """取平均概率最大的类别，返回 1..4 的标签。"""
        probabilities = _as_matrix(self.predict_proba(x))
        return _argmax_labels(probabilities)

    def state_dict(self) -> Mapping[str, object]:
        """导出全部树的节点数组、超参数和种子，可直接 pickle 保存。"""
        if not self.trees:
            raise ValueError("model is not fitted, nothing to save")
        return {
            "trees": [tree.state_dict() for tree in self.trees],
            "options": self.options,
            "seed": self.seed,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """恢复整片森林，不重新训练；新实例上可直接调用。"""
        saved = cast(Sequence[Mapping[str, object]], state["trees"])
        self.trees = [_Tree.from_state(item) for item in saved]
        self.options = dict(cast(Mapping[str, object], state["options"]))
        self.seed = cast(int, state["seed"])


class MajorityClassifier:
    """最朴素的基线：永远预测训练集里样本最多的类别。

    它和手写树、手写森林共用同一套 fit/predict 接口，所以基线和主模型
    可以用同一份划分、同一套指标直接对比。
    """

    def __init__(self) -> None:
        self.label = _CLASSES[0]

    def fit(self, x: DenseMatrix, y: Sequence[int]) -> None:
        """数一遍训练集各个类别的样本数，记下最多的那个。

        参数：
            x (DenseMatrix)：特征矩阵；本基线不使用，保留是为了与其它模型同接口
            y (Sequence[int])：训练标签，取值 1..4
        返回：
            None：只更新对象状态。
        """
        counts = np.bincount(_class_indices(y), minlength=len(_CLASSES))
        self.label = _CLASSES[int(np.argmax(counts))]

    def predict(self, x: DenseMatrix) -> Sequence[int]:
        """无论输入什么都返回训练集里的多数类。"""
        return [self.label] * len(x)
