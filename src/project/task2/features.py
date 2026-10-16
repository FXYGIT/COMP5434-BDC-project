"""本地文本及元数据稀疏特征；保留未来稳定哈希和 Spark 转换的实现边界。"""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize


@dataclass
class ReviewFeatures:
    """特征批次：ids 与 CSR 矩阵逐行对应，ratings 仅在有真实标签时存在。

    矩阵按标题、正文、元数据顺序拼接；不包含 ID 或真实评分。独立结构避免将
    每行稀疏向量塞入 DataFrame，后续分布式版本可替换为 id,rating,features 表。
    """

    ids: Sequence[str]
    matrix: Any  # scipy CSR；稀疏矩阵类型随 SciPy 版本变化。
    ratings: NDArray[np.float64] | None


class ReviewFeaturePipeline:
    """训练折拟合 IDF 和用户/商品频率，验证/测试只转换，支持持久化恢复。

    MVP 使用 sklearn 哈希及 TF-IDF、SciPy CSR、joblib 本地线程；标题/正文
    分别占用固定维数。目标均值编码暂未接入，元数据完全不使用 rating。
    后续手写：规范化、token/ngram、稳定哈希碰撞聚合、IDF 及缩放；Spark 只负责
    分区转换和统计聚合。目标编码如实现，训练行必须使用不含自身/重复组的 OOF 值。
    """

    def __init__(self, options: Mapping[str, object]) -> None:
        """记录特征开关和布局，创建无词表的哈希器；不读取数据或学习标签。

        options 支持 use_title/comment/metadata/idf、text_dimension_per_field、
        ngram_max、n_jobs。创建后先 fit 或 load，再 transform。
        """
        self.options = dict(options)
        self.fields = [field for field in ("title", "comment") if options.get(f"use_{field}", True)]
        self.vectorizers = {
            field: HashingVectorizer(
                n_features=int(cast(int, options.get("text_dimension_per_field", 65536))),
                alternate_sign=False,
                norm=None,
                ngram_range=(1, int(cast(int, options.get("ngram_max", 1)))),
            )
            for field in self.fields
        }
        self.idfs: dict[str, Any] = {}
        self.frequencies: dict[str, dict[str, float]] = {}

    def fit(self, train: pd.DataFrame) -> None:
        """只用训练表拟合各文本域的 IDF 和用户/商品出现次数，返回 None。

        输入需含 title/comment 及元数据列；rating 完全不参与。IDF 使用训练行的
        文档频次，未知用户/商品频率回退 0。重复调用会重新建立全部拟合状态。
        """
        self.idfs = {}
        for field in self.fields:
            if self.options.get("use_idf", True):
                counts = self.vectorizers[field].transform(train[field])
                self.idfs[field] = TfidfTransformer().fit(counts)
        self.frequencies = {
            column: train[column].value_counts().to_dict()
            for column in ("user_id", "prod_id", "parent_prod_id")
        }

    def _transform_chunk(self, frame: pd.DataFrame) -> Any:
        """转换一个本地块为 CSR；读取拟合状态，不更新共享状态或使用评分。

        文本做 TF-IDF 或 L2 归一化词频；元数据包含购买标记、log1p 票数、文本长度
        和用户/商品频率，并按行归一化。此纯转换可在多个本地线程并行执行。
        """
        parts = []
        for field in self.fields:
            counts = self.vectorizers[field].transform(frame[field])
            parts.append(
                self.idfs[field].transform(counts) if field in self.idfs else normalize(counts)
            )
        if self.options.get("use_metadata", True):
            metadata = pd.DataFrame(index=frame.index)
            metadata["purchased"] = frame["purchased"].astype(str).str.lower().isin(["true", "1"])
            metadata["votes"] = np.log1p(pd.to_numeric(frame["votes"]).fillna(0).clip(lower=0))
            # 文本域被关闭时同时去除其长度，确保消融真正移除该域的信息。
            for field in self.fields:
                metadata[f"{field}_length"] = np.log1p(frame[field].str.len())
            for column, counts in self.frequencies.items():
                metadata[column] = np.log1p(frame[column].map(counts).fillna(0))
            parts.append(normalize(sparse.csr_matrix(metadata.to_numpy(dtype=float))))
        return sparse.hstack(parts, format="csr") if parts else sparse.csr_matrix((len(frame), 1))

    def transform_training(self, train: pd.DataFrame) -> ReviewFeatures:
        """转换训练折并保留 ratings，返回与输入顺序一致的 ReviewFeatures。

        当前无目标编码，因此复用 transform；未来启用目标编码时应在本入口构造
        按重复组排除自身的 OOF 特征，不能直接套用训练全体目标均值。
        """
        return self.transform(train)

    def transform(self, frame: pd.DataFrame) -> ReviewFeatures:
        """使用已拟合状态转换任意表，保持全部 ID、顺序和固定列布局。

        n_jobs 控制实际 joblib 线程数：每线程处理连续行块，再按原顺序纵向拼接。
        返回 ids、CSR 矩阵及可选 ratings；测试集无评分时 ratings=None。
        """
        workers = int(cast(int, self.options.get("n_jobs", 1)))
        if workers == 1 or len(frame) == 0:
            matrix = self._transform_chunk(frame)
        else:
            chunks = [
                frame.iloc[index]
                for index in np.array_split(np.arange(len(frame)), workers)
                if len(index)
            ]
            matrices = Parallel(n_jobs=workers, prefer="threads")(
                delayed(self._transform_chunk)(chunk) for chunk in chunks
            )
            matrix = sparse.vstack(matrices, format="csr")
        ratings = frame["rating"].to_numpy(dtype=float) if "rating" in frame else None
        return ReviewFeatures(frame["id"].astype(str).tolist(), matrix, ratings)

    def save(self, directory: Path) -> None:
        """把选项、哈希器、IDF、特征域顺序和频率表写入 features.pkl。

        directory 为模型包中的特征子目录，会自动创建；返回 None。当前 pickle
        保存 sklearn 状态，后续手写版本改为显式版本、词法规则、IDF 和统计表。
        """
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "features.pkl").open("wb") as file:
            pickle.dump(self.__dict__, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, directory: Path) -> None:
        """从 features.pkl 恢复全部训练状态，不重新 fit；返回 None。

        directory 必须指向 save 生成的特征目录。训练布局及选项覆盖构造时配置，
        保证新进程推理与训练一致；仅加载本项目生成且可信的模型包。
        """
        with (directory / "features.pkl").open("rb") as file:
            self.__dict__.update(pickle.load(file))


def tokenize(text: str, options: Mapping[str, object]) -> list[str]:
    """按 sklearn 默认词法规则小写化、去标点并生成 1..ngram_max 的词组。

    输入为空字符串时返回空列表；使用与 HashingVectorizer 相同的 analyzer。
    后续手写版本应保持两字符以上词元及 ngram 顺序一致，再迁移到 Spark executor。
    """
    analyzer = CountVectorizer(
        ngram_range=(1, int(cast(int, options.get("ngram_max", 1))))
    ).build_analyzer()
    return cast(list[str], analyzer(text))


def hash_features(tokens: list[str], dimension: int, salt: str) -> Mapping[int, float]:
    """将词元用 FeatureHasher 稳定映射为稀疏词频字典，碰撞时相加。

    dimension 为哈希空间大小；salt 为固定域前缀；返回索引到计数的映射。
    禁用符号哈希和进程随机化的内置 hash。此独立函数供后续手写哈希交叉核对；
    主流程通过分离矩阵空间区分标题/正文，不需要额外 salt。
    """
    matrix = FeatureHasher(
        n_features=dimension, input_type="string", alternate_sign=False
    ).transform([[f"{salt}:{token}" for token in tokens]])
    return dict(zip(matrix.indices.tolist(), matrix.data.tolist(), strict=False))
