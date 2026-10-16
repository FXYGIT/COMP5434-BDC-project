# Task 2：评论评分本地 MVP

本阶段优先跑通所有命令，使用 pandas 读取 CSV、sklearn 哈希文本/TF-IDF 与
SGDRegressor 评分回归、SciPy 稀疏矩阵和 joblib 本地特征转换线程。
当前全部执行路径为 `local`，不需要 Java 或 SparkSession；原 Spark 会话接口和
分区梯度说明保留在代码中，供后续替换使用。

## 1. 数据与默认范围

所有命令在项目根目录运行。数据目录为：

```text
data/task-2/
├── train.csv
├── test.csv
└── sampleSubmission.csv
```

配置为 `configs/task2.json`，路径相对于配置文件所在目录解析。
默认 `max_train_rows=50000`，表示读取 CSV 的前 50,000 条训练记录，不是随机抽样。
这是为了快速验证 MVP；报告中的指标仅代表这部分数据上的验证结果。
将该参数设为 `null` 可读取全量训练数据，但本地实现会把文本和稀疏特征放在内存中，
运行时间和内存占用会增大。测试读取不受此上限影响，始终预测全部记录。

## 2. 推荐执行顺序

```text
experiment → baseline / ablate / benchmark → 手动确定配置 → train-final → predict → report-tables
```

没有模型文件也能执行实验：前四个命令均会从零训练临时模型，不保存正式模型。
只有 `predict` 依赖已保存的模型包。只需要生成提交文件时，运行 `train-final` 和 `predict`。

## 3. 各命令的输入与产出

### experiment：训练临时模型并验证

```powershell
uv run project experiment --config configs/task2.json
```

输入为配置和训练 CSV。默认按标题/正文完全相同的评论组做稳定哈希划分，期望约
80% 训练、20% 验证；同一评论组不会同时出现在两侧。仅用训练折拟合 IDF、实体频率
和回归器，验证折只转换、预测与评估，不读取测试数据。

产出 `runs/task2/<时间>_experiment/`：

- `manifest.json`：配置、种子、路径、local 后端及训练读取上限。
- `split.json`：本次训练/验证 ID，供重放及方案间共用。
- `metrics.json`：RMSE、MAE、已见/未见实体等切片指标、实际样本数、逐轮日志和耗时。

RMSE 越低越好。早停会恢复最佳验证轮次的模型；`selected_epoch` 是选中轮次，
`epochs` 列表保留实际执行过的全部轮次。空切片只报告 count=0，不生成虚假的 RMSE。

### baseline：比较三种基线

```powershell
uv run project baseline --config configs/task2.json
```

输入为同一配置和训练数据，在同一份划分上比较训练全局均值、纯文本回归、文本加
元数据回归。产出 `runs/task2/<时间>_baseline/` 下的清单、划分和各方案指标。
均值基线只执行一次拟合，其余方案按配置执行 SGD 和验证早停。

### ablate：逐项关闭特征或后处理

```powershell
uv run project ablate --config configs/task2.json
```

输入为同一配置和训练数据。比较完整模型、关闭标题、关闭正文、关闭元数据、关闭
IDF、关闭评分裁剪六种方案。各方案复用同一份划分，每次只覆盖一项配置。
关闭标题/正文时也去掉相应文本长度元数据，确保对应信息真正被移除。

产出 `runs/task2/<时间>_ablate/` 下的清单、划分和各方案结果；不会自动修改配置。

### benchmark：比较本地特征转换线程数

```powershell
uv run project benchmark --config configs/task2.json --parallelisms 1 2 4 8
```

输入为配置、训练数据和线程数列表。使用 joblib 实际执行分块特征转换，计时范围
为已拟合特征状态下的训练/验证转换与矩阵拼接，不含 IDF 拟合；模型训练也记录耗时。
各方案使用相同特征、种子和批次，禁用早停并固定执行配置中的全部 epochs。

产出 `runs/task2/<时间>_benchmark/`。`feature_transform_seconds` 是比较的耗时，
`speedup` 为列表第一项耗时除以当前耗时，因此建议把 1 放在首位。
它不是 Spark 分布式效率结果；小数据、线程调度及 Python GIL 可能导致线程越多越慢。

### train-final：训练并保存正式模型

```powershell
uv run project train-final --config configs/task2.json
```

输入为配置和训练数据。使用配置范围内全部训练记录，不再切验证集或使用验证早停。
默认使用前 50,000 条；只有 `max_train_rows=null` 才是整个训练 CSV。
不会自动选取历史实验中的最佳配置或 epoch，需要手动修改配置后训练。

产出：

```text
models/task2/rating_regressor/
├── manifest.json
├── state.pkl                 # 配置、种子、实际训练行数和子目录位置
├── features/features.pkl     # 哈希器、IDF、特征布局、实体频率
└── regressor/model.pkl       # 估计器、评分均值、参数、选中轮次和训练日志

runs/task2/<时间>_train-final/
├── manifest.json
└── metrics.json              # 训练范围、数据概况、逐轮日志、训练和保存耗时
```

已有模型目录不会覆盖。再次训练时，把配置里的 `model_dir` 改为新目录，例如
`../models/task2/rating_regressor_v2`，预测时同步修改 `--model`。

### predict：恢复模型并生成提交 CSV

```powershell
uv run project predict --config configs/task2.json --model models/task2/rating_regressor
```

输入为模型包、完整测试 CSV 和提交样例。不需要训练 CSV，也不会重新拟合。
预测使用模型包内保存的特征和模型参数，当前配置仅提供输入输出路径。

产出 `runs/task2/<时间>_predict/` 下的 `manifest.json`、`metrics.json` 和
`submission.csv`。提交文件按样例 ID 顺序输出 `id,rating`，保留连续评分。
默认裁剪到 `[1,5]`，不四舍五入；裁剪选项和验证阶段保持一致。不会自动上传 Kaggle。

### report-tables：汇总已有运行

下面为 PowerShell 示例，替换为实际运行目录：

```powershell
uv run project report-tables `
  --runs `
    runs/task2/<时间>_experiment `
    runs/task2/<时间>_baseline `
    runs/task2/<时间>_ablate `
    runs/task2/<时间>_benchmark `
    runs/task2/<时间>_train-final `
    runs/task2/<时间>_predict `
  --output-dir outputs/task2-tables
```

输入是指定目录中的 `metrics.json`；产出 `outputs/task2-tables/results.csv`。
每行记录来源目录及已有指标，未测量字段留空。该命令不重新训练，可仅汇总部分运行。

## 4. 主要配置

| 参数 | 当前含义 |
|---|---|
| max_train_rows | 训练 CSV 前 N 行；null 为全量，测试读取不受影响 |
| split_strategy | stable_hash_group 按完全相同的标题/正文分组；stable_hash 按 ID；time 按时间 |
| valid_fraction | 验证比例，哈希分组后的实际比例可能略有差异 |
| text_dimension_per_field | 标题和正文各自的哈希维数，默认 65536 |
| use_title / use_comment / use_metadata / use_idf | 对应特征开关 |
| ngram_max | 可选词组长度上限，默认 1 |
| n_jobs | 本地特征转换线程数，默认 1 |
| epochs / batches_per_epoch | 最大训练轮数及每轮 SGD 批数 |
| learning_rate / l2 | SGD 步长和 L2 正则系数 |
| early_stopping_patience | 验证指标连续未改善的等待轮数；0 禁用早停 |
| clip_predictions | 是否将预测裁剪到 [1,5]，验证与提交保持一致 |
| use_target_encoding | 预留选项，当前未实现，保持 false |

元数据包括购买标记、log1p 票数、启用文本域的长度，以及训练中用户/商品/父商品
出现次数。未知实体频率为 0，不把原始实体 ID 当连续数值，也不使用评分构造特征。

## 5. 后续手写实现入口

- `data.py`：CSV 读取、分组哈希和时间划分，后续迁移 Spark schema/聚合。
- `features.py`：分词、稳定哈希、IDF、元数据；保持训练拟合与预测转换边界。
- `model.py`：当前 sklearn SGD；后续替换为同步小批次稀疏梯度算法。
- `optimization.py`：提供可独立运行的分区梯度、聚合及一次 L2 参数更新参考函数；
  当前主训练流程不调用这些函数，避免声称已实现自主分布式训练。
- `evaluation.py` / `submission.py`：当前 sklearn 指标和 pandas 导出，后续改为分布式聚合/流式导出。
- `spark_runtime.py`：保留会话创建、硬件记录及关闭契约，当前不调用。

类和函数注释记录了输入、输出、状态变化和后续替换方向。当前 MVP 信任输入，
没有额外增加逐字段校验或异常处理框架。
