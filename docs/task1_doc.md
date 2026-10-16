## Task 1：天气分类任务执行说明

所有命令均在项目根目录执行，使用 `configs/task1.json` 作为配置文件。

### 1. 数据和配置准备

确认数据文件已放置在以下位置：

```text
data/task-1/
├── train.csv
├── test.csv
└── sample_submission.csv
```

配置文件中的主要路径如下，均相对于配置文件所在目录：

```json
{
  "train_path": "../data/task-1/train.csv",
  "test_path": "../data/task-1/test.csv",
  "sample_path": "../data/task-1/sample_submission.csv",
  "output_dir": "../runs/task1",
  "model_dir": "../models/task1/random_forest"
}
```

### 2. 推荐执行顺序

完整流程：

```text
experiment → baseline / ablate / benchmark → 确定配置 → train-final → predict → report-tables
```

只需训练并生成提交文件时：

```text
train-final → predict
```

**初始没有模型文件也可以执行实验。** `experiment`、`baseline`、`ablate` 和 `benchmark` 都会从零训练临时模型，再在验证集上评估；只有 `predict` 需要读取已保存的正式模型。

### 3. 验证实验：experiment

```powershell
uv run project experiment --config configs/task1.json
```

**输入：** 配置文件和 `train.csv`。

执行流程：

1. 根据 `valid_fraction` 划分训练集和验证集，默认比例为 80% / 20%。
2. 仅使用训练折拟合预处理并从零训练模型。
3. 预测验证折，计算 Macro-F1、准确率、逐类 F1 和混淆矩阵。
4. 保存划分、指标和耗时。

**产出：**

```text
runs/task1/<时间>_experiment/
├── manifest.json
├── split.json
└── metrics.json
```

该命令用于评估当前配置，**不保存正式模型**。

### 4. 基线比较：baseline

```powershell
uv run project baseline --config configs/task1.json
```

**输入：** 配置文件和 `train.csv`。

在同一份训练/验证划分上，分别训练并评估：

- 多数类预测模型。
- 单棵决策树。
- 随机森林。

**产出：** `runs/task1/<时间>_baseline/`，包含 `manifest.json`、`split.json` 和记录各模型指标及耗时的 `metrics.json`。

### 5. 消融实验：ablate

```powershell
uv run project ablate --config configs/task1.json
```

**输入：** 配置文件和 `train.csv`。

在同一份划分上比较以下方案：

- 当前完整配置。
- 取消类别权重。
- 取消缺失指示特征。
- 将随机森林替换为单棵决策树。

**产出：** `runs/task1/<时间>_ablate/`，包含运行清单、划分和各方案的指标及耗时。

该命令用于观察各组件对验证结果的影响，不会自动修改配置。

### 6. 并行效率实验：benchmark

```powershell
uv run project benchmark --config configs/task1.json --parallelisms 1 2 4 8
```

**输入：** 配置文件、`train.csv` 和并行度列表。

固定数据划分和模型参数，使用 sklearn 随机森林的 `n_jobs` 比较不同 CPU 并行度。

**产出：** `runs/task1/<时间>_benchmark/`，包含各并行度的指标、训练耗时和相对列表中第一个并行度的加速比。

### 7. 全量训练并保存模型：train-final

根据实验结果，手动确定 `configs/task1.json` 中的参数，然后执行：

```powershell
uv run project train-final --config configs/task1.json
```

**输入：** 配置文件和完整 `train.csv`。

使用全部训练数据重新拟合预处理并从零训练模型，不再划分验证集。

**产出：**

```text
models/task1/random_forest/
├── manifest.json
└── state.pkl

runs/task1/<时间>_train-final/
├── manifest.json
└── metrics.json
```

模型包保存配置、预处理状态和随机森林状态；运行目录保存模型位置、硬件信息和耗时。

已有模型目录不会被覆盖。再次训练时，将配置中的 `model_dir` 改为新目录，例如：

```json
"model_dir": "../models/task1/random_forest_v2"
```

### 8. 生成提交文件：predict

```powershell
uv run project predict --config configs/task1.json --model models/task1/random_forest
```

**输入：**

- `--model` 指定的模型目录。
- 配置中的 `test.csv` 和 `sample_submission.csv`。
- 当前数据加载接口还会读取 `train.csv`，因此该文件也需要存在，但不会用于重新训练。

执行时恢复模型包中的预处理和模型状态，预测测试集，并按提交样例的 ID 顺序导出结果。

**产出：**

```text
runs/task1/<时间>_predict/
├── manifest.json
├── metrics.json
└── submission.csv
```

`submission.csv` 包含 `id,label` 两列，标签为整数 `1..4`。该命令不会重新训练模型，也不会自动提交到 Kaggle。

如果正式模型保存在其他目录，需要同步修改 `--model` 参数。

### 9. 汇总实验结果：report-tables

```powershell
uv run project report-tables `
  --runs `
    runs/task1/<时间>_experiment `
    runs/task1/<时间>_baseline `
    runs/task1/<时间>_ablate `
    runs/task1/<时间>_benchmark `
    runs/task1/<时间>_train-final `
    runs/task1/<时间>_predict `
  --output-dir outputs/task1-tables
```

将 `<时间>` 替换为各次运行生成的实际目录名。可以只选择需要汇总的运行目录。

**输入：** 各运行目录中的 `metrics.json`。

**产出：**

```text
outputs/task1-tables/results.csv
```

表格记录各实验的指标、耗时和来源目录；未测量的字段留空。此命令不执行训练。

### 10. 运行记录与当前实现

每次实验、训练和预测都会创建独立的北京时间运行目录，格式为：

```text
YYYY-MM-DD_HH-MM-SS-微秒_命令
```

目前为最小 MVP：模型使用 sklearn，数据处理和报告使用 pandas。代码中的算法说明保留，供后续替换为手写实现使用。类别特征编码和重复记录分组划分尚未完成。