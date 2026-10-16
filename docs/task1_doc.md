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
- 取消 bootstrap 抽样（每棵树改用全部训练行）。
- 将随机森林替换为单棵决策树。

缺失指示特征不再作为消融项：本数据没有任何缺失值，7 个掩码列恒为 0，而模型在训练前
会把整列恒定的特征从候选池里剔除，因此 `full` 和 `no_missing_indicators` 会拿到同一份
候选池、建出完全相同的树，差异恒为 0。这一臂换成了信息量真实的 bootstrap 消融。

**产出：** `runs/task1/<时间>_ablate/`，包含运行清单、划分和各方案的指标及耗时。

该命令用于观察各组件对验证结果的影响，不会自动修改配置。

### 6. 并行效率实验：benchmark

```powershell
uv run project benchmark --config configs/task1.json --parallelisms 1 2 4 8
```

**输入：** 配置文件、`train.csv` 和并行度列表。

固定数据划分和模型参数，只改手写森林的 `n_jobs`（按树并行训练用的线程数），比较不同并行度的训练耗时和加速比。

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

`submission.csv` 包含 `id,label` 两列，标签为整数 `1..4`（1 晴、2 多云、3 雨、4 雪）。该命令不会重新训练模型，也不会自动提交到 Kaggle。

注意 `sample_submission.csv` 的 `label` 列填的是类别名 `Rainy` 而不是整数，那只是出题方
随手拿一个类别当占位；已确认提交要求是整数 `1..4`。导出时 `export_weather_submission`
会直接用预测结果覆盖样例里的这一列，所以不需要任何额外转换，也不要把这里改成类别名。

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

模型是手写的 CART 决策树和随机森林（`src/project/task1/model.py`），特征处理、参数比较和报告使用 pandas。**sklearn 只用于数据划分（`train_test_split`）和评估指标（`f1_score` 等），不参与预测**，符合"库只负责数组运算和 I/O"的约定。

已完成：类别特征编码（按训练折的类别清单取整数编码，未见类别归入未知桶）、数值列中位数填充、手写 CART 与随机森林（含按树并行）。

尚未实现：重复记录分组划分。当前数据经核对没有重复记录，按行分层即可；若以后出现重复记录，需要改成按组划分，避免同一条记录同时落进训练集和验证集。

### 11. 算法说明：手写 CART 与随机森林

作业要求算法细节由自己实现，不允许把现成的分类模型当黑盒调用，所以模型部分是自己写的。入口是 `configs/task1.json` 里的 `"model": "custom_random_forest"`。

#### 输入特征

预处理后是 17 列：10 个原始特征（7 个数值列 + 3 个类别列）加上 7 个数值列的缺失掩码。类别列按训练折的类别清单编码成整数，未见类别（含缺失）归到与所有已知类别都不同的未知桶；数值列缺失用训练折中位数填充。**模型拿到的是填好缺失的数值矩阵，所以树里不需要任何缺失值分支。**

#### 单棵树（CART）

- **分裂准则**：加权 Gini 不纯度 `1 - Σ p_c²`。每个节点枚举候选列的候选切点，取"两侧加权不纯度之和"最小的那个；父节点那一项对同一节点的所有候选都相同，所以最小化它就等价于最大化 Gini 下降。
- **候选切点**：把该列在本节点的取值排序，用累积和一次性算出"切在第 i 个位置之后"时两侧的各类别权重和，再向量化算两侧 Gini。阈值取相邻两个**不同**取值的中点，保证 `x <= 阈值` 不会正好落在训练值上，没有归属歧义。
- **停止条件**：到 `max_depth`；样本数小于 `min_samples_split`；样本数小于 `2 × min_samples_leaf`（分完一侧会不够）；节点只剩一类；或者所有候选列在本节点都是常数（切不动）。
- **叶节点分布**：该叶的加权类别频率，归一化后直接当概率。**不做拉普拉斯平滑**——`min_samples_leaf = 5` 保证每个叶子至少 5 个样本，分母不可能为 0，而加伪计数只会把概率往均匀分布拉、损害准确率。
- **节点级子特征抽样**：每个节点从候选池里不放回随机抽 `max_features` 列。这是随机森林降相关的关键，也是单棵树和森林的每棵树完全同规则的地方。候选池在建树前就算好——把**整列取值恒定**的特征剔除掉，因为这种列在任何节点上都不可能成为分裂点，留在池子里只会白占一个名额（本数据 17 列里有 7 列正是如此）。节点内部恰好也是常数的列由切点搜索自己跳过兜底。

#### 森林

- **bootstrap**：第 i 棵树有放回抽 n 行（n 是训练样本数）。重复的行用重复下标表示，不复制数据。
- **概率聚合**：每棵树对每个样本给一个 4 维概率，森林取平均后再取最大者作为标签。类别顺序固定为 1、2、3、4，某棵树没抽到某个类别也不需要特判。
- **按树并行**：`n_jobs` 为 `None` 或 `1` 时顺序训练，`-1` 用满所有核，其它正整数用该线程数（默认 `-1`）。线程之间共享同一份训练矩阵，不复制。

#### 结果与线程数无关

第 i 棵树的随机源固定是 `seed + i`，与线程调度、任务完成顺序都无关；`ThreadPoolExecutor.map` 也按输入顺序返回结果。所以同一份数据在不同 `n_jobs` 下训练出的模型完全相同——`benchmark` 里各并行度的 `macro_f1` 必须一模一样，这正是并行实现正确性的检验。

#### 实测：按树线程并行在本任务上不划算

`benchmark` 的结果是四条变体的 `macro_f1` 完全相同（正确性通过），但加速比全部小于 1：
`n_jobs=1` 最快，`2/4/8` 反而**慢** 1.3~1.8 倍。为了排除"笔记本降频"这种干扰，又按
`1、2、1、2` 的顺序交错重跑了一遍，结论稳定复现——顺序的那两次快，并行的那两次慢，
与时间先后无关。

原因是每个节点的 numpy 运算都很小（几百到几千个元素），耗时主要在 Python 层的调用开销上，
多线程只能轮流持有 GIL，等于白添了线程切换的成本。**这是实测得到的负结论**，报告里如实写：
在本任务的数据规模下，按树线程并行没有收益；要真正加速得换多进程，绕开 GIL。

#### 两个和本数据集有关的必然结论

报告里要如实说明这两点，它们不是实现缺陷：

- **类别权重在本数据上恒为 1.0**：train 共 9900 行、四类各 2475，训练折 7920 行、每类恰好 1980，`n / (K × count_c) = 1.0`。加权 Gini 因此退化成普通 Gini，消融里的 `no_class_weight` 差异**恒为 0.0000**。结论是"在本数据上类别权重不可能有任何作用"，而不是实验坏了。
- **缺失掩码是全零列**：train 和 test 的每一列都**没有缺失值**，所以 7 个缺失掩码列恒为常数 0。模型现在在建树前就把整列恒定的特征剔出候选池（见上一节），这些死列不再稀释 `max_features` 的抽样。实测拿掉这个剔除动作，同一份划分上 macro-F1 从 **0.9143 掉到 0.9120**（−0.0023，0.9143 已略高于先前用 sklearn 随机森林记录的 0.9130），单次森林训练从约 **25.0 秒降到约 16.8 秒**。省时间的原因是：抽到死列时切点搜索会提前返回，连排序都不用做。这也解释了更早 `no_missing_indicators` 的差异为什么在 `+0.0036` 和 `+0.0005` 之间跳、符号还不稳定——掩码列从未携带信息，差异只来自"少抽到死列"这个副作用。