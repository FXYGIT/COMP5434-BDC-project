# 模板配置使用说明

`task1.json` 对应天气分类，`task2.json` 对应评论评分预测。
参数是待验证的起始值，不是已调优结果；当前 Task 1/2 均可运行本地 MVP。

## 公共字段

| 字段 | 作用 |
|---|---|
| task | 选择 task1 或 task2 |
| train_path | 训练数据文件 |
| test_path | 测试数据文件 |
| sample_path | 提交示例文件 |
| output_dir | 按时间命名的运行记录根目录 |
| model_dir | 正式模型保存目录，与运行目录独立 |
| seed | 随机种子 |
| params | 任务专属配置，原样传给后续模块 |

所有字段均需填写；路径相对于 JSON 文件所在目录，而不是启动命令的目录。
例如 configs/task1.json 中 ../data/task1/train.csv 指向项目的 data/task1/train.csv。
也可填写绝对路径，Windows 路径建议使用 D:/data/train.csv 形式。

## Task 1 参数

split_strategy / valid_fraction：分组分层划分及验证比例；model：自定义森林；
n_trees / max_depth / min_leaf_samples：树数量、最大深度及叶节点最小样本数；
class_weight：类别权重策略；missing_indicators：是否增加缺失指示特征。

## Task 2 参数

当前使用本地后端，不创建 Spark 会话。max_train_rows：训练 CSV 前 N 行，默认 50000，null 为全量；
n_jobs：本地特征转换线程数，默认 1，测试集始终完整读取；
split_strategy / valid_fraction：稳定哈希分组划分及验证比例；
model：当前为 sklearn SGD 回归，后续替换为自定义分布式线性回归；text_dimension_per_field：标题和正文各自的哈希维度；
use_title / use_comment / use_metadata / use_idf：对应特征开关；
use_target_encoding：预留折外目标编码选项，当前未实现，保持 false；
epochs / batches_per_epoch：训练轮数及每轮批次数；learning_rate：学习率；
l2：正则化系数；early_stopping_patience：验证指标连续不改善的等待轮数；
clip_predictions：是否将评分裁剪到 [1,5]。

## 调用

```powershell
project experiment --config configs/task1.json
project experiment --config configs/task2.json
```

Task 2 全部命令、输入和产出详见 [执行说明](../docs/task2_doc.md)。

模型目录示例：`../models/task1/random_forest`、`../models/task2/rating_regressor`。
运行子目录按北京时间命名：`2026-10-16_14-30-25-123456_experiment`。
预测时直接用 `--model models/task1/random_forest` 指定已有模型。
