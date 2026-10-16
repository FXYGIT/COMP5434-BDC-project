# COMP5434-BDC-project


## Step 1
拉取项目后执行

1. 项目初始化

同步依赖
```python
uv sync --locked
uv run pre-commit install
```

下载数据集
```bash
kaggle auth login

kaggle competitions download comp-5434-20261-project-task-1 -p ./data/task-1
python -m zipfile -e "./data/task-1/comp-5434-20261-project-task-1.zip" "./data/task-1"

kaggle competitions download comp-5434-20261-project-task-2 -p ./data/task-2
python -m zipfile -e "./data/task-2/comp-5434-20261-project-task-2.zip" "./data/task-2"
```

2. 开始实验

task1 实验文档： ./docs/task1_doc.md## Task 2：本地 MVP

执行顺序、参数与产出见 [Task 2 执行说明](docs/task2_doc.md)。
当前使用 pandas、sklearn、SciPy 和 joblib，暂不启动 Spark。
默认读取前 50,000 条训练记录，预测覆盖全部测试记录。

```powershell
uv run project experiment --config configs/task2.json
uv run project train-final --config configs/task2.json
uv run project predict --config configs/task2.json --model models/task2/rating_regressor
```
