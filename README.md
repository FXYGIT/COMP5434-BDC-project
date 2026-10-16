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