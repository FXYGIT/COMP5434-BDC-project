"""命令行入口：解析参数，读取配置，再调用对应任务函数。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from project import experiments
from project.common.config import load_config
from project.common.contracts import ModelBundle
from project.task1 import pipeline as task1
from project.task2 import pipeline as task2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令及参数；argv 默认读取终端输入，返回包含命令和选项的对象。"""
    parser = argparse.ArgumentParser(description="COMP5434 项目")
    commands = parser.add_subparsers(dest="command", required=True)

    # experiment：在训练/验证划分上评估模型。
    experiment = commands.add_parser("experiment", help="运行验证实验")
    experiment.add_argument("--config", type=Path, required=True, help="配置文件路径")

    # train-final：使用全部训练数据重训最终模型。
    train_final = commands.add_parser("train-final", help="使用全部训练数据训练最终模型")
    train_final.add_argument("--config", type=Path, required=True, help="配置文件路径")

    # predict：读取保存的模型，生成预测文件。
    predict = commands.add_parser("predict", help="加载模型并生成预测 CSV")
    predict.add_argument("--config", type=Path, required=True, help="配置文件路径")
    predict.add_argument("--model", type=Path, required=True, help="模型目录")

    # baseline：运行基线实验。
    baseline = commands.add_parser("baseline", help="运行基线实验")
    baseline.add_argument("--config", type=Path, required=True, help="配置文件路径")

    # ablate：运行消融实验。
    ablate = commands.add_parser("ablate", help="运行消融实验")
    ablate.add_argument("--config", type=Path, required=True, help="配置文件路径")

    # benchmark：比较不同并行度下的运行效率。
    benchmark = commands.add_parser("benchmark", help="运行本地并行度实验（Spark 待接入）")
    benchmark.add_argument("--config", type=Path, required=True, help="配置文件路径")
    benchmark.add_argument(
        "--parallelisms",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="并行度列表，默认 1 2 4 8",
    )

    # 汇总已有实验，不需要训练配置。
    report = commands.add_parser("report-tables", help="生成报告表格")
    report.add_argument("--runs", type=Path, nargs="+", required=True, help="实验目录列表")
    report.add_argument("--output-dir", type=Path, required=True, help="表格输出目录")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """分发命令并打印结果；假定输入正确，错误由下游模块直接抛出。

    argv 为不含程序名的参数列表，默认读取终端输入；无返回值。
    """
    args = parse_args(argv)
    result: object

    if args.command == "report-tables":
        result = experiments.export_report_tables(args.runs, args.output_dir)
        print(result)
        return

    config = load_config(args.config)
    pipeline = {"task1": task1, "task2": task2}[config.task]

    if args.command == "experiment":
        result = pipeline.run_experiment(config)
    elif args.command == "train-final":
        result = pipeline.train_final(config)
    elif args.command == "predict":
        model = ModelBundle(args.model, args.model / "manifest.json")
        result = pipeline.predict_submission(config, model)
    elif args.command == "baseline":
        result = experiments.run_baselines(config)
    elif args.command == "ablate":
        result = experiments.run_ablations(config)
    elif args.command == "benchmark":
        result = experiments.run_scaling_study(config, args.parallelisms)

    print(result)
