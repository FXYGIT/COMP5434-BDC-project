"""只验证正常输入下的 CLI 命令分发。"""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch

from project import cli
from project.common.contracts import ModelBundle, TaskConfig


class CliTests(TestCase):
    def test_command_dispatch(self):
        config = TaskConfig(
            "task1",
            Path("train.csv"),
            Path("test.csv"),
            Path("sample.csv"),
            Path("runs"),
            5434,
            {},
        )
        for task in ["task1", "task2"]:
            current = replace(config, task=task)
            for command, method in [
                ("experiment", "run_experiment"),
                ("train-final", "train_final"),
                ("predict", "predict_submission"),
            ]:
                with (
                    self.subTest(task=task, command=command),
                    patch("project.cli.load_config", return_value=current),
                    patch(f"project.{task}.pipeline.{method}", return_value="done") as run,
                    redirect_stdout(StringIO()) as output,
                ):
                    args = [command, "--config", "config.json"]
                    if command == "predict":
                        args += ["--model", "model"]
                    cli.main(args)
                    if command == "predict":
                        run.assert_called_once_with(
                            current,
                            ModelBundle(Path("model"), Path("model/manifest.json")),
                        )
                    else:
                        run.assert_called_once_with(current)
                    self.assertEqual(output.getvalue(), "done\n")
        for command, method in [
            ("baseline", "run_baselines"),
            ("ablate", "run_ablations"),
            ("benchmark", "run_scaling_study"),
        ]:
            with (
                self.subTest(command=command),
                patch("project.cli.load_config", return_value=current),
                patch(f"project.experiments.{method}", return_value=[]) as run,
                redirect_stdout(StringIO()),
            ):
                cli.main([command, "--config", "config.json"])
                if command == "benchmark":
                    run.assert_called_once_with(current, [1, 2, 4, 8])
                else:
                    run.assert_called_once_with(current)

    def test_report_tables(self):
        with (
            patch("project.experiments.export_report_tables", return_value=Path("tables")) as run,
            patch("project.cli.load_config") as load,
            redirect_stdout(StringIO()),
        ):
            cli.main(["report-tables", "--runs", "run1", "run2", "--output-dir", "tables"])
        run.assert_called_once_with([Path("run1"), Path("run2")], Path("tables"))
        load.assert_not_called()

    def test_parallelisms(self):
        args = cli.parse_args(["benchmark", "--config", "config.json", "--parallelisms", "2", "6"])
        self.assertEqual(args.parallelisms, [2, 6])

    def test_help(self):
        with (
            redirect_stdout(StringIO()) as output,
            self.assertRaises(SystemExit) as exit,
        ):
            cli.main(["--help"])
        self.assertEqual(exit.exception.code, 0)
        self.assertIn("report-tables", output.getvalue())


if __name__ == "__main__":
    main()
