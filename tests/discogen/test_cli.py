import os
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml
from click.testing import CliRunner

import discogen
from discogen.cli import cli


@pytest.fixture
def runner() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


@pytest.mark.parametrize(
    ["task_domain", "expected_task_domain"],
    [["OnPolicyRL", "OnPolicyRL"], ["OnPoLiCyRl", "OnPolicyRL"], ["OffPolicyRL", "OffPolicyRL"], ["abcdef", "abcdef"]],
)
@pytest.mark.parametrize("test", [True, False])
@pytest.mark.parametrize("example", [True, False])
@pytest.mark.parametrize("use_base", [True, False])
@pytest.mark.parametrize("no_data", [True, False])
@pytest.mark.parametrize("eval_type", ["performance", "time", "energy", "abcdef"])
@pytest.mark.parametrize("baseline_scale", [0.5, 1.5, 1.0])
@pytest.mark.parametrize("cache_root", ["cache1", "cache2"])
def test_create_task_cli(
    runner: CliRunner,
    task_domain: str,
    expected_task_domain: str,
    test: bool,
    example: bool,
    use_base: bool,
    no_data: bool,
    eval_type: str,
    baseline_scale: float,
    cache_root: str,
) -> None:
    """Test that create_task is called correctly. We already test create_task separately, so we just want to check echoes."""
    with patch("discogen.cli.create_task") as mock_create:
        args = [
            "create-task",
            "--task-domain",
            task_domain,
            "--eval-type",
            eval_type,
            "--baseline-scale",
            str(baseline_scale),
            "--cache-root",
            cache_root,
        ]
        if test:
            args.append("--test")
        if example:
            args.append("--example")
        if use_base:
            args.append("--use-base")
        if no_data:
            args.append("--no-data")

        results = runner.invoke(cli, args)

        mode = "test" if test else "training"

        if task_domain == "abcdef" or eval_type == "abcdef":
            if task_domain == "abcdef" or eval_type == "abcdef":
                mock_create.assert_not_called()
                assert results.exit_code != 0

                if task_domain == "abcdef":
                    assert "Invalid value for '--task-domain'" in results.output
                elif eval_type == "abcdef":
                    assert "Invalid value for '--eval-type'" in results.output

        else:
            # Click passes, so the mock MUST be called
            mock_create.assert_called_once_with(
                task_domain=expected_task_domain,
                test=test,
                config_path=None,
                example=example,
                use_base=use_base,
                no_data=no_data,
                eval_type=eval_type,
                baseline_scale=baseline_scale,
                cache_root=cache_root,
            )
            assert results.exit_code == 0
            assert f"Successfully created {mode} task for domain: {expected_task_domain}" in results.output

            if test and use_base:
                assert "--use-base has no effect" in results.output


def test_get_domains(runner: CliRunner) -> None:
    """Test that get_domains echoes in the expected way."""
    expected_domains = discogen.get_domains()

    results = runner.invoke(cli, "get-domains")

    assert results.exit_code == 0
    for domain in expected_domains:
        assert domain in results.output


def test_get_modules(runner: CliRunner) -> None:
    """Test that get_modules echoes in the expected way."""
    expected_modules = discogen.get_modules()

    results = runner.invoke(cli, "get-modules")

    assert results.exit_code == 0
    for domain, modules in expected_modules.items():
        expected_line = f"{domain}: {', '.join(modules)}"
        assert expected_line in results.output


def test_create_config_cmd(runner: CliRunner, tmp_path: Path) -> None:
    """Test that create-config from path works."""
    task_domain = "OnPolicyRL"

    save_dir = str(tmp_path / "custom_configs")
    expected_file_path = os.path.join(save_dir, f"task_config_{task_domain}.yaml")

    dummy_config = {"mocked_key": "mocked_value"}

    with patch("discogen.cli.create_config") as mock_create_config:
        mock_create_config.return_value = dummy_config

        results = runner.invoke(cli, ["create-config", "--task-domain", task_domain, "--save-dir", save_dir])

        assert results.exit_code == 0
        mock_create_config.assert_called_once_with(task_domain)

    assert os.path.exists(save_dir)
    assert os.path.isfile(expected_file_path)

    with open(expected_file_path) as f:
        saved_data = yaml.safe_load(f)

    assert saved_data == dummy_config


@pytest.mark.parametrize(
    ["task_name", "expected_task_name"],
    [
        ["OnPolicyRL_all", "OnPolicyRL_all"],
        ["OnPolicyRL_optim", "OnPolicyRL_optim"],
        ["OnPoLiCyRl_OpTiM", "OnPolicyRL_optim"],
        ["abcdef", "abcdef"],
    ],
)
@pytest.mark.parametrize("test", [True, False])
@pytest.mark.parametrize("use_base", [True, False])
@pytest.mark.parametrize("no_data", [True, False])
@pytest.mark.parametrize("eval_type", ["performance", "time", "energy", "abcdef"])
@pytest.mark.parametrize("cache_root", ["cache1", "cache2"])
def test_create_discobench_cli(
    runner: CliRunner,
    task_name: str,
    expected_task_name: str,
    test: bool,
    use_base: bool,
    no_data: bool,
    eval_type: str,
    cache_root: str,
) -> None:
    """Test that create_discobench is called correctly."""
    with patch("discogen.cli.create_discobench") as mock_create:
        args = ["create-discobench", "--task-name", task_name, "--eval-type", eval_type, "--cache-root", cache_root]
        if test:
            args.append("--test")
        if use_base:
            args.append("--use-base")
        if no_data:
            args.append("--no-data")

        results = runner.invoke(cli, args)

        mode = "test" if test else "training"

        if task_name == "abcdef" or eval_type == "abcdef":
            mock_create.assert_not_called()
            assert results.exit_code != 0

            if task_name == "abcdef":
                assert "Invalid value for '--task-name'" in results.output
            elif eval_type == "abcdef":
                assert "Invalid value for '--eval-type'" in results.output

        else:
            # Click passes, so the mock MUST be called
            mock_create.assert_called_once_with(
                task_name=expected_task_name,
                test=test,
                use_base=use_base,
                no_data=no_data,
                eval_type=eval_type,
                cache_root=cache_root,
            )
            assert results.exit_code == 0
            assert f"Successfully created {mode} discobench task: {expected_task_name}" in results.output

            if test and use_base:
                assert "--use-base has no effect" in results.output


class TestSampleTaskConfigCli:
    """Tests for the sample-task-config CLI command."""

    BASE_ARGS: ClassVar[list[str]] = [
        "sample-task-config",
        "--p-edit",
        "0.5",
        "--p-data",
        "[0.2,0.4,0.3]",
        "--p-use-base",
        "0.6",
        "--eval-type",
        "random",
        "--source-path",
        "task_src",
        "--max-attempts",
        "10",
        "--seed",
        "1",
        "--config-dest",
        "config.yaml",
    ]

    def _make_args(self, **overrides: str | float) -> list[str]:
        """Build args list, replacing any base args with overrides."""
        args = list(self.BASE_ARGS)
        for key, value in overrides.items():
            flag = f"--{key}"
            if flag in args:
                idx = args.index(flag)
                args[idx + 1] = str(value)
            else:
                args.append(flag)
        return args

    @patch("builtins.open", new_callable=mock_open)
    @patch("discogen.cli.sample_task_config")
    def test_happy_path(self, mock_sample: MagicMock, mock_file: MagicMock, runner: CliRunner) -> None:
        """Test basic successful invocation."""
        mock_sample.return_value = ("fake_domain", {"fake_key": "fake_value"})
        result = runner.invoke(cli, self.BASE_ARGS)

        assert result.exit_code == 0
        mock_sample.assert_called_once_with(
            p_edit=0.5,
            p_data=[0.2, 0.4, 0.3],
            p_use_base=0.6,
            eval_type="random",
            use_backends=True,
            source_path="task_src",
            max_attempts=10,
            seed=1,
        )
        mock_file.assert_called_once_with("config.yaml", "w")
        assert "Successfully saved new task_config for the fake_domain domain at config.yaml" in result.output

    @patch("builtins.open", new_callable=mock_open)
    @patch("discogen.cli.sample_task_config")
    def test_no_backends_flag(self, mock_sample: MagicMock, mock_file: MagicMock, runner: CliRunner) -> None:
        """Test that --no-backends sets use_backends to False."""
        mock_sample.return_value = ("fake_domain", {"fake_key": "fake_value"})
        args = [*self.BASE_ARGS, "--no-backends"]
        result = runner.invoke(cli, args)

        assert result.exit_code == 0
        assert mock_sample.call_args.kwargs["use_backends"] is False

    @pytest.mark.parametrize("eval_type", ["random", "performance", "time", "energy"])
    @patch("builtins.open", new_callable=mock_open)
    @patch("discogen.cli.sample_task_config")
    def test_valid_eval_types(
        self, mock_sample: MagicMock, mock_file: MagicMock, runner: CliRunner, eval_type: str
    ) -> None:
        """Test that valid eval types are passed through correctly."""
        mock_sample.return_value = ("fake_domain", {"fake_key": "fake_value"})
        result = runner.invoke(cli, self._make_args(**{"eval-type": eval_type}))

        assert result.exit_code == 0
        assert mock_sample.call_args.kwargs["eval_type"] == eval_type

    def test_invalid_eval_type(self, runner: CliRunner) -> None:
        """Test that an invalid eval type is rejected by Click."""
        result = runner.invoke(cli, self._make_args(**{"eval-type": "abcdef"}))

        assert result.exit_code != 0
        assert "Invalid value for '--eval-type'" in result.output

    @pytest.mark.parametrize(
        ["p_data", "expected"],
        [
            ("[0.1,0.7]", [0.1, 0.7]),
            ("[0.2,0.4,0.3]", [0.2, 0.4, 0.3]),
            ("[0.1,1.0]", [0.1, 1.0]),
            ("[1,2,3]", [1, 2, 3]),
        ],
    )
    @patch("builtins.open", new_callable=mock_open)
    @patch("discogen.cli.sample_task_config")
    def test_valid_p_data(
        self, mock_sample: MagicMock, mock_file: MagicMock, runner: CliRunner, p_data: str, expected: list[float]
    ) -> None:
        """Test that valid p_data values are parsed correctly."""
        mock_sample.return_value = ("fake_domain", {"fake_key": "fake_value"})
        result = runner.invoke(cli, self._make_args(**{"p-data": p_data}))

        assert result.exit_code == 0
        assert mock_sample.call_args.kwargs["p_data"] == expected

    @pytest.mark.parametrize("p_data", ["1,2,3]", "[1,2,3"])
    def test_invalid_p_data(self, runner: CliRunner, p_data: str) -> None:
        """Test that mismatched brackets in p_data are rejected."""
        result = runner.invoke(cli, self._make_args(**{"p-data": p_data}))

        assert result.exit_code != 0
        assert "p-data must either" in result.output

    def test_config_dest_must_end_with_yaml(self, runner: CliRunner) -> None:
        """Test that config_dest without .yaml extension is rejected."""
        result = runner.invoke(cli, self._make_args(**{"config-dest": "xyz"}))

        assert result.exit_code != 0
        assert "must end with .yaml" in result.output

    @pytest.mark.parametrize("p_edit", [0.3, 0.5, 0.7])
    @patch("builtins.open", new_callable=mock_open)
    @patch("discogen.cli.sample_task_config")
    def test_p_edit_passed_through(
        self, mock_sample: MagicMock, mock_file: MagicMock, runner: CliRunner, p_edit: float
    ) -> None:
        """Test that p_edit values are passed through correctly."""
        mock_sample.return_value = ("fake_domain", {"fake_key": "fake_value"})
        result = runner.invoke(cli, self._make_args(**{"p-edit": p_edit}))

        assert result.exit_code == 0
        assert mock_sample.call_args.kwargs["p_edit"] == p_edit

    @pytest.mark.parametrize("p_use_base", [0.0, 0.5, 1.0])
    @patch("builtins.open", new_callable=mock_open)
    @patch("discogen.cli.sample_task_config")
    def test_p_use_base_passed_through(
        self, mock_sample: MagicMock, mock_file: MagicMock, runner: CliRunner, p_use_base: float
    ) -> None:
        """Test that p_use_base values are passed through correctly."""
        mock_sample.return_value = ("fake_domain", {"fake_key": "fake_value"})
        result = runner.invoke(cli, self._make_args(**{"p-use-base": p_use_base}))

        assert result.exit_code == 0
        assert mock_sample.call_args.kwargs["p_use_base"] == p_use_base
