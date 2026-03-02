import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

import discobench
from discobench.cli import cli


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
    """Test that create_task called correctly. We already test create_task separately, so we just want to check echos."""
    with patch("discobench.cli.create_task") as mock_create:
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
    """Test that get_domains echos in the expected way."""
    expected_domains = discobench.get_domains()

    results = runner.invoke(cli, "get-domains")

    assert results.exit_code == 0
    for domain in expected_domains:
        assert domain in results.output


def test_get_modules(runner: CliRunner) -> None:
    """Test that get_modules echos in the expected way."""
    expected_modules = discobench.get_modules()

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

    with patch("discobench.cli.create_config") as mock_create_config:
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
    """Test that create_discobench called correctly."""
    with patch("discobench.cli.create_discobench") as mock_create:
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
