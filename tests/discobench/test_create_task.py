from collections.abc import Iterator
from unittest.mock import ANY, MagicMock, mock_open, patch

import pytest

from discobench import create_task


@pytest.fixture
def mock_make_files() -> Iterator[MagicMock]:
    """Mock make_files as this is tested separately, and we don't want to deal with making new files here."""
    # Patch the class itself. We yield the class mock 'mocked' so we can
    # assert the constructor was called.
    with patch("discobench.create_task.MakeFiles") as mocked:
        yield mocked


@pytest.fixture
def mock_yaml() -> Iterator[None]:
    """Mock yaml.safe_load so we don't need to access real files."""
    with (
        patch("yaml.safe_load", return_value={"mock_key": "mock_val"}),
        patch("builtins.open", mock_open(read_data="config content")),
    ):
        yield


def test_create_task_too_many_configs() -> None:
    """Ensure the function enforces mutual exclusivity."""
    with pytest.raises(ValueError, match="Provide only one of"):
        create_task("OnPolicyRL", test=True, example=True, config_path="discobench/tasks/OnPolicyRL/task_config.yaml")

    with pytest.raises(ValueError, match="Provide only one of"):
        create_task("OnPolicyRL", test=True, example=True, config_dict={"abc": "123"})

    with pytest.raises(ValueError, match="Provide only one of"):
        create_task(
            "OnPolicyRL",
            test=True,
            config_dict={"abc": "123"},
            config_path="discobench/tasks/OnPolicyRL/task_config.yaml",
        )


def test_create_task_example_path_resolution(mock_make_files: MagicMock, mock_yaml: None) -> None:
    """Verifies that setting example=True points to the correct subdirectory."""
    task_domain = "LanguageModelling"
    create_task(task_domain, test=False, example=True, cache_root="abc")

    # Verify MakeFiles was instantiated with the correct domain using the fixture
    mock_make_files.assert_called_once_with(task_domain, cache_root="abc")

    # Check if make_files was called with the 'train' flag correctly (test=False -> train=True)
    # Note: We access .return_value because mock_make_files is the Class Mock
    mock_make_files.return_value.make_files.assert_called_once()
    _, kwargs = mock_make_files.return_value.make_files.call_args
    assert kwargs["train"] is True


@pytest.mark.parametrize("eval_type", ["performance", "energy", "time"])
@pytest.mark.parametrize("baseline_scale", [0.5, 1.0, 1.5])
def test_create_task_with_config_dict(mock_make_files: MagicMock, eval_type: str, baseline_scale: float) -> None:
    """Ensures that passing a dict bypasses file reading entirely."""
    custom_config = {"train_task_id": ["Ackley1D", "Ackley2D"]}

    create_task(
        "BayesianOptimisation", test=True, config_dict=custom_config, baseline_scale=baseline_scale, eval_type=eval_type
    )

    mock_make_files.return_value.make_files.assert_called_once_with(
        custom_config, train=False, use_base=False, no_data=False, baseline_scale=baseline_scale, eval_type=eval_type
    )


@pytest.mark.parametrize("test_mode, expected_train", [(True, False), (False, True)])
@pytest.mark.parametrize("eval_type", ["performance", "energy", "time"])
@pytest.mark.parametrize("baseline_scale", [0.5, 1.0, 1.5])
def test_create_task_train_test_toggle(
    mock_make_files: MagicMock,
    mock_yaml: None,
    test_mode: bool,
    expected_train: bool,
    eval_type: str,
    baseline_scale: float,
) -> None:
    """Verifies the boolean inversion logic for the 'train' parameter."""
    create_task("GreenhouseGasPrediction", test=test_mode, baseline_scale=baseline_scale, eval_type=eval_type)

    mock_make_files.return_value.make_files.assert_called_with(
        ANY, train=expected_train, use_base=False, no_data=False, baseline_scale=baseline_scale, eval_type=eval_type
    )


@pytest.mark.parametrize(
    "example, expected_path",
    [
        (True, "discobench/example_configs/GreenhouseGasPrediction.yaml"),
        (False, "discobench/tasks/GreenhouseGasPrediction/task_config.yaml"),
    ],
)
def test_create_task_config_path_selection(mock_make_files: MagicMock, example: bool, expected_path: str) -> None:
    """Verifies the path selection logic based on the example flag."""
    with (
        patch("builtins.open", mock_open(read_data="mock_yaml: true")) as mocked_file,
        patch("yaml.safe_load", return_value={"mock_yaml": True}),
    ):
        create_task("GreenhouseGasPrediction", test=False, example=example)

        actual_path_called = mocked_file.call_args[0][0]

        assert str(actual_path_called).endswith(expected_path)
        assert "discobench" in str(actual_path_called)


def test_create_task_invalid_eval_type(mock_make_files: MagicMock) -> None:
    """Ensure invalid eval_type raises ValueError."""
    with pytest.raises(ValueError, match="eval_type"):
        create_task("OnPolicyRL", test=False, config_dict={"a": 1}, eval_type="invalid")


@pytest.mark.parametrize("scale", [0.0, -1.0, -0.001])
def test_create_task_invalid_baseline_scale(mock_make_files: MagicMock, scale: float) -> None:
    """Ensure non-positive baseline_scale raises ValueError."""
    with pytest.raises(ValueError, match="greater than 0"):
        create_task("OnPolicyRL", test=False, config_dict={"a": 1}, baseline_scale=scale)


def test_create_task_no_data_and_use_base_forwarded(mock_make_files: MagicMock) -> None:
    """Ensure no_data and use_base are passed through to make_files."""
    config = {"train_task_id": ["t1"]}
    create_task("OnPolicyRL", test=False, config_dict=config, no_data=True, use_base=True)

    mock_make_files.return_value.make_files.assert_called_once_with(
        config, train=True, use_base=True, no_data=True, eval_type="performance", baseline_scale=1.0
    )


def test_create_task_default_eval_type(mock_make_files: MagicMock) -> None:
    """Ensure eval_type defaults to 'performance' when not specified."""
    config = {"train_task_id": ["t1"]}
    create_task("OnPolicyRL", test=False, config_dict=config)

    _, kwargs = mock_make_files.return_value.make_files.call_args
    assert kwargs["eval_type"] == "performance"
    assert kwargs["baseline_scale"] == 1.0


def test_create_task_with_config_path(mock_make_files: MagicMock) -> None:
    """Ensure passing config_path reads from that path."""
    with (
        patch("builtins.open", mock_open(read_data="key: val")) as mocked_file,
        patch("yaml.safe_load", return_value={"key": "val"}),
    ):
        create_task("OnPolicyRL", test=False, config_path="/some/custom/path.yaml")

        mocked_file.assert_called_once_with("/some/custom/path.yaml")
        mock_make_files.return_value.make_files.assert_called_once_with(
            {"key": "val"}, train=True, use_base=False, no_data=False, eval_type="performance", baseline_scale=1.0
        )
