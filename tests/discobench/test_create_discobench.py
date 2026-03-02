from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
import yaml

from discobench import create_discobench
from discobench.utils import get_discobench_tasks


@pytest.fixture
def mock_make_files() -> Iterator[MagicMock]:
    """Mock make_files as this is tested separately, and we don't want to deal with making new files here."""
    # Patch the class itself. We yield the class mock 'mocked' so we can
    # assert the constructor was called.
    with patch("discobench.create_discobench.MakeFiles") as mocked:
        yield mocked


@pytest.mark.parametrize("task_name", get_discobench_tasks())
@pytest.mark.parametrize("eval_type", ["performance", "energy", "time"])
@pytest.mark.parametrize("test_mode, expected_train", [(True, False), (False, True)])
def test_create_discobench_task(
    mock_make_files: MagicMock, test_mode: bool, expected_train: bool, eval_type: str, task_name: str
) -> None:
    """Tests that each discobench task can be created."""
    create_discobench(task_name, test=test_mode, eval_type=eval_type, no_data=True)

    with open(f"discobench/discobench_configs/{task_name}.yaml") as f:
        task_config = yaml.safe_load(f)

    mock_make_files.return_value.make_files.assert_called_with(
        task_config, train=expected_train, use_base=False, no_data=True, baseline_scale=1.0, eval_type=eval_type
    )
