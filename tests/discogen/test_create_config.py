from collections.abc import Iterator
from unittest.mock import mock_open, patch

import pytest

from discogen import create_config
from discogen.utils import get_domains


@pytest.fixture
def mock_yaml() -> Iterator[None]:
    """Mock yaml.safe_load to return not a dict, to check TypeError."""
    with patch("yaml.safe_load", return_value="test123"), patch("builtins.open", mock_open(read_data="config content")):
        yield


def test_domain_doesnt_exist() -> None:
    """Make sure the function raises for invalid task domains."""
    with pytest.raises(ValueError, match="Invalid task_domain!"):
        create_config("abcdefg")


@pytest.mark.parametrize("domain", get_domains())
def test_all_domains_valid_config(domain: str) -> None:
    """Ensure all domains supported in DiscoGen can be loaded in, and produce a dict."""
    config = create_config(domain)
    assert type(config) is dict


def test_config_is_dict(mock_yaml: None) -> None:
    """Ensure that if a task_domain did not have a valid config, a TypeError is raised."""
    with pytest.raises(TypeError, match="not a dictionary"):
        create_config("OnPolicyRL")
