import pytest

from discogen import create_config
from discogen.utils import get_domains, get_modules


@pytest.mark.parametrize("domain, modules", get_modules().items())
def test_all_domains_valid_config(domain: str, modules: list[str]) -> None:
    """Ensure the modules loaded by get_modules matches the modules in the config file."""
    config = create_config(domain)
    expected_modules = [k[7:] for k in config if k.startswith("change_")]

    assert sorted(modules) == sorted(expected_modules)


def test_get_modules_all_domains() -> None:
    """Ensure all domains from get_domains() match the domains from get_modules()."""
    modules_dict = get_modules()
    domains_list = get_domains()

    assert sorted(modules_dict.keys()) == sorted(domains_list)
