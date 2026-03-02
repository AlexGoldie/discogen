import os

import pytest
import yaml

from discogen import create_config
from discogen.utils import get_domains

DISCOGEN_CONFIG_PATH = "discogen/discobench_configs/"


@pytest.mark.parametrize("domain", get_domains())
def test_single_config_valid(domain: str) -> None:
    """Ensure single change DiscoBench configs exist."""
    expected_config = create_config(domain)

    for k, _ in expected_config.items():
        if "change_" in k:
            module_name = k[7:]

            discobench_change_config_path = f"{DISCOGEN_CONFIG_PATH}/{domain}_{module_name}.yaml"
            assert os.path.exists(discobench_change_config_path)
            with open(discobench_change_config_path) as f:
                discobench_config = yaml.safe_load(f)

            # Check all keys are there
            assert sorted(discobench_config.keys()) == sorted(expected_config.keys())

            changes = 0
            for d_k, d_v in discobench_config.items():
                assert type(d_v) is type(expected_config[d_k])
                if "change_" in d_k:
                    changes += d_v
            assert changes == 1

            # Ensure default key values are used when appropriate
            assert discobench_config["template_backend"] == "default"
            assert discobench_config["source_path"] == f"task_src/{domain}_{module_name}"
            assert len(discobench_config["train_task_id"]) >= 1
            assert len(discobench_config["test_task_id"]) >= 1
            assert list(set(discobench_config["train_task_id"]) & set(discobench_config["test_task_id"])) == []


@pytest.mark.parametrize("domain", get_domains())
def test_all_config_valid(domain: str) -> None:
    """Ensure all change DiscoBench configs exists."""
    expected_config = create_config(domain)

    discobench_all_config_path = f"{DISCOGEN_CONFIG_PATH}/{domain}_all.yaml"
    assert os.path.exists(discobench_all_config_path)
    with open(discobench_all_config_path) as f:
        discobench_config = yaml.safe_load(f)

    # Check all keys are there
    assert sorted(discobench_config.keys()) == sorted(expected_config.keys())

    changes = 0
    for k, v in discobench_config.items():
        assert type(v) is type(expected_config[k])
        if "change_" in k:
            assert v is True

    # Ensure default key values are used when appropriate
    assert discobench_config["template_backend"] == "default"
    assert discobench_config["source_path"] == f"task_src/{domain}_all"
    assert len(discobench_config["train_task_id"]) >= 1
    assert len(discobench_config["test_task_id"]) >= 1
    assert list(set(discobench_config["train_task_id"]) & set(discobench_config["test_task_id"])) == []
