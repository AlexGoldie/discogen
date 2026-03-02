import os

import pytest

UTILS_PATH = "discobench/utils"


def test_exist() -> None:
    """Ensure expected files are in discobench/utils."""
    expected_files = ["__init__.py", "description.md", "get_domains.py", "get_modules.py", "make_files.py"]

    for file in expected_files:
        if not os.path.exists(f"{UTILS_PATH}/{file}"):
            pytest.fail(f"{file} does not exist!")


def test_run_main_exist() -> None:
    """Ensure run_main_{x} is in discobench/utils/run_mains."""
    expected_run_mains = ["energy", "performance", "time"]

    for name in expected_run_mains:
        if not os.path.exists(f"{UTILS_PATH}/run_mains/run_main_{name}.py"):
            pytest.fail(f"{name} does not exist!")
