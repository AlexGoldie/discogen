from discogen.utils import get_discobench_tasks, get_modules


def test_single_config_valid() -> None:
    """Ensure get_discobench_tasks gives all expected discobench tasks."""
    discobench_tasks = get_discobench_tasks()

    expected_modules = get_modules()

    for domain in expected_modules:
        expected_modules[domain].append("all")

    expected_discobench = []
    for domain, modules in expected_modules.items():
        for module in modules:
            expected_discobench.append(f"{domain}_{module}")

    assert sorted(expected_discobench) == sorted(discobench_tasks)
