import pathlib


def get_discobench_tasks() -> list[str]:
    """Function to get all DiscoBench task names.

    Returns:
        List of [discobench_task_names].

    Note:
        Entries in this list can be passed to discobench.create_discobench() to load discobench tasks.
    """
    task_path = pathlib.Path(__file__).parent.parent / "discobench_configs"
    discobench_tasks = sorted([p.name.split(".")[0] for p in task_path.iterdir()])
    return discobench_tasks
