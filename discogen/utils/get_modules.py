import pathlib


def get_modules() -> dict[str, list[str]]:
    """Function to see all modules available in each DiscoGen domain.

    Returns:
        Dictionary of {domain: modules}.
    """
    task_path = pathlib.Path(__file__).parent.parent / "domains"
    module_dict = {}
    for task in task_path.iterdir():
        domain_path = task / "templates/default/base"

        modules = [
            mod.stem  # removes .py
            for mod in domain_path.iterdir()
            if mod.is_file() and mod.suffix == ".py"
        ]

        module_dict[task.name] = modules
    return module_dict
