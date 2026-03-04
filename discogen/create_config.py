"""A function so you can just import create_task and it will return make_files_public, make_files_private and task_description."""

import os
from typing import Any

import yaml


def create_config(task_domain: str) -> dict[str, Any]:
    """Prepare a default config for the specific task.

    Notes:
        The default config will return with:
            - all possible train_task_ids
            - all possible test_task_ids
            - a single example model (if applies)
            - default task_destination of `task_src/`
            - default backend
            - all change_modules set to false

    Args:
        task_domain: The task domain to create the default config for.

    Returns:
        A dictionary containing the default config for the task.

    Raises:
        TypeError: If the loaded task_config is not a dictionary.
    """
    config_path = os.path.join(os.path.dirname(__file__), f"domains/{task_domain}/task_config.yaml")

    if not os.path.exists(config_path):
        raise ValueError(
            "Invalid task_domain! Please make sure to select a valid task_domain. You can use discogen.utils.get_domains() to get a full list of task domains in discogen."
        )

    with open(config_path) as f:
        task_config = yaml.safe_load(f)

    if not isinstance(task_config, dict):
        raise TypeError("Loaded task_config is not a dictionary")

    return task_config
