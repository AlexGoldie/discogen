"""A function to create a task given a task_domain, and whether to implement the meta-train or meta-test task. You can choose to generate an example task, feed in a config path or a config dict. You can specify the type of evaluation. It also supports creating a task without data, to allow you to observe the structure of a task."""

from pathlib import Path
from typing import Any

import yaml

from discogen.utils.make_files import MakeFiles


def create_task(
    task_domain: str,
    test: bool,
    example: bool = False,
    use_base: bool | None = None,
    no_data: bool = False,
    config_path: str | None = None,
    config_dict: dict[str, Any] | None = None,
    eval_type: str | None = None,
    baseline_scale: float = 1.0,
    cache_root: str = "cache",
) -> None:
    """Prepare files for the training or testing subset of the task.

    Args:
        task_domain: The task domain to create the task for.
        test: Whether to create the train or test version of a task (as defined by the config).
        example: Whether to use the pre-built example task_config for the task_domain.
        use_base: Whether to use the baseline implementations for each editable module. Defaults to False, meaning a default task will use an `edit` implementation (i.e., only the interface for a module is defined). This can also be defined in task_config. If use_base is provided as both an arg and in the config, this will raise an error. Defaults to False if neither are provided.
        no_data: Whether to create the codebase without loading any of the data files. If the code loads a pretrained model, this will also be skipped.
        config_path: The path to the task configuration file. If not provided, the default task configuration file will be used. Check `discogen/domains/{task_domain}/task_config.yaml` for expected structure for a given task.
        config_dict: A pre-built config dictionary, following the expected structure from `discogen/domains/{task_domain}/task_config.yaml`.
        eval_type: What type of evaluation to use. One of ['performance', 'time', 'energy']. In 'performance', the goal is to discover algorithms which maximise performance. In 'time', the goal is to discover algorithms that match the baseline performance in the shortest length of time. In 'energy', the objective is to discover algorithms which match the baseline performance using the least amount of estimated energy. This can also be provided in task_config. If eval_type is also provided in the config, passing an eval_type arg will raise an error. If neither are provided, defaults to performance.
        baseline_scale: What relative scale to allow compared to the baseline when using either the 'time' or 'energy' eval_type. If not provided, this will default to 1.0. Must be greater than 0.
        cache_root: A directory which data can be cached in.

    Notes:
        Only one of config_path, example OR config_dict (not more than one) should be passed as an argument here, to avoid any conflict.
    """
    explicit_configs = sum(arg is not None for arg in [config_path, config_dict])

    example_flag = 1 if example is True else 0

    if explicit_configs + example_flag > 1:
        raise ValueError("Provide only one of config_path, example=True, or config_dict.")

    if baseline_scale <= 0.0:
        raise ValueError("Relative tolerance must be greater than 0.")

    if config_path is None and config_dict is None:
        if example is True:
            config_path = str(Path(__file__).parent / f"example_configs/{task_domain}.yaml")
        else:
            config_path = str(Path(__file__).parent / f"domains/{task_domain}/task_config.yaml")
    if config_dict is not None:
        task_config = config_dict
    else:
        if config_path is None:
            if example is True:
                config_path = str(Path(__file__).parent / f"example_configs/{task_domain}.yaml")
            else:
                config_path = str(Path(__file__).parent / f"domains/{task_domain}/task_config.yaml")
        with open(config_path) as f:
            task_config = yaml.safe_load(f)

    train = not test

    eval_type, use_base = _resolve_config_overrides(task_config, eval_type, use_base)

    MakeFiles(task_domain, cache_root=cache_root).make_files(
        task_config, train=train, no_data=no_data, use_base=use_base, eval_type=eval_type, baseline_scale=baseline_scale
    )


def _resolve_config_overrides(
    task_config: dict[str, Any], eval_type: str | None, use_base: bool | None
) -> tuple[str, bool]:
    """Resolve eval_type and use_base from arg and config, raising on conflict."""
    config_eval = task_config.get("eval_type")

    if eval_type is not None and config_eval is not None and eval_type != config_eval:
        raise ValueError(
            f"eval_type specified both as argument ('{eval_type}') and in config ('{config_eval}'). "
            "Provide only one to avoid ambiguity."
        )

    eval_type = eval_type or config_eval or "performance"
    if eval_type not in ["performance", "time", "energy"]:
        raise ValueError("Ensure eval_type is one of ['performance', 'time', 'energy'].")

    config_use_base = task_config.get("use_base")

    if use_base is not None and config_use_base is not None and use_base != config_use_base:
        raise ValueError(
            f"use_base specified both as argument ({use_base}) and in config ({config_use_base}). "
            "Provide only one to avoid ambiguity."
        )

    if use_base is not None:
        use_base = use_base
    elif config_use_base is not None:
        use_base = config_use_base
    else:
        use_base = False

    return eval_type, use_base
