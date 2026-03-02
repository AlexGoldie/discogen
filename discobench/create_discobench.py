"""A function to create a discobench task given the task_name. A list of task names can got from discobench.utils.get_discobench_tasks."""

from pathlib import Path

import yaml

from discobench.utils.make_files import MakeFiles


def create_discobench(
    task_name: str,
    test: bool,
    use_base: bool = False,
    no_data: bool = False,
    eval_type: str | None = "performance",
    cache_root: str = "cache",
) -> None:
    """Prepare files for the training or testing subset of a discobench task.

    Args:
        task_name: The DiscoBench task name.
        test: Whether to create the train or test version of a task (as defined by the config).
        use_base: Whether to use the baseline implementations for each editable module. Defaults to False, meaning a default task will use an `edit` implementation (i.e., only the interface for a module is defined).
        no_data: Whether to create the codebase without loading any of the data files. If the code loads a pretrained model, this will also be skipped.
        eval_type: What type of evaluation to use. One of ['performance', 'time', 'energy']. In 'performance', the goal is to discover algorithms which maximise performance. In 'time', the goal is to discover algorithms that match the baseline performance in the shortest length of time. In 'energy', the objective is to discover algorithms which match the baseline performance using the least amount of estimate emissions.
        cache_root: A directory to which data can be downloaded and cached.
    """
    config_path = str(Path(__file__).parent / f"discobench_configs/{task_name}.yaml")

    if eval_type not in ["performance", "time", "energy"]:
        raise ValueError("Ensure eval_type is one of ['performance', 'time', 'energy'].")

    with open(config_path) as f:
        task_config = yaml.safe_load(f)

    train = not test

    task_domain = task_name.split("_")[0]

    MakeFiles(task_domain, cache_root=cache_root).make_files(
        task_config, train=train, use_base=use_base, no_data=no_data, eval_type=eval_type, baseline_scale=1
    )
