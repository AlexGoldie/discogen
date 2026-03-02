"""Command line interface for DiscOGen."""

import os

import click
import yaml

from discogen import (
    create_config,
    create_discobench,
    create_task,
    get_discobench_tasks,
    get_domains,
    get_modules,
    sample_task_config,
)


@click.group()
def cli() -> None:
    """DiscoGen: Procedural Generation of Algorithm Discovery Tasks in Machine Learning."""
    pass


@cli.command("create-task")
@click.option(
    "--task-domain",
    type=click.Choice(get_domains(), case_sensitive=False),
    required=True,
    help="The task domain to create the task for.",
)
@click.option("--test", is_flag=True, help="If passed, create test task instead of training task.")
@click.option("--example", is_flag=True, help="If passed, use example task config rather than your own.")
@click.option(
    "--config-path",
    type=str,
    help="The path to your task_config.yaml. If not provided, this will default to the DiscoGen task_config.yaml for your provided task_domain.",
)
@click.option(
    "--use-base",
    is_flag=True,
    help="If passed, will initialise editable modules with baseline implementations instead of interface-only `edit` implementations. Has no effect with --test.",
)
@click.option(
    "--no-data",
    is_flag=True,
    help="If passed, will create the task without downloading the data. The task code will generally not be able to run, but this will allow you to see how the code looks for a specific task.",
)
@click.option(
    "--eval-type",
    type=click.Choice(["performance", "time", "energy"], case_sensitive=False),
    default="performance",
    help="What type of evaluation to use. Options are 'performance' (find the highest performance algorithm), 'time' (find the algorithm which matches baseline performance in the least time) and 'energy' (find the algorithm which matched the baseline performance using the least energy). Default: performance",
)
@click.option(
    "--baseline-scale",
    type=float,
    default=1.0,
    help="If using 'time' or 'energy' evaluation, what tolerance is allowed compared to baseline score. For instance, if this is 0.5, an algorithm is valid if it reaches a score within 0.5 of the baseline. Default: 1.0. Must be above 0.",
)
@click.option("--cache-root", type=str, default="cache", help="A directory to which data can be downloaded and cached.")
def create_task_cmd(
    task_domain: str,
    test: bool,
    example: bool,
    use_base: bool,
    no_data: bool,
    eval_type: str,
    baseline_scale: float,
    cache_root: str,
    config_path: str | None = None,
) -> None:
    """Create task source files for a specified task domain."""
    if test and use_base:
        click.echo("Warning: --use-base has no effect with --test. Test tasks use discovered files from training.")
    if example and config_path:
        click.echo("Warning: passing example and config_path will cause an error.")

    create_task(
        task_domain=task_domain,
        test=test,
        config_path=config_path,
        use_base=use_base,
        example=example,
        no_data=no_data,
        eval_type=eval_type,
        baseline_scale=baseline_scale,
        cache_root=cache_root,
    )
    mode = "test" if test else "training"
    click.echo(f"Successfully created {mode} task for domain: {task_domain}.")


@cli.command("get-domains")
def get_domains_cmd() -> None:
    """List all available task domains in DiscoGen."""
    domains = get_domains()
    click.echo("\n".join(domains))


@cli.command("get-modules")
def get_modules_cmd() -> None:
    """List all available modules for a specified task domain."""
    module_dict = get_modules()
    for domain, modules in module_dict.items():
        click.echo(f"{domain}: {', '.join(modules)}")


@cli.command("get-discobench")
def get_discobench_tasks_cmd() -> None:
    """List name of all discobench tasks currently supported."""
    discobench_list = get_discobench_tasks()
    click.echo("\n".join(discobench_list))


@cli.command("create-config")
@click.option("--task-domain", type=str, required=True, help="The task domain to create the task for.")
@click.option(
    "--save-dir", type=str, required=False, default="task_configs", help="The directory to save the config to."
)
def create_config_cmd(task_domain: str, save_dir: str) -> None:
    """Save default config for editing for a specified task domain."""
    config = create_config(task_domain)

    os.makedirs(save_dir, exist_ok=True)

    with open(f"{save_dir}/task_config_{task_domain}.yaml", "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


@cli.command("sample-task-config")
@click.option(
    "--p-edit",
    type=float,
    default=0.3,
    help="The probability a module is marked as editable. Must be between 0. and 1.",
)
@click.option(
    "--p-data",
    type=str,
    default="[0.4,0.4,0.2]",
    help="A list of probabilities or weights for sampling. Supports either a list of 2 values, which must be [p_meta_train, p_meta_test], or a list of 3 values, which can be probabilities or weights [w_meta_train, w_meta_test, w_exclude]. Can be passed with or without [].",
)
@click.option(
    "--eval-type",
    type=click.Choice(["performance", "time", "energy", "random"], case_sensitive=False),
    default="random",
    help="What eval_type to use. Supports 'random', which will select a random eval_type, or one of ['performance', 'energy', 'time']. Defaults to 'random'.",
)
@click.option(
    "--no-backends",
    is_flag=True,
    help="Whether to only use the default backend, or randomly sample from the supported backend for each domain. Defaults to True.",
)
@click.option(
    "--source-path",
    type=str,
    required=False,
    default="task_src",
    help="Where the task code should be saved after calling create_task() on the returned config.",
)
@click.option(
    "--max-attempts",
    type=int,
    required=False,
    default=10,
    help="The max number of attempts supported for sampling a task from DiscoGen. Prevents the risk of inifinite or very long loops, if probabilities are set in such a way that tasks are valid tasks are hard to sample. Defaults to 10.",
)
@click.option(
    "--seed",
    type=int,
    required=False,
    help="A random seed for reproducible task sampling. Defaults to None, in which case sampling will be non-deterministic.",
)
@click.option(
    "--config-dest",
    type=str,
    required=False,
    default="task_config.yaml",
    help="Where the config should be saved after sampling.",
)
def sample_task_config_cmd(
    p_edit: float,
    p_data: str,
    eval_type: str,
    no_backends: bool,
    source_path: str,
    max_attempts: int,
    seed: int | None,
    config_dest: str,
) -> None:
    """Create task source files for a specified task domain."""
    use_backends = not no_backends

    if not config_dest.endswith(".yaml"):
        raise click.BadParameter("config-dest must end with .yaml.", param_hint="'--config-dest'")

    if p_data.startswith("[") or p_data.endswith("]"):
        if not (p_data.startswith("[") and p_data.endswith("]")):
            raise click.BadParameter(
                "p-data must either have no square brackets, or be completely enclosed by square brackets.",
                param_hint="'--p-data'",
            )
        p_data = p_data[1:-1]

    p_data_list: list[float] = [float(x) for x in p_data.split(",")]

    task_domain, task_config = sample_task_config(
        p_edit=p_edit,
        p_data=p_data_list,
        eval_type=eval_type,
        use_backends=use_backends,
        source_path=source_path,
        max_attempts=max_attempts,
        seed=seed,
    )

    with open(config_dest, "w") as outfile:
        yaml.dump(task_config, outfile, default_flow_style=False)

    click.echo(f"Successfully saved new task_config for the {task_domain} domain at {config_dest}.")


@cli.command("create-discobench")
@click.option(
    "--task-name",
    type=click.Choice(get_discobench_tasks(), case_sensitive=False),
    required=True,
    help="The name of the discobench task to create.",
)
@click.option("--test", is_flag=True, help="If passed, create test task instead of training task.")
@click.option(
    "--use-base",
    is_flag=True,
    help="If passed, will initialise editable modules with baseline implementations instead of interface-only `edit` implementations. Has no effect with --test.",
)
@click.option(
    "--no-data",
    is_flag=True,
    help="If passed, will create the task without downloading the data. The task code will generally not be able to run, but this will allow you to see how the code looks for a specific task.",
)
@click.option(
    "--eval-type",
    type=click.Choice(["performance", "time", "energy"], case_sensitive=False),
    default="performance",
    help="What type of evaluation to use. Options are 'performance' (find the highest performance algorithm), 'time' (find the algorithm which matches baseline performance in the least time) and 'energy' (find the algorithm which matched the baseline performance using the least energy). Default: performance",
)
@click.option("--cache-root", type=str, default="cache", help="A directory to which data can be downloaded and cached.")
def create_discobench_task_cmd(
    task_name: str, test: bool, use_base: bool, no_data: bool, eval_type: str, cache_root: str
) -> None:
    """Create task source files for a specified task domain."""
    if test and use_base:
        click.echo("Warning: --use-base has no effect with --test. Test tasks use discovered files from training.")

    create_discobench(
        task_name=task_name, test=test, use_base=use_base, no_data=no_data, eval_type=eval_type, cache_root=cache_root
    )
    mode = "test" if test else "training"
    click.echo(f"Successfully created {mode} discobench task: {task_name}.")


if __name__ == "__main__":
    cli()
