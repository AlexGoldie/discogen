"""Command line interface for DiscoBench."""

import os

import click
import yaml

from discobench import create_config, create_discobench, create_task, get_discobench_tasks, get_domains, get_modules


@click.group()
def cli() -> None:
    """DiscoBench - Modular Benchmark for Automated Algorithm Discovery."""
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
    help="The path to your task_config.yaml. If not provided, this will default to the DiscoBench task_config.yaml for your provided task_domain.",
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
    """List all available task domains in DiscoBench."""
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
    """List all available modules for a specified task domain."""
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
