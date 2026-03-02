import pathlib


def get_domains() -> list[str]:
    """Function to see all domains available in DiscoGen registration.

    Returns:
        List of [domains].
    """
    task_path = pathlib.Path(__file__).parent.parent / "domains"
    domains = [p.name for p in task_path.iterdir() if p.is_dir()]
    return domains
