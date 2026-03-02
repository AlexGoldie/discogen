"""A function to sample a randomly generated task config. To prevent bias, this first uniformly samples a random task domain before sampling parameters for a task config for this task domain. Uses rejection sampling to remove invalid configs."""

from pathlib import Path
from typing import Any

import numpy as np
import yaml


def sample_task_config(
    p_edit: float,
    p_data: list[float],
    eval_type: str = "random",
    use_backends: bool = True,
    source_path: str = "task_src",
    max_attempts: int = 10,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Sample a random task using user defined variables.

    Args:
        p_edit: The probability a module is marked as editable. Must be between 0. and 1.
        p_data: A list of probabilities or weights for sampling. Supports either a list of 2 values, which must be [p_meta_train, p_meta_test], or a list of 3 values, which can be probabilities or weights [w_meta_train, w_meta_test, w_exclude].
        eval_type: What eval_type to use. Supports 'random', which will select a random eval_type, or one of ['performance', 'energy', 'time']. Defaults to 'random'.
        use_backends: Whether to only use the default backend, or randomly sample from the supported backend for each domain. Defaults to True.
        source_path: Where the task code should be saved after calling create_task() on the returned config.
        max_attempts: The max number of attempts supported for sampling a task from DiscoGen. Prevents the risk of infinite or very long loops, if probabilities are set in such a way that valid tasks are hard to sample. Defaults to 10.
        rng (optional): An np random generator for deterministic stochasticity.
        seed (optional): A random seed which can be used instead of rng.

    Returns:
        random_domain: The randomly sampled domain.
        new_config: A DiscoGen configuration dictionary.

    Notes:
        At most one of seed and rng should be set. If both are set, an error will be returned.

    """
    p_data = _check_args(p_edit, p_data, eval_type, use_backends, source_path, max_attempts, rng, seed)

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng(None)

    discogen_path = Path(__file__).parent / "domains"

    task_domains = [x.name for x in discogen_path.iterdir() if x.is_dir()]

    total_edit = 0
    total_train = 0
    total_test = 0
    attempts = 0

    for _ in range(max_attempts):
        result = _generate_config(discogen_path, p_edit, p_data, use_backends, source_path, rng)
        if result:
            return result

    raise RuntimeError(
        f"The number of attempts to sample a valid config exceeded {max_attempts} attempts. Either increase max_attempts or reduce the difficulty to sample a valid config."
    )


def _generate_config(
    base_path: Path, p_edit: float, p_data: list[float], use_backends: bool, source_path: str, rng: np.random.Generator
) -> tuple[str, dict[str, Any]] | None:
    """Attempt to generate a single valid configuration. Returns None if invalid."""
    # Ensure consistent sorting for reproducibility
    task_domains = sorted([x.name for x in base_path.iterdir() if x.is_dir()])
    random_domain = str(rng.choice(task_domains))

    domain_yaml = base_path.joinpath(f"{random_domain}/task_config.yaml").read_text()
    domain_config = yaml.safe_load(domain_yaml)

    new_config: dict[str, Any] = {"train_task_id": [], "test_task_id": []}

    if not use_backends:
        new_config["template_backend"] = "default"
    else:
        templates_path = base_path.joinpath(f"{random_domain}/templates")
        backends = sorted([x.name for x in templates_path.iterdir() if x.is_dir()])
        new_config["template_backend"] = str(rng.choice(backends))

    datasets = domain_config["train_task_id"]
    # 0=Train, 1=Test, 2=Exclude
    dataset_idxes = rng.choice([0, 1, 2], size=len(datasets), p=p_data)

    for i, task_id in enumerate(datasets):
        if dataset_idxes[i] == 0:
            new_config["train_task_id"].append(task_id)
        elif dataset_idxes[i] == 1:
            new_config["test_task_id"].append(task_id)

    # Define which modules are edits
    total_edit = 0
    for k in domain_config:
        if "change_" in k:
            should_edit = rng.random() < p_edit
            new_config[k] = should_edit
            if should_edit:
                total_edit += 1

    new_config["source_path"] = source_path

    # Ensure valid config
    has_train = len(new_config["train_task_id"]) > 0
    has_test = len(new_config["test_task_id"]) > 0
    has_edits = total_edit > 0

    if has_train and has_test and has_edits:
        return random_domain, new_config

    return None


def _check_args(
    p_edit: float,
    p_data: list[float],
    eval_type: str,
    use_backends: bool,
    source_path: str,
    max_attempts: int,
    rng: np.random.Generator | None,
    seed: int | None,
) -> list[float]:
    if not (0 < p_edit <= 1):
        raise ValueError("p_edit must be between 0 and 1.")

    p_data = _normalize_p_data(p_data)

    if eval_type not in ["random", "performance", "energy", "time"]:
        raise ValueError("eval_type must be one of  ['random', 'performance', 'energy', 'time'].")

    if seed is not None and rng is not None:
        raise ValueError("When sampling a task, at most only one of seed and rng can be set.")

    return p_data


def _normalize_p_data(p_data: list[float]) -> list[float]:
    if len(p_data) not in (2, 3):
        raise ValueError("p_data must be length 2 [train, test] or 3 [train, test, exclude].")

    if any(p < 0 for p in p_data):
        raise ValueError("All p_data entries must be >= 0.")

    # Normalize logic
    if len(p_data) == 3:
        total_p = sum(p_data)
        if total_p <= 0:
            raise ValueError("Sum of p_data weights must be positive.")
        p_data = [i / total_p for i in p_data]
    else:
        remainder = 1.0 - sum(p_data)
        if remainder < -1e-9:
            raise ValueError(f"p_data sums to {sum(p_data)}, which is > 1.")
        p_data.append(max(0.0, remainder))

    if any(p >= 1.0 - 1e-9 for p in p_data):
        raise ValueError("Each entry in p_data must be < 1 to allow for a mix of train/test tasks.")

    return p_data
