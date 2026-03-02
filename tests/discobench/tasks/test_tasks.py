import os
from pathlib import Path

import pytest
import yaml

from discobench import create_config
from discobench.utils import get_domains


@pytest.mark.parametrize("domain", get_domains())
def test_task_datasets(domain: str) -> None:
    """Ensure every dataset has a description."""
    task_path = f"discobench/tasks/{domain}"
    dataset_path = f"{task_path}/datasets"

    file_list = []

    for _, (_, _, filenames) in enumerate(os.walk(dataset_path)):
        file_list.extend(filenames)
        break

    assert file_list == []  # There should only be directories here.

    task_config = create_config(domain)

    assert task_config["train_task_id"] == task_config["test_task_id"]

    found_in_datasets = set()
    task_data = set(task_config["train_task_id"])
    dataset_p = Path(dataset_path)
    for root, dirs, _ in os.walk(dataset_path, topdown=True):
        rel_path = Path(root).relative_to(dataset_p).as_posix()

        if rel_path == ".":
            continue

        if rel_path in task_data:
            found_in_datasets.add(rel_path)
            # We found a valid dataset, so we don't need to check subdirectories.
            dirs[:] = []
            assert os.path.exists(f"{root}/description.md")
        else:
            # If this is a leaf node and NOT in data_ids, it's an orphan and needs to be added to the config
            if not dirs:
                pytest.fail(f"Unauthorized leaf directory found: {rel_path}")

    # Check every dataset found is in the configs
    missing = task_data - found_in_datasets
    if missing:
        pytest.fail(f"Config IDs not found on disk: {missing}")


@pytest.mark.parametrize("domain", get_domains())
def test_task_utils(domain: str) -> None:
    """Ensure every dataset has the correct files in its setup, including that all files are copied over."""
    task_path = f"discobench/tasks/{domain}"
    utils_path = f"{task_path}/utils"

    expected_files = [
        "_reference.txt",
        "description.md",
        "requirements.txt",
        "task_information.yaml",
        "task_spec.yaml",
        "baseline_scores.yaml",
    ]

    for file in expected_files:
        assert os.path.exists(f"{utils_path}/{file}")

    with open(f"{utils_path}/task_spec.yaml") as f:
        task_spec = yaml.safe_load(f)

    assert sorted(task_spec.keys()) == ["fixed_files", "module_files"]

    assert "main.py" in task_spec["fixed_files"]
    assert list(set(task_spec["fixed_files"]) & set(task_spec["module_files"])) == []

    for file in task_spec["fixed_files"]:
        assert task_spec["fixed_files"].count(file) == 1

    for file in task_spec["module_files"]:
        assert task_spec["module_files"].count(file) == 1

    with open(f"{utils_path}/task_information.yaml") as f:
        task_information = yaml.safe_load(f)

    for file in task_spec["fixed_files"]:
        template_path = f"{task_path}/templates/default/{file}"

        if os.path.exists(template_path):
            pass
        else:
            datasets_dir = f"{task_path}/datasets"

            for dataset in create_config(domain)["train_task_id"]:
                dataset_file_path = os.path.join(datasets_dir, dataset, file)

                if not os.path.exists(dataset_file_path):
                    pytest.fail(
                        f"File {file} does not exist in either the main template, or the per-dataset templates."
                    )

    expected_information_keys = []

    for file in task_spec["module_files"]:
        base_path = f"{task_path}/templates/default/base/{file}"
        edit_path = f"{task_path}/templates/default/edit/{file}"

        assert os.path.exists(base_path)
        assert os.path.exists(edit_path)
        file_name_only = os.path.splitext(file)[0]
        expected_information_keys.append(f"{file_name_only}_prompt")

    assert sorted(task_information.keys()) == sorted(expected_information_keys)


@pytest.mark.parametrize("domain", get_domains())
def test_task_scores(domain: str) -> None:
    """Ensure every dataset and backend is reflected in baseline_scores.yaml."""
    task_path = f"discobench/tasks/{domain}"
    utils_path = f"{task_path}/utils"

    baselines_path = f"{utils_path}/baseline_scores.yaml"

    with open(baselines_path) as f:
        baseline_scores = yaml.safe_load(f)

    templates_path = f"{task_path}/templates"
    backends = sorted([x.name for x in Path(templates_path).iterdir() if x.is_dir()])
    expected_datasets = create_config(domain)["train_task_id"]

    errors = []

    for score, score_values in baseline_scores.items():
        if "objective" not in score_values:
            errors.append(f"key 'objective' not in baseline_scores for '{score}'")

    for backend in backends:
        for dataset in expected_datasets:
            dataset_found = False

            # Search through all scores to see if this backend/dataset combo exists
            for _, score_values in baseline_scores.items():
                backend_data = score_values.get(backend)

                # Check if the backend exists and is a dictionary containing the dataset
                if isinstance(backend_data, dict) and dataset in backend_data:
                    dataset_found = True
                    break  # Found it, move on to the next dataset

            if not dataset_found:
                errors.append(f"Dataset '{dataset}' is missing for backend '{backend}' across all scores")

    # Check if we collected any errors and fail the test if we did
    if errors:
        error_msg = "The following baseline configurations are missing:\n* " + "\n* ".join(errors)
        pytest.fail(error_msg)
