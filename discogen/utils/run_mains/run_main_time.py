import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any

# Baselines are hand-recorded to 5-6 decimal places; tolerate that rounding error.
BASELINE_TOLERANCE = 1e-5


@dataclass
class RunResult:
    """Dataclass for storing all outputs of run_main.py."""

    results: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


def run_all_main_py(start_dir: str = ".") -> RunResult:
    """Run all main.py files in the given directory and its subdirectories.

    Args:
        start_dir: The directory to start the search for main.py files from.

    Returns:
        results: A dictionary of scores for all successful runs
        errors: A dictionary of error messages for all unsuccessful runs
    """
    results: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for root, dirs, files in os.walk(start_dir):
        dirs[:] = [d for d in dirs if d != "data"]

        if "main.py" not in files:
            continue

        main_path = os.path.abspath(os.path.join(root, "main.py"))

        baseline_path = os.path.abspath(os.path.join(root, "baseline_scores.json"))
        with open(baseline_path) as f:
            baseline_scores = json.load(f)

        print(f"Running: {main_path}")
        try:
            start = time.perf_counter()
            result = subprocess.run([sys.executable, main_path], check=True, capture_output=True, text=True)  # noqa: S603
            end = time.perf_counter()

            metrics = next(
                (
                    json.loads(line)
                    for line in reversed(result.stdout.strip().split("\n"))
                    if line.strip().startswith("{")
                ),
                None,
            )

            if metrics:
                _extract_scores(baseline_scores, metrics, root, main_path, start, end, results, errors)
            else:
                errors[root] = result.stdout
        except subprocess.CalledProcessError as e:
            error_message = e.stderr
            errors[root] = error_message
        except json.JSONDecodeError as e:
            errors[root] = f"Failed to parse metrics JSON: {e}"

    print(json.dumps({"results": results, "errors": errors}))
    return RunResult(results, errors)


def _get_nested_metric(metrics: dict[str, Any], path: str) -> float | dict[str, Any] | None:
    """Traverse a nested dictionary using a dot-separated path."""
    keys = path.split(".")
    current = metrics
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def _extract_scores(
    baseline_scores: dict[str, Any],
    metrics: dict[str, Any],
    root: str,
    main_path: str,
    start: float,
    end: float,
    results: dict[str, Any],
    errors: dict[str, str],
) -> None:
    results[root] = metrics
    results[root]["time_to_completion (s)"] = end - start
    missing_metrics = []
    for metric_name, baseline_score in baseline_scores.items():
        metric_value = _get_nested_metric(metrics, metric_name)

        if metric_value is not None and isinstance(metric_value, (int, float)):
            baseline = baseline_score[0]
            baseline_objective = baseline_score[1]

            def comp_baseline(x: float, y: float) -> bool:
                if math.isclose(x, y, rel_tol=0.0, abs_tol=BASELINE_TOLERANCE):
                    return False
                return x < y if baseline_objective == "max" else x > y

            if comp_baseline(metric_value, baseline):
                results[root][f"Exceeded Threshold For {metric_name}"] = False
            else:
                results[root][f"Exceeded Threshold For {metric_name}"] = True
        else:
            missing_metrics.append(metric_name)

    if len(missing_metrics) > 0:
        errors[root] = f"Script {main_path} did not produce any metric for {missing_metrics}."


if __name__ == "__main__":
    run_all_main_py()
