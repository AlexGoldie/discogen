import json
import os
import subprocess
import sys
from typing import Any

from codecarbon import EmissionsTracker  # type: ignore[import-untyped]


def run_all_main_py(start_dir: str = ".") -> dict[str, Any]:
    """Run all main.py files in the given directory and its subdirectories.

    Args:
        start_dir: The directory to start the search for main.py files from.
    """
    results: dict[str, Any] = {}
    for root, dirs, files in os.walk(start_dir):
        dirs[:] = [d for d in dirs if d != "data"]

        if "main.py" in files:
            main_path = os.path.abspath(os.path.join(root, "main.py"))

            baseline_path = os.path.abspath(os.path.join(root, "baseline_scores.json"))
            with open(baseline_path) as f:
                baseline_scores = json.load(f)

            print(f"Running: {main_path}")
            try:
                with EmissionsTracker(log_level="error", save_to_file=False) as tracker:
                    result = subprocess.run([sys.executable, main_path], check=True, capture_output=True, text=True)  # noqa: S603
                energy = tracker._total_energy.kWh

                output_lines = result.stdout.strip().split("\n")
                metrics = None

                for line in reversed(output_lines):
                    if line.strip().startswith("{"):
                        try:
                            metrics = json.loads(line)
                            break
                        except json.JSONDecodeError:
                            continue

                if metrics:
                    results[root] = _extract_scores(baseline_scores, metrics, root, main_path, results, energy)
                else:
                    raise RuntimeError(
                        f"Script {main_path} did not produce metrics.\n--- STDOUT ---\n{result.stdout}\n"
                    )
            except subprocess.CalledProcessError as e:
                print(f"Error running {main_path}: {e}")
                error_message = f"--- STDERR ---\n{e.stderr}\n"
                raise RuntimeError(error_message) from e

    print(json.dumps(results))
    return results


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
    results: dict[str, Any],
    energy: float,
) -> dict[str, Any]:
    metrics["Energy (kWh)"] = energy
    metrics["Exceeded Threshold"] = True
    for metric_name, baseline_score in baseline_scores.items():
        metric_value = _get_nested_metric(metrics, metric_name)

        if metric_value is not None:
            if metric_value < baseline_score:
                metrics["Exceeded Threshold"] = False
                break
        else:
            raise RuntimeError(f"Script {main_path} did not produce any metric for {metric_name}!\n")
    return metrics


if __name__ == "__main__":
    run_all_main_py()
