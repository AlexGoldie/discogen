import json
import math
import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Self

import psutil  # type: ignore[import-untyped]
import pynvml  # type: ignore[import-untyped]

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
            gpu_baseline = get_gpu_baseline()

            env = os.environ.copy()
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
            proc = subprocess.Popen(  # noqa: S603
                [sys.executable, main_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
            )
            with PeakMemoryTracker(proc.pid, gpu_baseline) as mem_tracker:  # pass child PID
                stdout, stderr = proc.communicate()

            _check_proc(proc, main_path, stderr)

            ram = mem_tracker.peak_ram_gb
            gpu_memory = mem_tracker.peak_gpu_gb

            metrics = next(
                (json.loads(line) for line in reversed(stdout.strip().split("\n")) if line.strip().startswith("{")),
                None,
            )

            if metrics:
                _extract_scores(baseline_scores, metrics, root, main_path, ram, gpu_memory, results, errors)
            else:
                errors[root] = stdout
        except subprocess.CalledProcessError as e:
            error_message = e.stderr
            errors[root] = error_message
        except json.JSONDecodeError as e:
            errors[root] = f"Failed to parse metrics JSON: {e}"

    print(json.dumps({"results": results, "errors": errors}))
    return RunResult(results, errors)


def get_gpu_baseline() -> int:
    """Calculates GPU memory usage before running main.py."""
    try:
        pynvml.nvmlInit()
        baseline = 0
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            baseline += pynvml.nvmlDeviceGetMemoryInfo(handle).used
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        return 0

    return baseline


def _check_proc(proc: subprocess.Popen[str], main_path: str, stderr: str) -> None:
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, main_path, stderr=stderr)


class PeakMemoryTracker:
    """A class to track the peak RAM and GPU memory usage while main.py runs."""

    def __init__(self, pid: int, gpu_baseline: int) -> None:
        """Creates PeakMemoryTracker object."""
        self._pid = pid
        self._gpu_baseline = gpu_baseline

    def __enter__(self) -> Self:
        self._stop = threading.Event()
        self._result: dict[str, int] = {}
        self._thread = threading.Thread(
            target=self._monitor,
            args=(self._pid,),
            daemon=True,  # use stored PID
        )
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()

    @property
    def peak_ram_gb(self) -> float:
        """Returns the peak RAM used in GB."""
        ram: float = self._result.get("ram", 0) / 1024 / 1024 / 1024
        return ram

    @property
    def peak_gpu_gb(self) -> float:
        """Returns the peak GPU memory used in GB."""
        gpu: float = self._result.get("gpu", 0) / 1024 / 1024 / 1024
        return gpu

    def _monitor(self, pid: int) -> None:  # add self, change return to None
        peak_ram = 0
        peak_gpu = 0

        gpu_available = False
        try:
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            gpu_available = gpu_count > 0
        except pynvml.NVMLError:
            pass

        try:
            proc = psutil.Process(pid)
            while not self._stop.is_set():  # use self._stop instead of stop_event
                try:
                    peak_ram = max(peak_ram, proc.memory_info().rss)
                except psutil.NoSuchProcess:
                    break

                if gpu_available:
                    gpu_mem = 0
                    for i in range(gpu_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        gpu_mem += mem_info.used
                    peak_gpu = max(peak_gpu, gpu_mem)

                self._stop.wait(0.01)
        finally:
            if gpu_available:
                pynvml.nvmlShutdown()

        self._result["ram"] = peak_ram  # store on self instead of returning
        self._result["gpu"] = max(0, peak_gpu - self._gpu_baseline)


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
    ram: float,
    gpu_mem: float,
    results: dict[str, Any],
    errors: dict[str, str],
) -> None:
    results[root] = metrics
    results[root]["RAM (GB)"] = ram
    if gpu_mem > 0:
        results[root]["GPU Memory (GB)"] = gpu_mem
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
