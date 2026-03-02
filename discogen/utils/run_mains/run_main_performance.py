import json
import os
import subprocess
import sys
from typing import Any


def run_all_main_py(start_dir: str = ".") -> dict[str, Any]:
    """Run all main.py files in the given directory and its subdirectories.

    Args:
        start_dir: The directory to start the search for main.py files from.
    """
    results = {}
    for root, dirs, files in os.walk(start_dir):
        dirs[:] = [d for d in dirs if d != "data"]

        if "main.py" in files:
            main_path = os.path.abspath(os.path.join(root, "main.py"))
            print(f"Running: {main_path}")
            try:
                result = subprocess.run([sys.executable, main_path], check=True, capture_output=True, text=True)  # noqa: S603

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
                    results[root] = metrics
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


if __name__ == "__main__":
    run_all_main_py()
