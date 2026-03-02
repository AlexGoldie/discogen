import importlib
import importlib.util
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml


class MakeFiles:
    """A class to prepare the training and test files for a task."""

    def __init__(self, task_domain: str, cache_root: str = "cache") -> None:
        """Initialize the MakeFiles class.

        Args:
            task_domain: The task domain to create the task for.
            cache_root: A cache directory to store data in.
        """
        self.base_path = Path(__file__).parent.parent / "tasks" / task_domain
        task_spec_path = self.base_path / "utils" / "task_spec.yaml"
        with open(task_spec_path) as f:
            self.task_spec = yaml.safe_load(f)
        self.template_path = self.base_path / "templates"  # will make this /default too
        self.cache_root = cache_root

    def _setup_source_directory(self, train: bool) -> None:
        """Setup the source directory by cleaning it appropriately.

        Args:
            train: Whether this is for training (complete wipe) or test (preserve discovered/).
        """
        if train:
            if self.source_path.exists():
                shutil.rmtree(self.source_path)
        else:
            for item in self.source_path.iterdir():
                if item.name != "discovered":
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()

    def _normalize_task_ids(self, config: dict[str, Any], train_test: str) -> list[str]:
        """Ensure task_id is always a list.

        Args:
            config: The task configuration.
            train_test: Either "train" or "test".

        Returns:
            List of task IDs.
        """
        task_ids = config[f"{train_test}_task_id"]
        if not isinstance(task_ids, list):
            return [task_ids]
        return task_ids

    def _normalize_model_ids(self, config: dict[str, Any], train_test: str, task_ids: list[str]) -> list[str | None]:
        """Ensure model_id is always a list matching the length of task_ids.

        Args:
            config: The task configuration.
            train_test: Either "train" or "test".
            task_ids: List of task IDs.

        Returns:
            List of model IDs matching length of task_ids.

        Raises:
            ValueError: If model_ids list length doesn't match task_ids length.
        """
        model_ids_key = f"{train_test}_model_id"

        # If no model_ids specified, return None for each task
        if model_ids_key not in config:
            return [None] * len(task_ids)

        model_ids = config[model_ids_key]

        # If single string, repeat for all tasks
        if isinstance(model_ids, str):
            return [model_ids] * len(task_ids)
        elif isinstance(model_ids, list) and len(model_ids) == 1:
            return model_ids * len(task_ids)

        # If list, validate length matches
        if isinstance(model_ids, list):
            if len(model_ids) != len(task_ids):
                raise ValueError(
                    f"Length of {model_ids_key} ({len(model_ids)}) must match "
                    f"length of {train_test}_task_id ({len(task_ids)})"
                )
            return model_ids

        raise ValueError(f"{model_ids_key} must be a string or list")

    def _build_base_description(self, template_backend: str) -> str:
        """Build the base description from discobench and domain descriptions.

        Args:
            template_backend: The template backend to use.

        Returns:
            Combined base description string.
        """
        discobench_description = self._load_discobench_description()
        domain_description = self._load_domain_description(template_backend)
        return f"{discobench_description}\n\n{domain_description}"

    def _get_eval_description(self, eval_type: str) -> str:
        """Get the pre-written evaluation type description. This explains what the objective is for optimising the algorithm.

        Args:
            eval_type: The evaluation type which the algorithm will be measured for.

        Returns:
            A description of the evaluation type.
        """
        eval_description_path = Path(__file__).parent / "eval_descriptions.yaml"
        with open(eval_description_path) as f:
            eval_descriptions: dict[str, str] = yaml.safe_load(f)

        eval_description = eval_descriptions[f"{eval_type}_description"]

        return eval_description

    def _load_model_description(self, model_path: Path) -> str:
        """Load the model description from models/{model_id}/description.md.

        Args:
            model_path: the path to the model directory.

        Returns:
            Model description string.
        """
        model_description_path = model_path / "description.md"
        if model_description_path.exists():
            return model_description_path.read_text(encoding="utf-8")
        return ""

    def _copy_model_files(self, model_path: Path, dest_loc: Path) -> None:
        """Copy files from models/{model_id}/ to the destination location.

        Args:
            model_path: The path to the model directory.
            dest_loc: The destination directory.
        """
        if not model_path.exists():
            raise ValueError("model_path does not exist")

        # Copy all files except description.md (already handled separately)
        for item in model_path.iterdir():
            if item.name == "description.md":
                continue

            dest_item = dest_loc / item.name
            if item.is_file():
                shutil.copy2(item, dest_item)
            elif item.is_dir():
                if dest_item.exists():
                    shutil.rmtree(dest_item)
                shutil.copytree(item, dest_item)

    def _process_single_task(
        self,
        task_id: str,
        model_id: str | None,
        config: dict[str, Any],
        train_test: str,
        template_backend: str,
        train: bool,
        use_base: bool,
        no_data: bool | None,
        baselines: dict[str, Any] | None,
        baseline_scale: float,
    ) -> tuple[list[str], str, str]:
        """Process a single task: create files and return discovered files and description.

        Args:
            task_id: The task identifier.
            model_id: An optional model_id, if using a pre-trained model in the codebase.
            config: The task configuration.
            train_test: Either "train" or "test".
            template_backend: The template backend to use.
            train: Whether this is for training.
            use_base: Whether to create editable modules using the baseline implementation (if True) or interface-only implementation (if False).
            no_data: Whether to create the task without loading any data files.
            baselines: A dictionary of baselines scores for this task_domain.
            baseline_scale: A tolerance/scale factor to multiply the baseline by.

        Returns:
            Tuple of (discovered_files, data_description).
        """
        self.source_path = Path(config.get("source_path", "task_src"))
        task_path = self.base_path / "datasets" / task_id
        if model_id is not None:
            model_path = self.base_path / "models" / model_id
            dest_loc = self.source_path / f"{task_id}_{model_id}"
        else:
            dest_loc = self.source_path / task_id
        dest_loc.mkdir(parents=True, exist_ok=True)

        # Get data description
        data_description = self._get_data_description(task_path)

        # Get files for pretrained model
        model_description = ""
        if model_id:
            model_description = self._load_model_description(model_path)
            # Copy model files to destination
            self._copy_model_files(model_path, dest_loc)

        # Create fixed files
        for fixed_file in self.task_spec["fixed_files"]:
            self._create_fixed(fixed_file, task_path=task_path, dest_loc=dest_loc, template_backend=template_backend)

        # Create module files and track discovered ones
        discovered_files = []
        for module_file in self.task_spec["module_files"]:
            file_no_ext = Path(module_file).stem
            change = config.get(f"change_{file_no_ext}", False)

            self._create_editable(
                module_file,
                task_path=task_path,
                dest_loc=dest_loc,
                change=change,
                template_backend=template_backend,
                train=train,
                use_base=use_base,
            )

            if change:
                discovered_files.append(module_file)

        # Ensure dataset exists
        self._ensure_dataset_cached_and_copied(task_id=task_id, task_path=task_path, dest_loc=dest_loc, no_data=no_data)

        if baselines is not None:
            self._create_baseline_scores(
                baselines, dest_loc, template_backend=template_backend, task_id=task_id, baseline_scale=baseline_scale
            )

        return discovered_files, data_description, model_description

    def _build_full_description(
        self,
        base_description: str,
        eval_description: str,
        all_discovered_files: list[str],
        data_descriptions: list[str],
        model_descriptions: list[str],
        task_information: dict[str, str],
    ) -> str:
        """Build the complete description including task information and data descriptions.

        Args:
            base_description: The base discobench + domain description.
            eval_description: A description of what the algorithm should be optimising for.
            all_discovered_files: List of all discovered files across tasks.
            data_descriptions: List of data descriptions from each task.
            model_descriptions: List of model descriptions from each task. These default to empty strings if pretrained models are not used.
            task_information: Task information dictionary.

        Returns:
            Complete description string.
        """
        full_description = base_description

        # Add task information for discovered files
        for discovered_file in all_discovered_files:
            file_no_ext = Path(discovered_file).stem
            prompt = task_information.get(f"{file_no_ext}_prompt", "")
            if prompt:
                full_description += f"\n\n{prompt}"

        # Add data and (optional) model descriptions
        for idx, (data_description, model_description) in enumerate(
            zip(data_descriptions, model_descriptions, strict=False)
        ):
            full_description += f"\n\nProblem {idx}"
            full_description += f"\n\n{data_description}"
            full_description += f"\n\n{model_description}"

        full_description += f"{eval_description}"

        return full_description

    def _create_symlinks_for_discovered(
        self, discovered_files: list[str], task_ids: list[str], model_ids: list[str | None]
    ) -> None:
        """Create symlinks from discovered files to each task directory.

        Args:
            discovered_files: List of discovered file names.
            task_ids: List of task IDs to link to.
            model_ids: List of (optional) model IDs to link to.
        """
        for discovered_file in discovered_files:
            for task_id, model_id in zip(task_ids, model_ids, strict=False):
                if model_id:
                    self._create_sym_link(discovered_file, f"{task_id}_{model_id}")
                else:
                    self._create_sym_link(discovered_file, task_id)

    def _create_fixed(self, file_name: str, task_path: Path, dest_loc: Path, template_backend: str) -> None:
        """Create a fixed file for a task.

        Args:
            file_name: The name of the file to create.
            task_path: The path to the task directory.
            dest_loc: The destination location of the file.
            template_backend: The template backend to use.
        """
        template = self._get_template(f"{file_name}", task_path, template_backend)
        dest = dest_loc / f"{file_name}"

        # Copy the template file or folder to the source directory
        if template.is_dir():
            self._copy_dir(template, dest)
        else:
            shutil.copy2(template, dest)

    def _create_editable(
        self,
        file_name: str,
        task_path: Path,
        dest_loc: Path,
        change: bool,
        template_backend: str,
        train: bool,
        use_base: bool,
    ) -> None:
        if change:
            if not train:
                return
            module_path = f"base/{file_name}" if use_base else f"edit/{file_name}"
            template = self._get_template(module_path, task_path, template_backend)
            dest = self.source_path / "discovered" / f"{file_name}"
        else:
            template = self._get_template(f"base/{file_name}", task_path, template_backend)
            dest = dest_loc / f"{file_name}"

        # Copy the template file to the source directory
        shutil.copy2(template, dest)

    def _create_baseline_scores(
        self, baselines: dict[str, Any], dest_loc: Path, template_backend: str, task_id: str, baseline_scale: float
    ) -> None:
        """Create a fixed file for a task.

        Args:
            baselines: A dictionary of baseline scores for this task domain.
            dest_loc: The destination location of the file.
            template_backend: The template backend to use.
            task_id: The specific task domain id.
            baseline_scale: The scale factor for the baseline score.
        """
        targets = {}
        for metric_name, metrics in baselines.items():
            mult_factor = -1 if metrics["objective"] == "min" else 1
            metric_backend = metrics[template_backend]
            if task_id in metric_backend:
                targets.update({metric_name: metric_backend[task_id] * mult_factor * baseline_scale})

        dict_dest = dest_loc / "baseline_scores.json"
        dict_dest.write_text(json.dumps(targets))

    def _create_sym_link(self, discovered_file: str, dest_loc: str) -> None:
        """Create sym link from discovered file to task directory."""
        master_file = self.source_path / "discovered" / discovered_file

        # Destination in task_src/task_id/
        dest_file = self.source_path / dest_loc / discovered_file

        if master_file.exists():
            # Ensure destination directory exists
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            if dest_file.exists() or dest_file.is_symlink():
                dest_file.unlink()

            # Create a relative symlink (more portable within Docker)
            relative_target = os.path.relpath(master_file, start=dest_file.parent)
            dest_file.symlink_to(relative_target)

    def _load_discobench_description(self) -> str:
        """Load the base description once during initialization."""
        discobench_path = Path(__file__).parent / "description.md"
        return discobench_path.read_text(encoding="utf-8")

    def _load_domain_description(self, template_backend: str) -> str:
        """Load the base description once during initialization."""
        template_backend_path = self.base_path / template_backend / "utils" / "description.md"
        if template_backend_path.exists():
            domain_task_path = template_backend_path
        else:
            domain_task_path = self.base_path / "utils" / "description.md"
        return domain_task_path.read_text(encoding="utf-8")

    def _load_domain_task_information(self, template_backend: str) -> dict[str, str]:
        """Load the base task information once during initialization."""
        template_backend_path = self.base_path / template_backend / "utils" / "task_information.yaml"
        if template_backend_path.exists():
            domain_task_information_path = template_backend_path
        else:
            domain_task_information_path = self.base_path / "utils" / "task_information.yaml"
        with open(domain_task_information_path) as f:
            task_info: dict[str, str] = yaml.safe_load(f)
        return task_info

    def _get_data_description(self, task_path: Path) -> str:
        """Get task-specific description and information."""
        data_task_path = task_path / "description.md"
        return data_task_path.read_text(encoding="utf-8")

    def _save_description(self, description: str) -> None:
        # Write the description to the file
        output_file = self.source_path / "description.md"
        output_file.write_text(description, encoding="utf-8")

    def _load_run_main(self, eval_type: str) -> None:
        """Load run_main.py."""
        run_main_path = Path(__file__).parent / "run_mains" / f"run_main_{eval_type}.py"

        dest = self.source_path / "run_main.py"
        shutil.copy2(run_main_path, dest)

    def _save_requirements(self, eval_type: str) -> None:
        requirements = self.base_path / "utils" / "requirements.txt"
        dest = self.source_path / "requirements.txt"

        # Copy the template file to the source directory
        shutil.copy2(requirements, dest)

        if eval_type == "energy":
            with dest.open("a", encoding="utf-8") as f:
                f.write("\ncodecarbon\n")

    def _get_template(self, file: str, task_path: Path, template_backend: str) -> Path:
        data_template = task_path / file
        if data_template.exists():
            return data_template

        template_backend_path = self.template_path / template_backend / file
        if template_backend_path.exists():
            return template_backend_path

        return self.template_path / "default" / file

    def _dir_empty(self, p: Path) -> bool:
        """Determine if the folder in the cache already has the data, or if we need to download into cache."""
        try:
            return (not p.exists()) or (p.is_dir() and not any(p.iterdir()))
        except Exception:
            return True

    def _get_download_dataset(self, task_id: str, task_path: Path) -> Callable[[Path], None] | None:
        """Try to import the optional make_dataset.download_dataset."""
        download_dataset = None
        try:
            dataset_file = task_path / "make_dataset.py"
            if not dataset_file.is_file():
                return download_dataset
            spec = importlib.util.spec_from_file_location("make_dataset", dataset_file)
            if spec is None or spec.loader is None:
                return download_dataset
            task_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(task_module)

            download_dataset = getattr(task_module, "download_dataset", None)
        except ModuleNotFoundError as e:
            # If we cannot import a downloader, do nothing for this task
            print(f"Error during make_dataset import for {task_id}, skipping dataset download. Error: {e}")
        except Exception as e:
            # On unexpected import errors, also do nothing
            print(f"Error importing make_dataset for {task_id}: {e}")

        return download_dataset

    def _ensure_dataset_cached_and_copied(
        self, task_id: str, task_path: Path, dest_loc: Path, no_data: bool | None
    ) -> None:
        """Cache dataset under ./cache/<task_id> and copy it to dest_loc/"data".

        Workflow:
        - Cache root is a project-local ./cache directory.
        - If cache for task is missing/empty and a download_dataset(task_dir) exists, populate cache.
        - Copy cache into task's data directory each run.
        """
        cache_root = Path(self.cache_root).resolve()
        cache_dir = cache_root / task_id

        download_dataset = self._get_download_dataset(task_id, task_path)

        have_data = False

        if no_data:
            return

        # Only proceed when we have a downloader; create cache only when needed
        if self._dir_empty(cache_dir) and download_dataset:
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                download_dataset(cache_dir)
            except Exception as e:
                print(f"Failed to download dataset for {task_id}: {e}")

        # If directory now exists and is non-empty, we have data
        try:
            have_data = cache_dir.exists() and any(cache_dir.iterdir())
        except Exception:
            have_data = cache_dir.exists()

        # Only touch dest data if we have something to copy
        if have_data:
            dest_data = dest_loc / "data"

            # Ensure a clean dest directory
            if dest_data.exists() and dest_data.is_dir():
                shutil.rmtree(dest_data)
            elif dest_data.exists():
                dest_data.unlink()

            self._copy_dir(cache_dir, dest_data)
        else:
            # Nothing to do for this task (e.g., RL tasks without datasets)
            return

    def _copy_dir(self, src: Path, dst: Path) -> None:
        """Recursively copy a directory tree, preserving file metadata."""
        # shutil.copytree requires dst not to exist; ensure that here
        if dst.exists():
            shutil.rmtree(dst)
        try:
            shutil.copytree(src, dst)
        except TypeError:
            # For older python versions lacking dirs_exist_ok but we already removed dst
            shutil.copytree(src, dst)

    def make_files(
        self,
        config: dict[str, Any],
        train: bool,
        use_base: bool,
        no_data: bool | None,
        eval_type: str,
        baseline_scale: float = 1.0,
    ) -> None:
        """Prepare the training and test files for a task.

        Args:
            config: The task configuration.
            train: Whether to create the training subset of the task.
            use_base: If True, will use the baseline implementation for editable modules.
            no_data: If True, will create the codebase without loading any data files.
            eval_type: What type of evaluation to use. One of ['performance', 'time', 'energy']
            baseline_scale: When using 'time' or 'energy' eval_type, what relative scaling to apply to the baseline score. Default to 1.0 (i.e., match the baseline)
        """
        self.source_path = Path(config.get("source_path", "task_src"))

        train_test = "train" if train else "test"

        # Step 1: Setup directory structure
        self._setup_source_directory(train)

        # Step 2: Normalize task IDs to list
        task_ids = self._normalize_task_ids(config, train_test)

        # Step 3: Normalize model IDs to list (or list[None] if not needed)
        model_ids = self._normalize_model_ids(config, train_test, task_ids)

        # Step 4: Get template backend and prepare discovered path
        template_backend = config.get("template_backend", "default")
        discovered_path = self.source_path / "discovered"
        discovered_path.mkdir(parents=True, exist_ok=True)

        # Step 5: Build base description
        base_description = self._build_base_description(template_backend)
        task_information = self._load_domain_task_information(template_backend)

        # Step 6: Get evaluation type description
        eval_description = self._get_eval_description(eval_type)

        # Step 7: Process each task
        data_descriptions = []
        model_descriptions = []
        all_discovered_files = set()

        if eval_type in ["time", "energy"]:
            baseline_path = self.base_path / "utils" / "baseline_scores.yaml"
            with open(baseline_path) as f:
                baselines = yaml.safe_load(f)
        else:
            baselines = None

        for task_id, model_id in zip(task_ids, model_ids, strict=False):
            single_discovered_files, data_description, model_description = self._process_single_task(
                task_id,
                model_id,
                config,
                train_test,
                template_backend,
                train,
                use_base,
                no_data,
                baselines,
                baseline_scale,
            )
            data_descriptions.append(data_description)
            model_descriptions.append(model_description)

            # discovered_files is consistent in every task, but for clarity, track in a set.
            all_discovered_files.update(single_discovered_files)
        discovered_files = list(all_discovered_files)

        # Step 7: Build and save full description
        full_description = self._build_full_description(
            base_description,
            eval_description,
            discovered_files,
            data_descriptions,
            model_descriptions,
            task_information,
        )
        self._save_description(full_description)

        # Step 9: Create symlinks for discovered files
        unique_discovered_files = discovered_files
        self._create_symlinks_for_discovered(unique_discovered_files, task_ids, model_ids)

        # Step 10: Copy run_main and requirements
        self._load_run_main(eval_type)
        self._save_requirements(eval_type)
