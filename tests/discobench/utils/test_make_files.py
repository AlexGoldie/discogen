"""Tests for MakeFiles class, parameterised over all domains from discobench."""

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from discobench import create_config
from discobench.utils import get_domains
from discobench.utils.make_files import MakeFiles

ALL_DOMAINS = get_domains()


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Fixtures
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


@pytest.fixture(params=ALL_DOMAINS, ids=ALL_DOMAINS)
def domain(request: pytest.FixtureRequest) -> str:
    """Return each domain from get_domains() as a parameterised fixture."""
    return str(request.param)


@pytest.fixture
def mf(domain: str) -> MakeFiles:
    """Return a real MakeFiles instance for the given domain."""
    return MakeFiles(domain)


@pytest.fixture
def config(domain: str) -> dict[str, Any]:
    """Return a valid config for the given domain."""
    return create_config(domain)


@pytest.fixture
def example_config(domain: str) -> dict[str, Any]:
    """Return the example config, with disjoint meta-train/meta-test, for the given domain."""
    conf_path = f"discobench/example_configs/{domain}.yaml"
    with open(conf_path) as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


@pytest.fixture
def source_path(tmp_path: Path) -> Path:
    """Return a clean tmp source_path directory."""
    sp = tmp_path / "task_src"
    sp.mkdir()
    return sp


@pytest.fixture
def config_with_tmp(example_config: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    """Return a config whose source_path points to tmp_path."""
    cfg = dict(example_config)
    cfg["source_path"] = str(tmp_path / "task_src")
    return cfg


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _setup_source_directory
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestSetupSourceDirectory:
    """All tests for _setup_source_directory."""

    def test_train_wipes_entire_directory(self, mf: MakeFiles, source_path: Path) -> None:
        """Test that using train=True wipes the source directory."""
        sp = source_path

        (sp / "abc.txt").write_text("123")
        (sp / "discovered").mkdir()
        (sp / "discovered" / "network.py").write_text("xyz")

        mf.source_path = sp
        mf._setup_source_directory(train=True)
        assert not sp.exists()

    def test_test_preserves_discovered(self, mf: MakeFiles, source_path: Path) -> None:
        """Test that using train=False wipes the source directory *except* the discovered/ file."""
        sp = source_path

        (sp / "abc.txt").write_text("123")
        subdir = sp / "subdir"
        subdir.mkdir()
        (subdir / "inner.txt").write_text("inner")
        (sp / "discovered").mkdir()
        (sp / "discovered" / "network.py").write_text("xyz")

        mf.source_path = sp
        mf._setup_source_directory(train=False)

        assert (sp / "discovered").exists()
        assert (sp / "discovered/network.py").read_text() == "xyz"
        assert not (sp / "abc.txt").exists()
        assert not subdir.exists()


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _normalize_task_ids
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestNormalizeTaskIds:
    """All tests for _normalize_task_ids."""

    @pytest.mark.parametrize("train_test", ["train", "test"])
    @pytest.mark.parametrize("use_list", [True, False])
    def test_task_ids(self, mf: MakeFiles, example_config: dict[str, Any], train_test: str, use_list: bool) -> None:
        """Test that task_ids are always returned as a list, whether input is a list or scalar."""
        expected_task_ids = {}
        if not use_list:
            expected_task_ids["train"] = [example_config["train_task_id"][0]]
            expected_task_ids["test"] = [example_config["test_task_id"][0]]
            example_config["train_task_id"] = example_config["train_task_id"][0]
            example_config["test_task_id"] = example_config["test_task_id"][0]
        else:
            expected_task_ids["train"] = example_config["train_task_id"]
            expected_task_ids["test"] = example_config["test_task_id"]

        task_ids = mf._normalize_task_ids(example_config, train_test)

        assert task_ids == expected_task_ids[train_test]

    def test_missing_key_raises(self, mf: MakeFiles) -> None:
        """Test that a missing task_id key raises KeyError."""
        with pytest.raises(KeyError):
            mf._normalize_task_ids({}, "train")


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _normalize_model_ids
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestNormalizeModelIds:
    """All tests for _normalize_model_ids."""

    @pytest.mark.parametrize("train_test", ["train", "test"])
    @pytest.mark.parametrize("use_list", [True, False])
    def test_no_model_ids_returns_nones(
        self, mf: MakeFiles, example_config: dict[str, Any], train_test: str, use_list: bool
    ) -> None:
        """Test that model_id defaults to [None, ...] when not set in the config."""
        if f"{train_test}_model_id" not in example_config:
            model_ids = mf._normalize_model_ids(example_config, train_test, example_config[f"{train_test}_task_id"])
            assert model_ids == [None] * len(example_config[f"{train_test}_task_id"])

    @pytest.mark.parametrize("mode", ["train", "test"])
    @pytest.mark.parametrize(
        ("task_id", "model_id", "expected_outcome"),
        [
            (["123"], ["abc"], ["abc"]),
            (["123", "456", "789"], ["abc", "def", "ghi"], ["abc", "def", "ghi"]),
            (["123", "456"], ["abc", "def", "ghi"], "Length of"),
            (["123", "456", "789"], ["abc", "def"], "Length of"),
            (["123", "456"], ["abc"], ["abc", "abc"]),
            (["123", "456"], "abc", ["abc", "abc"]),
            (["123", "456"], 123, "string or list"),
        ],
    )
    def test_different_models(
        self,
        mf: MakeFiles,
        example_config: dict[str, Any],
        task_id: list[str],
        model_id: list[str] | str | int,
        mode: str,
        expected_outcome: list[str] | str,
    ) -> None:
        """Test model_id output is always the right length, or raises when appropriate."""
        example_config[f"{mode}_task_id"] = task_id
        example_config[f"{mode}_model_id"] = model_id

        if isinstance(expected_outcome, str):
            with pytest.raises(ValueError, match=expected_outcome):
                mf._normalize_model_ids(example_config, mode, task_id)
        else:
            actual_outcome = mf._normalize_model_ids(example_config, mode, task_id)
            assert expected_outcome == actual_outcome


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _build_base_description
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestBuildBaseDescription:
    """All tests for _build_base_description."""

    def test_returns_nonempty_string(self, mf: MakeFiles, config: dict[str, Any]) -> None:
        """Test that the base description combines discobench and domain descriptions."""
        template_backend = config.get("template_backend", "default")
        result = mf._build_base_description(template_backend)
        assert isinstance(result, str)
        assert len(result) > 0


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _get_eval_description
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestGetEvalDescription:
    """All tests for _get_eval_description."""

    @pytest.mark.parametrize("eval_type", ["performance", "time", "energy"])
    def test_returns_nonempty_string(self, mf: MakeFiles, eval_type: str) -> None:
        """Test that each valid eval_type returns a non-empty description."""
        result = mf._get_eval_description(eval_type)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_invalid_eval_type_raises(self, mf: MakeFiles) -> None:
        """Test that an invalid eval_type raises KeyError."""
        with pytest.raises(KeyError):
            mf._get_eval_description("nonexistent_eval_type")

    def test_different_eval_types_return_different_descriptions(self, mf: MakeFiles) -> None:
        """Test that each eval_type produces a distinct description."""
        descriptions = {et: mf._get_eval_description(et) for et in ["performance", "time", "energy"]}
        assert len(set(descriptions.values())) == 3

    @pytest.mark.parametrize(
        ("eval_type", "expected_substring"),
        [
            ("performance", "maximise the performance of your discovered algorithms"),
            ("time", "minimise the time taken by your discovered algorithms to match a performance threshold"),
            ("energy", "minimise the energy used by your discovered algorithms to match a performance threshold"),
        ],
    )
    def test_description_contains_expected_content(
        self, mf: MakeFiles, eval_type: str, expected_substring: str
    ) -> None:
        """Test that each eval description contains domain-specific expected content."""
        result = mf._get_eval_description(eval_type)
        assert expected_substring in result, f"Expected '{expected_substring}' in {eval_type} eval description"


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _load_run_main
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestLoadRunMain:
    """All tests for _load_run_main."""

    @pytest.mark.parametrize("eval_type", ["performance", "time", "energy"])
    def test_copies_run_main(self, mf: MakeFiles, source_path: Path, eval_type: str) -> None:
        """Test that run_main.py is created in the source directory."""
        mf.source_path = source_path
        mf._load_run_main(eval_type)
        assert (source_path / "run_main.py").exists()
        assert len((source_path / "run_main.py").read_text()) > 0

    def test_invalid_eval_type_raises(self, mf: MakeFiles, source_path: Path) -> None:
        """Test that an invalid eval_type raises FileNotFoundError."""
        mf.source_path = source_path
        with pytest.raises(FileNotFoundError):
            mf._load_run_main("nonexistent_eval_type")

    def test_different_eval_types_produce_different_content(self, mf: MakeFiles, source_path: Path) -> None:
        """Test that each eval_type copies a distinct run_main.py."""
        contents = {}
        for eval_type in ["performance", "time", "energy"]:
            mf.source_path = source_path
            mf._load_run_main(eval_type)
            contents[eval_type] = (source_path / "run_main.py").read_text()
        # At least two of the three should differ
        assert len(set(contents.values())) > 1

    @pytest.mark.parametrize("eval_type", ["performance", "time", "energy"])
    def test_run_main_contains_run_all_main_py(self, mf: MakeFiles, source_path: Path, eval_type: str) -> None:
        """Test that every run_main.py defines or calls run_all_main_py."""
        mf.source_path = source_path
        mf._load_run_main(eval_type)
        content = (source_path / "run_main.py").read_text()
        assert "run_all_main_py" in content

    @pytest.mark.parametrize("eval_type", ["performance", "time", "energy"])
    def test_run_main_contains_expected_imports(self, mf: MakeFiles, source_path: Path, eval_type: str) -> None:
        """Test that every run_main.py has the expected standard imports."""
        mf.source_path = source_path
        mf._load_run_main(eval_type)
        content = (source_path / "run_main.py").read_text()
        assert "import json" in content
        assert "import os" in content
        assert "import subprocess" in content

    @pytest.mark.parametrize(
        ("eval_type", "expected_substring"),
        [
            ("performance", "baseline_path"),
            ("time", "time.perf_counter()"),
            ("energy", """EmissionsTracker(log_level="error", save_to_file=False)"""),
        ],
    )
    def test_run_main_has_eval_specific_content(
        self, mf: MakeFiles, source_path: Path, eval_type: str, expected_substring: str
    ) -> None:
        """Test that each run_main variant contains content specific to its eval type."""
        mf.source_path = source_path
        mf._load_run_main(eval_type)
        content = (source_path / "run_main.py").read_text()
        if eval_type == "performance":
            assert expected_substring not in content, (
                f"Expected '{expected_substring}' not to be in run_main_{eval_type}.py"
            )
        else:
            assert expected_substring in content, f"Expected '{expected_substring}' in run_main_{eval_type}.py"

    @pytest.mark.parametrize("eval_type", ["performance", "time", "energy"])
    def test_run_main_is_executable_python(self, mf: MakeFiles, source_path: Path, eval_type: str) -> None:
        """Test that each run_main.py can be compiled (is valid Python syntax)."""
        mf.source_path = source_path
        mf._load_run_main(eval_type)
        content = (source_path / "run_main.py").read_text()
        compile(content, f"run_main_{eval_type}.py", "exec")  # raises SyntaxError if invalid

    @pytest.mark.parametrize("eval_type", ["performance", "time", "energy"])
    def test_run_main_has_main_guard(self, mf: MakeFiles, source_path: Path, eval_type: str) -> None:
        """Test that each run_main.py has an if __name__ == '__main__' guard."""
        mf.source_path = source_path
        mf._load_run_main(eval_type)
        content = (source_path / "run_main.py").read_text()
        assert "__name__" in content and "__main__" in content

    @pytest.mark.parametrize("eval_type", ["time", "energy"])
    @pytest.mark.parametrize("baseline_score", [0.5, 1.0, 1.5])
    def test_run_main_output_keys(
        self, mf: MakeFiles, source_path: Path, eval_type: str, baseline_score: float
    ) -> None:
        """Test that run_main.py, when mocked, produces the expected metric keys.

        We patch subprocess.run so that each main.py "returns" a known JSON line,
        then verify run_all_main_py picks up the metrics dict.
        """
        mf.source_path = source_path
        mf._load_run_main(eval_type)
        content = (source_path / "run_main.py").read_text()

        # Compile and extract run_all_main_py from the copied script
        code_globals: dict[str, Any] = {}
        exec(compile(content, "run_main.py", "exec"), code_globals)  # noqa: S102

        mock_tracker_class = MagicMock()

        mock_tracker_instance = mock_tracker_class.return_value
        mock_tracker_instance.__enter__.return_value._total_energy.kWh = 123

        code_globals["EmissionsTracker"] = mock_tracker_class

        run_fn = code_globals["run_all_main_py"]

        expected_keys: dict[str, list[str]] = {
            "time": ["time_to_completion (s)", "Exceeded Threshold"],
            "energy": ["Energy (kWh)", "Exceeded Threshold"],
        }

        # Create a fake task directory with a main.py
        task_dir = source_path / "fake_task"
        task_dir.mkdir()
        # The fake main.py just prints a JSON dict — content doesn't matter because
        # we mock subprocess.run below.
        (task_dir / "main.py").write_text("print(json.dumps({'dummy_metric':1.0}))")
        (task_dir / "baseline_scores.json").write_text(json.dumps({"dummy_metric": baseline_score}))

        # Build a mock metrics dict with all expected keys
        mock_metrics = {"dummy_metric": 1.0}
        mock_stdout = json.dumps(mock_metrics)

        mock_result = type("CompletedProcess", (), {"stdout": mock_stdout + "\n", "stderr": "", "returncode": 0})()

        with patch("subprocess.run", return_value=mock_result):
            results = run_fn(start_dir=str(source_path))

        assert len(results) > 0, "run_all_main_py returned no results"
        for _dir, metrics in results.items():
            for key in expected_keys[eval_type]:
                assert key in metrics, f"Expected key '{key}' missing from {eval_type} metrics output"
                if baseline_score > 1:
                    assert not metrics["Exceeded Threshold"]
                else:
                    assert metrics["Exceeded Threshold"]


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _load_model_description
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestLoadModelDescription:
    """All tests for _load_model_description."""

    def test_returns_description_when_exists(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that the description is returned when description.md exists."""
        model_path = tmp_path / "model_x"
        model_path.mkdir()
        (model_path / "description.md").write_text("Model X description")
        assert mf._load_model_description(model_path) == "Model X description"

    def test_returns_empty_when_missing(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that an empty string is returned when description.md is missing."""
        model_path = tmp_path / "model_y"
        model_path.mkdir()
        assert mf._load_model_description(model_path) == ""


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _copy_model_files
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestCopyModelFiles:
    """All tests for _copy_model_files."""

    def test_copies_files_excluding_description(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that all files except description.md are copied to the destination."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        (model_path / "description.md").write_text("desc")
        (model_path / "weights.bin").write_bytes(b"\x00\x01")
        (model_path / "config.json").write_text("{}")

        dest = tmp_path / "dest"
        dest.mkdir()

        mf._copy_model_files(model_path, dest)
        assert (dest / "weights.bin").exists()
        assert (dest / "config.json").exists()
        assert not (dest / "description.md").exists()

    def test_copies_subdirectories(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that subdirectories within the model path are recursively copied."""
        model_path = tmp_path / "model"
        sub = model_path / "layers"
        sub.mkdir(parents=True)
        (sub / "layer1.py").write_text("code")
        (model_path / "description.md").write_text("desc")

        dest = tmp_path / "dest"
        dest.mkdir()

        mf._copy_model_files(model_path, dest)
        assert (dest / "layers" / "layer1.py").read_text() == "code"

    def test_raises_if_path_missing(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that ValueError is raised when the model path does not exist."""
        with pytest.raises(ValueError, match="model_path does not exist"):
            mf._copy_model_files(tmp_path / "nonexistent", tmp_path)


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _create_baseline_scores
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestCreateBaselineScores:
    """All tests for _create_baseline_scores."""

    def test_writes_valid_json(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that baseline_scores.json is written with correct content."""
        dest = tmp_path / "dest"
        dest.mkdir()
        baselines = {"return_mean": {"objective": "min", "default": {"task_a": 2.0}}}
        mf._create_baseline_scores(baselines, dest, template_backend="default", task_id="task_a", baseline_scale=1.0)
        scores_path = dest / "baseline_scores.json"
        assert scores_path.exists()
        scores = json.loads(scores_path.read_text())
        # min objective flips sign: 2.0 * -1 * 1.0 = -2.0
        assert scores["return_mean"] == -2.0

    def test_max_objective_preserves_sign(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that max objective keeps positive sign."""
        dest = tmp_path / "dest"
        dest.mkdir()
        baselines = {"accuracy": {"objective": "max", "default": {"task_a": 0.95}}}
        mf._create_baseline_scores(baselines, dest, template_backend="default", task_id="task_a", baseline_scale=1.0)
        scores = json.loads((dest / "baseline_scores.json").read_text())
        # max objective: 0.95 * 1 * 1.0 = 0.95
        assert scores["accuracy"] == pytest.approx(0.95)

    def test_baseline_scale_applied(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that baseline_scale multiplies the score."""
        dest = tmp_path / "dest"
        dest.mkdir()
        baselines = {"return_mean": {"objective": "min", "default": {"task_a": 2.0}}}
        mf._create_baseline_scores(baselines, dest, template_backend="default", task_id="task_a", baseline_scale=0.5)
        scores = json.loads((dest / "baseline_scores.json").read_text())
        # 2.0 * -1 * 0.5 = -1.0
        assert scores["return_mean"] == pytest.approx(-1.0)

    def test_missing_task_id_produces_empty_targets(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that a task_id absent from the baseline dict writes an empty JSON object."""
        dest = tmp_path / "dest"
        dest.mkdir()
        baselines = {"return_mean": {"objective": "min", "default": {"other_task": 5.0}}}
        mf._create_baseline_scores(
            baselines, dest, template_backend="default", task_id="missing_task", baseline_scale=1.0
        )
        scores = json.loads((dest / "baseline_scores.json").read_text())
        assert scores == {}

    def test_multiple_metrics(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that multiple metrics are all written to the JSON file."""
        dest = tmp_path / "dest"
        dest.mkdir()
        baselines = {
            "metric_a": {"objective": "min", "default": {"task_a": 10.0}},
            "metric_b": {"objective": "max", "default": {"task_a": 3.0}},
        }
        mf._create_baseline_scores(baselines, dest, template_backend="default", task_id="task_a", baseline_scale=2.0)
        scores = json.loads((dest / "baseline_scores.json").read_text())
        # metric_a: 10.0 * -1 * 2.0 = -20.0
        assert scores["metric_a"] == pytest.approx(-20.0)
        # metric_b: 3.0 * 1 * 2.0 = 6.0
        assert scores["metric_b"] == pytest.approx(6.0)

    def test_uses_template_backend_key(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that the correct template_backend key is used to look up scores."""
        dest = tmp_path / "dest"
        dest.mkdir()
        baselines = {"loss": {"objective": "min", "default": {"task_a": 99.0}, "custom_backend": {"task_a": 1.5}}}
        mf._create_baseline_scores(
            baselines, dest, template_backend="custom_backend", task_id="task_a", baseline_scale=1.0
        )
        scores = json.loads((dest / "baseline_scores.json").read_text())
        # Should use custom_backend value, not default: 1.5 * -1 * 1.0 = -1.5
        assert scores["loss"] == pytest.approx(-1.5)


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _process_single_task
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestProcessSingleTask:
    """All tests for _process_single_task."""

    def test_processes_first_train_task(self, mf: MakeFiles, config: dict[str, Any], tmp_path: Path) -> None:
        """Test that processing a single task creates fixed files and returns descriptions."""
        task_ids = mf._normalize_task_ids(config, "train")
        model_ids = mf._normalize_model_ids(config, "train", task_ids)
        template_backend = config.get("template_backend", "default")
        sp = tmp_path / "task_src"
        (sp / "discovered").mkdir(parents=True, exist_ok=True)
        cfg = dict(config)
        cfg["source_path"] = str(sp)

        discovered, data_desc, model_desc = mf._process_single_task(
            task_id=task_ids[0],
            model_id=model_ids[0],
            config=cfg,
            train_test="train",
            template_backend=template_backend,
            train=True,
            use_base=True,
            no_data=True,
            baselines={"return_mean": {"objective": "min", template_backend: {task_ids[0]: 1}}},
            baseline_scale=1.0,
        )

        assert isinstance(discovered, list)
        assert isinstance(data_desc, str)
        assert len(data_desc) > 0
        assert isinstance(model_desc, str)

        dest = sp / f"{task_ids[0]}_{model_ids[0]}" if model_ids[0] is not None else sp / task_ids[0]

        for fixed_file in mf.task_spec["fixed_files"]:
            assert (dest / fixed_file).exists(), f"Fixed file '{fixed_file}' missing from {dest}"


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _build_full_description
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestBuildFullDescription:
    """All tests for _build_full_description."""

    def test_includes_all_parts(self, mf: MakeFiles) -> None:
        """Test that the full description includes base, prompts, data, and model descriptions."""
        result = mf._build_full_description(
            base_description="Base.",
            all_discovered_files=["model.py"],
            data_descriptions=["Data desc 0"],
            model_descriptions=["Model desc 0"],
            task_information={"model_prompt": "Improve it."},
            eval_description="Eval desc 0",
        )
        assert "Base." in result
        assert "Improve it." in result
        assert "Data desc 0" in result
        assert "Model desc 0" in result
        assert "Problem 0" in result
        assert "Eval desc 0" in result

    def test_no_prompt_for_unknown_file(self, mf: MakeFiles) -> None:
        """Test that a discovered file without a matching prompt key adds nothing."""
        result = mf._build_full_description(
            base_description="Base.",
            all_discovered_files=["unknown.py"],
            data_descriptions=["Data"],
            model_descriptions=[""],
            task_information={"model_prompt": "Only for model."},
            eval_description="",
        )
        assert "Only for model." not in result

    def test_multiple_data_descriptions(self, mf: MakeFiles) -> None:
        """Test that multiple tasks produce numbered Problem sections."""
        result = mf._build_full_description(
            base_description="Base.",
            all_discovered_files=[],
            data_descriptions=["Desc A", "Desc B"],
            model_descriptions=["", ""],
            task_information={},
            eval_description="",
        )
        assert "Problem 0" in result
        assert "Problem 1" in result
        assert "Desc A" in result
        assert "Desc B" in result

    def test_eval_description_appears_in_output(self, mf: MakeFiles) -> None:
        """Test that a non-empty eval_description is included in the full description."""
        eval_text = "You must optimise for wall-clock time."
        result = mf._build_full_description(
            base_description="Base.",
            all_discovered_files=[],
            data_descriptions=["Data"],
            model_descriptions=[""],
            task_information={},
            eval_description=eval_text,
        )
        assert eval_text in result

    def test_real_eval_descriptions_appear(self, mf: MakeFiles) -> None:
        """Test that real eval descriptions from _get_eval_description appear in full description."""
        for eval_type in ["performance", "time", "energy"]:
            eval_desc = mf._get_eval_description(eval_type)
            result = mf._build_full_description(
                base_description="Base.",
                all_discovered_files=[],
                data_descriptions=["Data"],
                model_descriptions=[""],
                task_information={},
                eval_description=eval_desc,
            )
            assert eval_desc in result


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _create_symlinks_for_discovered
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestCreateSymlinksForDiscovered:
    """All tests for _create_symlinks_for_discovered."""

    def test_creates_links_without_model_ids(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that symlinks are created for each task when no model_ids are used."""
        mf.source_path = tmp_path / "src"
        disc = mf.source_path / "discovered"
        disc.mkdir(parents=True)
        (disc / "model.py").write_text("code")
        (mf.source_path / "t1").mkdir()
        (mf.source_path / "t2").mkdir()

        mf._create_symlinks_for_discovered(["model.py"], ["t1", "t2"], [None, None])
        assert (mf.source_path / "t1" / "model.py").is_symlink()
        assert (mf.source_path / "t2" / "model.py").is_symlink()

    def test_creates_links_with_model_ids(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that symlinks use {task_id}_{model_id} directory naming."""
        mf.source_path = tmp_path / "src"
        disc = mf.source_path / "discovered"
        disc.mkdir(parents=True)
        (disc / "model.py").write_text("code")

        mf._create_symlinks_for_discovered(["model.py"], ["t1"], ["m1"])
        assert (mf.source_path / "t1_m1" / "model.py").is_symlink()


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _create_fixed
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestCreateFixed:
    """All tests for _create_fixed."""

    def test_copies_real_fixed_files(self, mf: MakeFiles, config: dict[str, Any], tmp_path: Path) -> None:
        """Test that all fixed files from the task_spec are created in the destination."""
        task_ids = mf._normalize_task_ids(config, "train")
        template_backend = config.get("template_backend", "default")
        task_path = mf.base_path / "datasets" / task_ids[0]
        dest = tmp_path / "dest"
        dest.mkdir()

        for fixed_file in mf.task_spec["fixed_files"]:
            mf._create_fixed(fixed_file, task_path, dest, template_backend)
            created = dest / fixed_file
            assert created.exists(), f"Fixed file '{fixed_file}' was not created."


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _create_editable
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestCreateEditable:
    """All tests for _create_editable."""

    def test_no_change_copies_base_to_dest(self, mf: MakeFiles, config: dict[str, Any], tmp_path: Path) -> None:
        """Test that change=False copies the base template into the task destination."""
        task_ids = mf._normalize_task_ids(config, "train")
        template_backend = config.get("template_backend", "default")
        task_path = mf.base_path / "datasets" / task_ids[0]
        dest = tmp_path / "dest"
        dest.mkdir()
        mf.source_path = tmp_path / "src"
        mf.source_path.mkdir()

        for module_file in mf.task_spec["module_files"]:
            mf._create_editable(
                module_file, task_path, dest, change=False, template_backend=template_backend, train=True, use_base=True
            )
            assert (dest / module_file).exists()

    def test_change_train_copies_to_discovered(self, mf: MakeFiles, config: dict[str, Any], tmp_path: Path) -> None:
        """Test that change=True during training copies the template into discovered/."""
        task_ids = mf._normalize_task_ids(config, "train")
        template_backend = config.get("template_backend", "default")
        task_path = mf.base_path / "datasets" / task_ids[0]
        dest = tmp_path / "dest"
        dest.mkdir()
        mf.source_path = tmp_path / "src"
        discovered = mf.source_path / "discovered"
        discovered.mkdir(parents=True)

        for module_file in mf.task_spec["module_files"]:
            mf._create_editable(
                module_file, task_path, dest, change=True, template_backend=template_backend, train=True, use_base=True
            )
            assert (discovered / module_file).exists()

    def test_change_test_skips(self, mf: MakeFiles, config: dict[str, Any], tmp_path: Path) -> None:
        """Test that change=True during test does not create any files."""
        task_ids = mf._normalize_task_ids(config, "train")
        template_backend = config.get("template_backend", "default")
        task_path = mf.base_path / "datasets" / task_ids[0]
        dest = tmp_path / "dest"
        dest.mkdir()
        mf.source_path = tmp_path / "src"
        discovered = mf.source_path / "discovered"
        discovered.mkdir(parents=True)

        for module_file in mf.task_spec["module_files"]:
            mf._create_editable(
                module_file, task_path, dest, change=True, template_backend=template_backend, train=False, use_base=True
            )
            assert not (discovered / module_file).exists()
            assert not (dest / module_file).exists()


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _create_sym_link
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestCreateSymLink:
    """All tests for _create_sym_link."""

    def test_creates_relative_symlink(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that a relative symlink is created pointing from task dir to discovered/."""
        mf.source_path = tmp_path / "src"
        discovered = mf.source_path / "discovered"
        discovered.mkdir(parents=True)
        (discovered / "model.py").write_text("code")

        task_dir = mf.source_path / "task_a"
        task_dir.mkdir(parents=True)

        mf._create_sym_link("model.py", "task_a")

        link = task_dir / "model.py"
        assert link.is_symlink()
        assert link.resolve() == (discovered / "model.py").resolve()
        assert link.read_text() == "code"

    def test_replaces_existing_file(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that an existing file at the destination is replaced by a symlink."""
        mf.source_path = tmp_path / "src"
        discovered = mf.source_path / "discovered"
        discovered.mkdir(parents=True)
        (discovered / "model.py").write_text("new code")

        task_dir = mf.source_path / "task_a"
        task_dir.mkdir(parents=True)
        (task_dir / "model.py").write_text("old")

        mf._create_sym_link("model.py", "task_a")

        link = task_dir / "model.py"
        assert link.is_symlink()
        assert link.read_text() == "new code"

    def test_no_master_file_does_nothing(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that no symlink is created when the master file does not exist."""
        mf.source_path = tmp_path / "src"
        discovered = mf.source_path / "discovered"
        discovered.mkdir(parents=True)

        task_dir = mf.source_path / "task_a"
        task_dir.mkdir(parents=True)

        mf._create_sym_link("model.py", "task_a")
        assert not (task_dir / "model.py").exists()


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _load_domain_description
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestLoadDomainDescription:
    """All tests for _load_domain_description."""

    def test_default_description_exists(self, mf: MakeFiles) -> None:
        """Test that the default domain description loads and is non-empty."""
        result = mf._load_domain_description("default")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_falls_back_to_utils_for_nonexistent_backend(self, mf: MakeFiles) -> None:
        """Test that a nonexistent backend falls back to utils/description.md."""
        result = mf._load_domain_description("nonexistent_backend_xyz")
        expected = (mf.base_path / "utils" / "description.md").read_text(encoding="utf-8")
        assert result == expected


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _load_domain_task_information
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestLoadDomainTaskInformation:
    """All tests for _load_domain_task_information."""

    def test_loads_task_information(self, mf: MakeFiles, config: dict[str, Any]) -> None:
        """Test that task information loads as a non-empty dict."""
        template_backend = config.get("template_backend", "default")
        result = mf._load_domain_task_information(template_backend)
        assert isinstance(result, dict)

    def test_falls_back_to_utils_for_nonexistent_backend(self, mf: MakeFiles) -> None:
        """Test that a nonexistent backend falls back to utils/task_information.yaml."""
        result = mf._load_domain_task_information("nonexistent_backend_xyz")
        expected_path = mf.base_path / "utils" / "task_information.yaml"
        with open(expected_path) as f:
            expected = yaml.safe_load(f)
        assert result == expected


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _get_data_description
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestGetDataDescription:
    """All tests for _get_data_description."""

    def test_real_datasets_have_descriptions(self, mf: MakeFiles, config: dict[str, Any]) -> None:
        """Test that every train dataset has a non-empty description.md."""
        task_ids = mf._normalize_task_ids(config, "train")
        for task_id in task_ids:
            task_path = mf.base_path / "datasets" / task_id
            result = mf._get_data_description(task_path)
            assert isinstance(result, str)
            assert len(result) > 0


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _save_description
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestSaveDescription:
    """All tests for _save_description."""

    def test_writes_file(self, mf: MakeFiles, source_path: Path) -> None:
        """Test that _save_description writes the description to description.md."""
        mf.source_path = source_path
        mf._save_description("Hello world")
        assert (mf.source_path / "description.md").read_text() == "Hello world"


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _save_requirements
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestSaveRequirements:
    """All tests for _save_requirements."""

    @pytest.mark.parametrize("eval_type", ["time", "energy", "performance"])
    def test_copies_requirements(self, mf: MakeFiles, source_path: Path, eval_type: str) -> None:
        """Test that requirements.txt is copied to the source directory."""
        mf.source_path = source_path
        mf._save_requirements(eval_type)
        result = (mf.source_path / "requirements.txt").read_text()
        assert len(result) > 0
        if eval_type == "energy":
            assert "codecarbon" in result

    @pytest.mark.parametrize("eval_type", ["performance", "time"])
    def test_no_codecarbon_for_non_energy(self, mf: MakeFiles, source_path: Path, eval_type: str) -> None:
        """Test that codecarbon is NOT included for non-energy eval types."""
        mf.source_path = source_path
        mf._save_requirements(eval_type)
        result = (mf.source_path / "requirements.txt").read_text()
        assert "codecarbon" not in result


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _get_template
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestGetTemplate:
    """All tests for _get_template."""

    def test_dataset_override_takes_priority(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that a file in the dataset directory is preferred over templates."""
        task_path = tmp_path / "datasets" / "task_a"
        task_path.mkdir(parents=True)
        (task_path / "override.txt").write_text("# dataset override")

        result = mf._get_template("override.txt", task_path, "default")
        assert result == task_path / "override.txt"

    def test_default_fallback_returns_path(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that a nonexistent backend falls back to the default template."""
        task_path = tmp_path / "datasets" / "task_a"
        task_path.mkdir(parents=True)

        result = mf._get_template("some_file.txt", task_path, "nonexistent_backend")
        assert result == mf.template_path / "default" / "some_file.txt"

    def test_real_fixed_files_resolve(self, mf: MakeFiles, config: dict[str, Any]) -> None:
        """Test that every fixed_file in task_spec resolves to an existing template."""
        task_ids = mf._normalize_task_ids(config, "train")
        template_backend = config.get("template_backend", "default")
        task_path = mf.base_path / "datasets" / task_ids[0]

        for fixed_file in mf.task_spec["fixed_files"]:
            result = mf._get_template(fixed_file, task_path, template_backend)
            assert result.exists(), (
                f"Template for fixed file '{fixed_file}' not found. "
                f"Checked: {task_path / fixed_file}, "
                f"{mf.template_path / template_backend / fixed_file}, "
                f"{mf.template_path / 'default' / fixed_file}"
            )

    def test_real_module_base_files_resolve(self, mf: MakeFiles, config: dict[str, Any]) -> None:
        """Test that every module_file base/ template resolves to an existing file."""
        task_ids = mf._normalize_task_ids(config, "train")
        template_backend = config.get("template_backend", "default")
        task_path = mf.base_path / "datasets" / task_ids[0]

        for module_file in mf.task_spec["module_files"]:
            result = mf._get_template(f"base/{module_file}", task_path, template_backend)
            assert result.exists(), f"Base template for module '{module_file}' not found."


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _dir_empty
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestDirEmpty:
    """All tests for _dir_empty."""

    def test_nonexistent_returns_true(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that a nonexistent path is considered empty."""
        assert mf._dir_empty(tmp_path / "nope") is True

    def test_empty_dir_returns_true(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that an empty directory is considered empty."""
        d = tmp_path / "empty"
        d.mkdir()
        assert mf._dir_empty(d) is True

    def test_nonempty_dir_returns_false(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that a directory with contents is not considered empty."""
        d = tmp_path / "has_stuff"
        d.mkdir()
        (d / "file.txt").write_text("x")
        assert mf._dir_empty(d) is False


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _get_download_dataset
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestGetDownloadDataset:
    """All tests for _get_download_dataset."""

    def test_no_make_dataset_file_returns_none(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that None is returned when make_dataset.py does not exist."""
        task_path = tmp_path / "datasets" / "t1"
        task_path.mkdir(parents=True)
        assert mf._get_download_dataset("t1", task_path) is None

    def test_make_dataset_without_function_returns_none(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that None is returned when make_dataset.py lacks download_dataset."""
        task_path = tmp_path / "datasets" / "t1"
        task_path.mkdir(parents=True)
        (task_path / "make_dataset.py").write_text("# no download_dataset function\nx = 1\n")
        assert mf._get_download_dataset("t1", task_path) is None

    def test_make_dataset_with_function_returns_callable(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that a callable is returned when make_dataset.py defines download_dataset."""
        task_path = tmp_path / "datasets" / "t1"
        task_path.mkdir(parents=True)
        (task_path / "make_dataset.py").write_text(
            "from pathlib import Path\ndef download_dataset(path):\n    Path(path / 'data.txt').write_text('hello')\n"
        )
        fn = mf._get_download_dataset("t1", task_path)
        assert callable(fn)

    def test_import_error_returns_none(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that None is returned when make_dataset.py has an import error."""
        task_path = tmp_path / "datasets" / "t1"
        task_path.mkdir(parents=True)
        (task_path / "make_dataset.py").write_text("import nonexistent_module_xyz\n")
        assert mf._get_download_dataset("t1", task_path) is None


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _ensure_dataset_cached_and_copied
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestEnsureDatasetCachedAndCopied:
    """All tests for _ensure_dataset_cached_and_copied."""

    def test_no_data_flag_skips(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that no_data=True skips dataset creation entirely."""
        task_path = tmp_path / "datasets" / "t1"
        task_path.mkdir(parents=True)
        dest = tmp_path / "dest"
        dest.mkdir()

        mf._ensure_dataset_cached_and_copied("t1", task_path, dest, no_data=True)
        assert not (dest / "data").exists()


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _copy_dir
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestCopyDir:
    """All tests for _copy_dir."""

    def test_copies_tree(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that a directory tree is recursively copied."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("a")
        sub = src / "sub"
        sub.mkdir()
        (sub / "b.txt").write_text("b")

        dst = tmp_path / "dst"
        mf._copy_dir(src, dst)

        assert (dst / "a.txt").read_text() == "a"
        assert (dst / "sub" / "b.txt").read_text() == "b"

    def test_overwrites_existing_dest(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that an existing destination is completely replaced."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "new.txt").write_text("new")

        dst = tmp_path / "dst"
        dst.mkdir()
        (dst / "old.txt").write_text("old")

        mf._copy_dir(src, dst)
        assert (dst / "new.txt").read_text() == "new"
        assert not (dst / "old.txt").exists()


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# End-to-end: make_files for train and test
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestMakeFilesEndToEnd:
    """End-to-end tests for the make_files entry point."""

    def test_make_files_train(self, mf: MakeFiles, config_with_tmp: dict[str, Any]) -> None:
        """Test that a full train run completes and produces all expected outputs."""
        mf.make_files(config_with_tmp, train=True, use_base=True, no_data=True, eval_type="performance")

        sp = Path(config_with_tmp["source_path"])
        assert sp.exists()
        assert (sp / "description.md").exists()
        assert (sp / "requirements.txt").exists()
        assert (sp / "run_main.py").exists()
        assert (sp / "discovered").is_dir()

        task_ids = mf._normalize_task_ids(config_with_tmp, "train")
        model_ids = mf._normalize_model_ids(config_with_tmp, "train", task_ids)
        for task_id, model_id in zip(task_ids, model_ids, strict=False):
            dest = sp / f"{task_id}_{model_id}" if model_id else sp / task_id
            assert dest.exists(), f"Task directory '{dest}' missing after train make_files."

        for train_task_id in config_with_tmp["train_task_id"]:
            train_dir = train_task_id
            if "train_model_id" in config_with_tmp:
                train_dir = train_dir + f"_{config_with_tmp['train_model_id'][0]}"

            assert (sp / train_dir).is_dir()

        for test_task_id in config_with_tmp["test_task_id"]:
            test_dir = test_task_id
            if "test_model_id" in config_with_tmp:
                test_dir = test_dir + f"_{config_with_tmp['test_model_id'][0]}"

            assert not (sp / test_dir).is_dir()

        for k, v in config_with_tmp.items():
            if k.startswith("change_"):
                if v:
                    assert (sp / "discovered" / f"{k[7:]}.py").exists()
                else:
                    assert not (sp / "discovered" / f"{k[7:]}.py").exists()

    def test_make_files_test_after_train(self, mf: MakeFiles, config_with_tmp: dict[str, Any]) -> None:
        """Test that a test run after train completes and preserves discovered/."""
        mf.make_files(config_with_tmp, train=True, use_base=True, no_data=True, eval_type="performance")

        sp = Path(config_with_tmp["source_path"])
        for k, v in config_with_tmp.items():
            if k.startswith("change_") and v:
                (sp / "discovered" / f"{k[7:]}.py").write_text("abc123")

        mf.make_files(config_with_tmp, train=False, use_base=True, no_data=True, eval_type="performance")

        for k, v in config_with_tmp.items():
            if k.startswith("change_") and v:
                assert (sp / "discovered" / f"{k[7:]}.py").read_text() == "abc123"

    def test_make_files_test_discovered_preserved(self, mf: MakeFiles, config_with_tmp: dict[str, Any]) -> None:
        """Test that any files written after train in discovered/ are preserved."""
        mf.make_files(config_with_tmp, train=True, use_base=True, no_data=True, eval_type="performance")
        mf.make_files(config_with_tmp, train=False, use_base=True, no_data=True, eval_type="performance")

        sp = Path(config_with_tmp["source_path"])
        assert sp.exists()
        assert (sp / "description.md").exists()
        assert (sp / "discovered").is_dir()

        for train_task_id in config_with_tmp["train_task_id"]:
            train_dir = train_task_id
            if "train_model_id" in config_with_tmp:
                train_dir = train_dir + f"_{config_with_tmp['train_model_id'][0]}"

            assert not (sp / train_dir).is_dir()

        for test_task_id in config_with_tmp["test_task_id"]:
            test_dir = test_task_id
            if "test_model_id" in config_with_tmp:
                test_dir = test_dir + f"_{config_with_tmp['test_model_id'][0]}"

            assert (sp / test_dir).is_dir()

    def test_make_files_train_use_base_false(self, mf: MakeFiles, config_with_tmp: dict[str, Any]) -> None:
        """Test that train with use_base=False (edit templates) completes without error."""
        mf.make_files(config_with_tmp, train=True, use_base=False, no_data=True, eval_type="performance")

        sp = Path(config_with_tmp["source_path"])
        assert sp.exists()
        assert (sp / "description.md").exists()

    def test_make_files_twice_train(self, mf: MakeFiles, config_with_tmp: dict[str, Any]) -> None:
        """Test that running train twice produces an identical description."""
        mf.make_files(config_with_tmp, train=True, use_base=True, no_data=True, eval_type="performance")
        sp = Path(config_with_tmp["source_path"])
        first_desc = (sp / "description.md").read_text()

        mf.make_files(config_with_tmp, train=True, use_base=True, no_data=True, eval_type="performance")
        second_desc = (sp / "description.md").read_text()

        assert first_desc == second_desc

    def test_description_is_nonempty(self, mf: MakeFiles, config_with_tmp: dict[str, Any]) -> None:
        """Test that the generated description has substantial content."""
        mf.make_files(config_with_tmp, train=True, use_base=True, no_data=True, eval_type="performance")
        sp = Path(config_with_tmp["source_path"])
        desc = (sp / "description.md").read_text()
        assert len(desc) > 50

    def test_symlinks_resolve(self, mf: MakeFiles, config_with_tmp: dict[str, Any]) -> None:
        """Test that all symlinks in task directories resolve to existing files."""
        mf.make_files(config_with_tmp, train=True, use_base=True, no_data=True, eval_type="performance")

        sp = Path(config_with_tmp["source_path"])
        task_ids = mf._normalize_task_ids(config_with_tmp, "train")
        model_ids = mf._normalize_model_ids(config_with_tmp, "train", task_ids)

        for task_id, model_id in zip(task_ids, model_ids, strict=False):
            dest = sp / f"{task_id}_{model_id}" if model_id else sp / task_id
            if not dest.exists():
                continue
            for item in dest.iterdir():
                if item.is_symlink():
                    assert item.resolve().exists(), f"Broken symlink: {item} -> {os.readlink(item)}"

    # --- New eval_type end-to-end tests ---

    @pytest.mark.parametrize("eval_type", ["time", "energy"])
    def test_make_files_train_creates_baseline_scores(
        self, mf: MakeFiles, config_with_tmp: dict[str, Any], eval_type: str
    ) -> None:
        """Test that time/energy eval types create baseline_scores.json in each task directory."""
        mf.make_files(config_with_tmp, train=True, use_base=True, no_data=True, eval_type=eval_type)

        sp = Path(config_with_tmp["source_path"])
        task_ids = mf._normalize_task_ids(config_with_tmp, "train")
        model_ids = mf._normalize_model_ids(config_with_tmp, "train", task_ids)

        for task_id, model_id in zip(task_ids, model_ids, strict=False):
            dest = sp / f"{task_id}_{model_id}" if model_id else sp / task_id
            scores_file = dest / "baseline_scores.json"
            assert scores_file.exists(), f"baseline_scores.json missing from {dest} for eval_type={eval_type}"
            scores = json.loads(scores_file.read_text())
            assert isinstance(scores, dict)

    def test_make_files_performance_no_baseline_scores(self, mf: MakeFiles, config_with_tmp: dict[str, Any]) -> None:
        """Test that performance eval type does NOT create baseline_scores.json."""
        mf.make_files(config_with_tmp, train=True, use_base=True, no_data=True, eval_type="performance")

        sp = Path(config_with_tmp["source_path"])
        task_ids = mf._normalize_task_ids(config_with_tmp, "train")
        model_ids = mf._normalize_model_ids(config_with_tmp, "train", task_ids)

        for task_id, model_id in zip(task_ids, model_ids, strict=False):
            dest = sp / f"{task_id}_{model_id}" if model_id else sp / task_id
            assert not (dest / "baseline_scores.json").exists()

    @pytest.mark.parametrize("eval_type", ["performance", "time", "energy"])
    def test_make_files_train_all_eval_types(
        self, mf: MakeFiles, config_with_tmp: dict[str, Any], eval_type: str
    ) -> None:
        """Test that train completes successfully for all eval types and produces core outputs."""
        mf.make_files(config_with_tmp, train=True, use_base=True, no_data=True, eval_type=eval_type)

        sp = Path(config_with_tmp["source_path"])
        assert sp.exists()
        assert (sp / "description.md").exists()
        assert (sp / "requirements.txt").exists()
        assert (sp / "run_main.py").exists()
        assert (sp / "discovered").is_dir()

    @pytest.mark.parametrize("eval_type", ["performance", "time", "energy"])
    def test_make_files_description_contains_eval_description(
        self, mf: MakeFiles, config_with_tmp: dict[str, Any], eval_type: str
    ) -> None:
        """Test that the generated description includes the eval type description."""
        mf.make_files(config_with_tmp, train=True, use_base=True, no_data=True, eval_type=eval_type)

        sp = Path(config_with_tmp["source_path"])
        desc = (sp / "description.md").read_text()
        eval_desc = mf._get_eval_description(eval_type)
        assert eval_desc in desc

    @pytest.mark.parametrize("eval_type", ["performance", "time", "energy"])
    def test_make_files_energy_requirements(
        self, mf: MakeFiles, config_with_tmp: dict[str, Any], eval_type: str
    ) -> None:
        """Test that requirements.txt contains codecarbon only for energy eval type."""
        mf.make_files(config_with_tmp, train=True, use_base=True, no_data=True, eval_type=eval_type)

        sp = Path(config_with_tmp["source_path"])
        reqs = (sp / "requirements.txt").read_text()
        if eval_type == "energy":
            assert "codecarbon" in reqs
        else:
            assert "codecarbon" not in reqs

    def test_make_files_baseline_scale(self, mf: MakeFiles, config_with_tmp: dict[str, Any]) -> None:
        """Test that baseline_scale is reflected in baseline_scores.json values."""
        mf.make_files(config_with_tmp, train=True, use_base=True, no_data=True, eval_type="time", baseline_scale=2.0)

        sp_scaled = Path(config_with_tmp["source_path"])
        task_ids = mf._normalize_task_ids(config_with_tmp, "train")
        model_ids = mf._normalize_model_ids(config_with_tmp, "train", task_ids)
        dest = sp_scaled / f"{task_ids[0]}_{model_ids[0]}" if model_ids[0] else sp_scaled / task_ids[0]
        scaled_scores = json.loads((dest / "baseline_scores.json").read_text())

        # Re-run with scale=1.0 to compare
        mf.make_files(config_with_tmp, train=True, use_base=True, no_data=True, eval_type="time", baseline_scale=1.0)
        unscaled_scores = json.loads((dest / "baseline_scores.json").read_text())

        for key in unscaled_scores:
            if unscaled_scores[key] != 0:
                assert scaled_scores[key] == pytest.approx(unscaled_scores[key] * 2.0)

    @pytest.mark.parametrize("eval_type", ["performance", "time", "energy"])
    def test_make_files_run_main_matches_eval_type(
        self, mf: MakeFiles, config_with_tmp: dict[str, Any], eval_type: str, tmp_path: Path
    ) -> None:
        """Test that the run_main.py copied by make_files matches the standalone _load_run_main output."""
        mf.make_files(config_with_tmp, train=True, use_base=True, no_data=True, eval_type=eval_type)

        sp = Path(config_with_tmp["source_path"])
        e2e_content = (sp / "run_main.py").read_text()

        # Load via _load_run_main into a separate temp dir and compare
        standalone_dir = tmp_path / "standalone"
        standalone_dir.mkdir()
        mf.source_path = standalone_dir
        mf._load_run_main(eval_type)
        standalone_content = (standalone_dir / "run_main.py").read_text()

        assert e2e_content == standalone_content
