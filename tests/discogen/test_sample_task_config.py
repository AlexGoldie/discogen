from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

from discogen.sample_task_config import _check_args, _generate_config, _normalize_p_data, sample_task_config


class TestNormalizePData:
    """Tests for the _normalize_p_data function."""

    def test_valid_length_2(self) -> None:
        """Test normalization with a length-2 input."""
        assert _normalize_p_data([0.4, 0.4]) == pytest.approx([0.4, 0.4, 0.2])

    def test_valid_length_3(self) -> None:
        """Test normalization with a length-3 input."""
        assert _normalize_p_data([1.0, 1.0, 2.0]) == pytest.approx([0.25, 0.25, 0.5])

    def test_invalid_lengths(self) -> None:
        """Test that invalid input lengths raise ValueError."""
        with pytest.raises(ValueError, match="length 2"):
            _normalize_p_data([0.5])
        with pytest.raises(ValueError, match="length 2"):
            _normalize_p_data([0.2, 0.2, 0.2, 0.2])

    def test_negative_values(self) -> None:
        """Test that negative values raise ValueError."""
        with pytest.raises(ValueError, match="All p_data entries must be >= 0"):
            _normalize_p_data([-0.1, 0.5])

    def test_sum_greater_than_one_len_2(self) -> None:
        """Test that sum > 1 for length-2 input raises ValueError."""
        with pytest.raises(ValueError, match="> 1"):
            _normalize_p_data([0.6, 0.5])

    def test_single_value_too_high(self) -> None:
        """Test that a single value >= 1 raises ValueError."""
        with pytest.raises(ValueError, match=r"Each entry.*must be < 1"):
            _normalize_p_data([1.0, 0.0])


class TestCheckArgs:
    """Tests for the _check_args function."""

    def test_invalid_p_edit(self) -> None:
        """Test that invalid p_edit values raise ValueError."""
        with pytest.raises(ValueError, match="p_edit must be between 0 and 1"):
            _check_args(0.0, [0.5, 0.5], 0.5, "random", True, "src", 10, None, None)
        with pytest.raises(ValueError, match="p_edit must be between 0 and 1"):
            _check_args(1.5, [0.5, 0.5], 0.5, "random", True, "src", 10, None, None)

    def test_invalid_p_use_base(self) -> None:
        """Test that invalid p_use_base values raise ValueError."""
        with pytest.raises(ValueError, match="p_use_base must be between 0 and 1"):
            _check_args(0.5, [0.5, 0.5], -0.1, "random", True, "src", 10, None, None)
        with pytest.raises(ValueError, match="p_use_base must be between 0 and 1"):
            _check_args(0.5, [0.5, 0.5], 1.5, "random", True, "src", 10, None, None)

    def test_valid_p_use_base_zero(self) -> None:
        """Test that p_use_base=0.0 is accepted."""
        _check_args(0.5, [0.5, 0.5], 0.0, "random", True, "src", 10, None, None)

    def test_invalid_eval_type(self) -> None:
        """Test that an invalid eval_type raises ValueError."""
        with pytest.raises(ValueError, match="eval_type must be one of"):
            _check_args(0.5, [0.5, 0.5], 0.5, "invalid_eval", True, "src", 10, None, None)


class TestGenerateConfig:
    """Tests for the _generate_config function."""

    @pytest.fixture
    def mock_filesystem(self, tmp_path: Path) -> Path:
        """Create a dummy task directory structure for testing."""
        base_path = tmp_path / "domains"
        base_path.mkdir()

        # Create Dummy Domain 1
        domain1 = base_path / "domain1"
        domain1.mkdir()

        config = {
            "train_task_id": ["dataset_A", "dataset_B", "dataset_C"],
            "change_module_1": True,
            "change_module_2": False,
        }

        with open(domain1 / "task_config.yaml", "w") as f:
            yaml.dump(config, f)

        templates = domain1 / "templates"
        templates.mkdir()
        (templates / "backend1").mkdir()

        return base_path

    def test_successful_generation(self, mock_filesystem: Path) -> None:
        """Test that config generation succeeds with valid inputs."""
        rng = np.random.default_rng(42)  # Fixed seed for predictability

        # p_edit=1.0 guarantees edits, [0.5, 0.5] guarantees roughly equal train/test splits
        result = _generate_config(
            base_path=mock_filesystem,
            p_edit=1.0,
            p_data=[0.5, 0.5, 0.0],
            p_use_base=0.5,
            use_backends=True,
            source_path="test_src",
            rng=rng,
            eval_type="time",
        )

        assert result is not None
        domain, config = result
        assert domain == "domain1"
        assert config["source_path"] == "test_src"
        assert config["template_backend"] == "backend1"
        assert config["change_module_1"] is True
        assert config["change_module_2"] is True  # Driven by p_edit=1.0
        assert config["eval_type"] == "time"
        assert isinstance(config["use_base"], bool)

    def test_failed_generation_returns_none(self, mock_filesystem: Path) -> None:
        """Test that config generation returns None when no edits are made."""
        rng = np.random.default_rng(42)

        # p_edit = 0 guarantees NO edits, which should invalidate the config
        result = _generate_config(
            base_path=mock_filesystem,
            p_edit=0.0,
            p_data=[0.5, 0.5, 0.0],
            p_use_base=0.5,
            use_backends=False,
            source_path="test_src",
            rng=rng,
            eval_type="energy",
        )

        assert result is None

    def test_random_eval_type(self, mock_filesystem: Path) -> None:
        """Test that eval_type='random' resolves to a valid eval_type."""
        rng = np.random.default_rng(42)
        result = _generate_config(
            base_path=mock_filesystem,
            p_edit=1.0,
            p_data=[0.5, 0.5, 0.0],
            p_use_base=0.5,
            use_backends=True,
            source_path="test_src",
            rng=rng,
            eval_type="random",
        )
        assert result is not None
        _, config = result
        assert config["eval_type"] in ["performance", "energy", "time"]

    def test_use_base_always_true(self, mock_filesystem: Path) -> None:
        """Test that p_use_base=1.0 always produces use_base=True."""
        rng = np.random.default_rng(42)
        result = _generate_config(
            base_path=mock_filesystem,
            p_edit=1.0,
            p_data=[0.5, 0.5, 0.0],
            p_use_base=1.0,
            use_backends=True,
            source_path="test_src",
            rng=rng,
            eval_type="performance",
        )
        assert result is not None
        _, config = result
        assert config["use_base"] is True

    def test_use_base_always_false(self, mock_filesystem: Path) -> None:
        """Test that p_use_base=0.0 always produces use_base=False."""
        rng = np.random.default_rng(42)
        result = _generate_config(
            base_path=mock_filesystem,
            p_edit=1.0,
            p_data=[0.5, 0.5, 0.0],
            p_use_base=0.0,
            use_backends=True,
            source_path="test_src",
            rng=rng,
            eval_type="performance",
        )
        assert result is not None
        _, config = result
        assert config["use_base"] is False


class TestSampleTask:
    """Tests for the sample_task_config function."""

    @patch("discogen.sample_task_config._generate_config")
    def test_max_attempts_exceeded(self, mock_generate: MagicMock) -> None:
        """Test that exceeding max attempts raises RuntimeError."""
        # Force generate_config to always return None (failure)
        mock_generate.return_value = None

        with pytest.raises(RuntimeError, match="exceeded 3 attempts"):
            sample_task_config(p_edit=0.5, p_data=[0.5, 0.5], max_attempts=3)

        assert mock_generate.call_count == 3

    @patch("discogen.sample_task_config._generate_config")
    def test_seed_reproducibility(self, mock_generate: MagicMock) -> None:
        """Test that the same seed produces a consistent RNG state."""
        mock_generate.return_value = ("domain", {"config": "val"})

        sample_task_config(p_edit=0.5, p_data=[0.5, 0.5], seed=42)

        rng_used = mock_generate.call_args[0][6]
        assert isinstance(rng_used, np.random.Generator)

    @patch("discogen.sample_task_config._generate_config")
    def test_different_seeds_yield_different_results(self, mock_generate: MagicMock) -> None:
        """Test that different seeds produce different RNG states."""
        # The RNG is the 7th argument (index 6) passed to _generate_config
        mock_generate.side_effect = lambda *args, **kwargs: ("dummy_domain", {"rand": args[6].random()})

        _, config1 = sample_task_config(p_edit=0.5, p_data=[0.5, 0.5], seed=42)
        _, config2 = sample_task_config(p_edit=0.5, p_data=[0.5, 0.5], seed=99)

        assert config1["rand"] != config2["rand"]

    @patch("discogen.sample_task_config._generate_config")
    def test_rng_is_forwarded(self, mock_generate: MagicMock) -> None:
        """Test that a caller-provided rng is passed through, not replaced."""
        mock_generate.return_value = ("domain", {"config": "val"})

        rng = np.random.default_rng(42)
        sample_task_config(p_edit=0.5, p_data=[0.5, 0.5], rng=rng)

        rng_used = mock_generate.call_args[0][6]
        assert rng_used is rng

    @patch("discogen.sample_task_config._generate_config")
    def test_eval_type_is_forwarded(self, mock_generate: MagicMock) -> None:
        """Test that eval_type is passed through to _generate_config."""
        mock_generate.return_value = ("domain", {"config": "val"})

        sample_task_config(p_edit=0.5, p_data=[0.5, 0.5], eval_type="energy")

        eval_type_used = mock_generate.call_args[0][7]
        assert eval_type_used == "energy"

    @patch("discogen.sample_task_config._generate_config")
    def test_p_use_base_is_forwarded(self, mock_generate: MagicMock) -> None:
        """Test that p_use_base is passed through to _generate_config."""
        mock_generate.return_value = ("domain", {"config": "val"})

        sample_task_config(p_edit=0.5, p_data=[0.5, 0.5], p_use_base=0.8)

        p_use_base_used = mock_generate.call_args[0][3]
        assert p_use_base_used == 0.8

    @patch("discogen.sample_task_config._generate_config")
    def test_different_rng_yield_different_results(self, mock_generate: MagicMock) -> None:
        """Test that different rngs produce different RNG states."""
        # The RNG is the 7th argument (index 6) passed to _generate_config
        mock_generate.side_effect = lambda *args, **kwargs: ("dummy_domain", {"rand": args[6].random()})

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(99)
        _, config1 = sample_task_config(p_edit=0.5, p_data=[0.5, 0.5], rng=rng1)
        _, config2 = sample_task_config(p_edit=0.5, p_data=[0.5, 0.5], rng=rng2)

        assert config1["rand"] != config2["rand"]

    @patch("discogen.sample_task_config._generate_config")
    def test_no_seed_yields_different_results(self, mock_generate: MagicMock) -> None:
        """Test that running without a seed produces non-deterministic results."""
        mock_generate.side_effect = lambda *args, **kwargs: ("dummy_domain", {"rand": args[6].random()})

        _, config1 = sample_task_config(p_edit=0.5, p_data=[0.5, 0.5], seed=None)
        _, config2 = sample_task_config(p_edit=0.5, p_data=[0.5, 0.5], seed=None)

        # It is astronomically unlikely for these to be equal if seeded randomly from the OS
        assert config1["rand"] != config2["rand"]

    def test_seed_and_rng_error(self) -> None:
        """Test that running with both a seed and an rng generator returns an error."""
        with pytest.raises(ValueError, match="When sampling a task, at most"):
            _, _ = sample_task_config(p_edit=0.5, p_data=[0.5, 0.5], seed=42, rng=np.random.default_rng(30))
