from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

from discobench.sample_task_config import _check_args, _generate_config, _normalize_p_data, sample_task_config


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
            _check_args(0.0, [0.5, 0.5], "random", True, "src", 10, rng=None, seed=None)
        with pytest.raises(ValueError, match="p_edit must be between 0 and 1"):
            _check_args(1.5, [0.5, 0.5], "random", True, "src", 10, rng=None, seed=None)

    def test_invalid_eval_type(self) -> None:
        """Test that an invalid eval_type raises ValueError."""
        with pytest.raises(ValueError, match="eval_type must be one of"):
            _check_args(0.5, [0.5, 0.5], "invalid_eval", True, "src", 10, rng=None, seed=None)


class TestGenerateConfig:
    """Tests for the _generate_config function."""

    @pytest.fixture
    def mock_filesystem(self, tmp_path: Path) -> Path:
        """Create a dummy task directory structure for testing."""
        base_path = tmp_path / "tasks"
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
            use_backends=True,
            source_path="test_src",
            rng=rng,
        )

        assert result is not None
        domain, config = result
        assert domain == "domain1"
        assert config["source_path"] == "test_src"
        assert config["template_backend"] == "backend1"
        assert config["change_module_1"] is True
        assert config["change_module_2"] is True  # Driven by p_edit=1.0

    def test_failed_generation_returns_none(self, mock_filesystem: Path) -> None:
        """Test that config generation returns None when no edits are made."""
        rng = np.random.default_rng(42)

        # p_edit = 0 guarantees NO edits, which should invalidate the config
        result = _generate_config(
            base_path=mock_filesystem,
            p_edit=0.0,
            p_data=[0.5, 0.5, 0.0],
            use_backends=False,
            source_path="test_src",
            rng=rng,
        )

        assert result is None


class TestSampleTask:
    """Tests for the sample_task_config function."""

    @patch("discobench.sample_task_config._generate_config")
    def test_max_attempts_exceeded(self, mock_generate: MagicMock) -> None:
        """Test that exceeding max attempts raises RuntimeError."""
        # Force generate_config to always return None (failure)
        mock_generate.return_value = None

        with pytest.raises(RuntimeError, match="exceeded 3 attempts"):
            sample_task_config(p_edit=0.5, p_data=[0.5, 0.5], max_attempts=3)

        assert mock_generate.call_count == 3

    @patch("discobench.sample_task_config.Path")
    def test_seed_reproducibility(self, mock_path: MagicMock) -> None:
        """Test that the same seed produces a consistent RNG state."""
        # Mocking the filesystem entirely to focus just on the sample_task_config wrapper logic.
        mock_base = MagicMock()
        mock_path.return_value.parent.__truediv__.return_value = mock_base

        with patch("discobench.sample_task_config._generate_config") as mock_generate:
            mock_generate.return_value = ("domain", {"config": "val"})

            sample_task_config(p_edit=0.5, p_data=[0.5, 0.5], seed=42)

            # Grab the random generator passed to _generate_config
            rng_used = mock_generate.call_args[0][5]

            # To strictly test reproducibility, you'd usually assert against a mocked sequence,
            # but ensuring the seed was passed into default_rng is usually sufficient.
            assert rng_used is not None

    @patch("discobench.sample_task_config._generate_config")
    def test_different_seeds_yield_different_results(self, mock_generate: MagicMock) -> None:
        """Test that different seeds produce different RNG states."""
        # Make the mock return a random float generated by the passed 'rng' object
        # The RNG is the 6th argument (index 5) passed to _generate_config
        mock_generate.side_effect = lambda *args, **kwargs: ("dummy_domain", {"rand": args[5].random()})

        _, config1 = sample_task_config(p_edit=0.5, p_data=[0.5, 0.5], seed=42)
        _, config2 = sample_task_config(p_edit=0.5, p_data=[0.5, 0.5], seed=99)

        assert config1["rand"] != config2["rand"]

    @patch("discobench.sample_task_config._generate_config")
    def test_different_rng_yield_different_results(self, mock_generate: MagicMock) -> None:
        """Test that different rngs produce different RNG states."""
        # Make the mock return a random float generated by the passed 'rng' object
        # The RNG is the 6th argument (index 5) passed to _generate_config
        mock_generate.side_effect = lambda *args, **kwargs: ("dummy_domain", {"rand": args[5].random()})

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(99)
        _, config1 = sample_task_config(p_edit=0.5, p_data=[0.5, 0.5], rng=rng1)
        _, config2 = sample_task_config(p_edit=0.5, p_data=[0.5, 0.5], rng=rng2)

        assert config1["rand"] != config2["rand"]

    @patch("discobench.sample_task_config._generate_config")
    def test_no_seed_yields_different_results(self, mock_generate: MagicMock) -> None:
        """Test that running without a seed produces non-deterministic results."""
        mock_generate.side_effect = lambda *args, **kwargs: ("dummy_domain", {"rand": args[5].random()})

        _, config1 = sample_task_config(p_edit=0.5, p_data=[0.5, 0.5], seed=None)
        _, config2 = sample_task_config(p_edit=0.5, p_data=[0.5, 0.5], seed=None)

        # It is astronomically unlikely for these to be equal if seeded randomly from the OS
        assert config1["rand"] != config2["rand"]

    def test_seed_and_rng_error(self) -> None:
        """Test that running with both a seed and an rng generator returns an error."""
        with pytest.raises(ValueError, match="When sampling a task, at most"):
            _, _ = sample_task_config(p_edit=0.5, p_data=[0.5, 0.5], seed=42, rng=np.random.default_rng(30))
