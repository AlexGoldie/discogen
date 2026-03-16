from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


class Dataset(FrozenDict):
    """Dataset class for offline RL.

    Wraps a dictionary of numpy arrays and provides batch sampling functionality.
    Supports returning next_actions for algorithms that need them.
    """

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from fields.

        Args:
            freeze: Whether to make arrays read-only.
            **fields: Dictionary fields (observations, actions, rewards, etc.).

        Returns:
            Dataset instance.
        """
        data = fields
        assert 'observations' in data, "Dataset must contain 'observations'"
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.return_next_actions = False  # Set externally if needed

        # Compute terminal locations for trajectory-aware sampling
        if 'terminals' in self._dict:
            self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
            self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        else:
            self.terminal_locs = np.array([self.size - 1])
            self.initial_locs = np.array([0])

    def get_random_idxs(self, num_idxs):
        """Return random indices for sampling."""
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample.
            idxs: Optional pre-specified indices.

        Returns:
            Dictionary of batched arrays.
        """
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        return batch

    def get_subset(self, idxs):
        """Get a subset of the dataset by indices.

        Args:
            idxs: Array of indices to retrieve.

        Returns:
            Dictionary of arrays at the specified indices.
        """
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)

        if self.return_next_actions:
            # Get next actions (clamped to valid indices)
            # WARNING: This is incorrect at trajectory boundaries - use with caution
            next_idxs = np.minimum(idxs + 1, self.size - 1)
            result['next_actions'] = self._dict['actions'][next_idxs]

        return result

    def copy(self, add_or_replace=None):
        """Create a copy of the dataset with optional modifications.

        Args:
            add_or_replace: Dictionary of fields to add or replace.

        Returns:
            New Dataset instance.
        """
        data = dict(self._dict)
        if add_or_replace is not None:
            data.update(add_or_replace)
        new_dataset = Dataset.create(freeze=False, **data)
        new_dataset.return_next_actions = self.return_next_actions
        return new_dataset
