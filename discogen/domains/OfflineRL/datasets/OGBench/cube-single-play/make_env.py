"""Environment creation for cube-single-play."""

import numpy as np
import ogbench

from datasets import Dataset


def make_env_and_datasets(env_name, action_clip_eps=1e-5):
    """Create OGBench environment and datasets.

    Args:
        env_name: Name of the OGBench environment.
        action_clip_eps: Epsilon for action clipping.

    Returns:
        Tuple of (env, eval_env, train_dataset, val_dataset).
    """
    # Create environment and load datasets
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name)
    eval_env = ogbench.make_env_and_datasets(env_name, env_only=True)

    # Clip actions to valid range
    if action_clip_eps is not None:
        train_dataset['actions'] = np.clip(
            train_dataset['actions'],
            -1 + action_clip_eps,
            1 - action_clip_eps
        ).astype(np.float32)
        if val_dataset is not None:
            val_dataset['actions'] = np.clip(
                val_dataset['actions'],
                -1 + action_clip_eps,
                1 - action_clip_eps
            ).astype(np.float32)

    # Ensure float32 dtype
    for key in train_dataset:
        train_dataset[key] = train_dataset[key].astype(np.float32)
    if val_dataset is not None:
        for key in val_dataset:
            val_dataset[key] = val_dataset[key].astype(np.float32)

    return env, eval_env, train_dataset, val_dataset
