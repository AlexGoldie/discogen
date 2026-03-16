import json
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from config import config
from activation import activation
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.environments import environment, spaces
from loss import loss_actor_and_critic
from make_env import make_env
from networks import ActorCritic
from optim import scale_by_optimizer
from train import make_train

def extract_norm_stats(env_state):
    """Extract normalization statistics from environment state."""
    stats = {}

    def _extract(state, prefix=""):
        if hasattr(state, 'mean') and hasattr(state, 'var') and hasattr(state, 'count'):
            # Found normalization wrapper
            key = f"{prefix}normalize" if prefix else "normalize"
            stats[key] = {
                'mean': state.mean,
                'var': state.var,
                'count': state.count
            }
        if hasattr(state, 'env_state'):
            # Recurse into wrapped state
            new_prefix = f"{prefix}wrapped_" if prefix else "wrapped_"
            _extract(state.env_state, new_prefix)

    _extract(env_state)
    return stats


def apply_norm_stats(env_state, stats):
    """Apply normalization statistics to environment state."""

    def _apply(state, prefix=""):
        key = f"{prefix}normalize" if prefix else "normalize"

        if key in stats and hasattr(state, 'mean'):
            # Apply stats to this normalization wrapper
            # Handle batch dimension - take first element if batched
            mean = stats[key]['mean']
            var = stats[key]['var']
            count = stats[key]['count']

            if mean.ndim > 1:
                mean = mean[0]
                var = var[0]
            if isinstance(count, jnp.ndarray) and count.ndim > 0:
                count = count[0]

            state = state.replace(mean=mean, var=var, count=count)

        if hasattr(state, 'env_state'):
            # Recurse into wrapped state
            new_prefix = f"{prefix}wrapped_" if prefix else "wrapped_"
            new_env_state = _apply(state.env_state, new_prefix)
            state = state.replace(env_state=new_env_state)

        return state

    return _apply(env_state, "")

# Evaluation loop
def make_eval(config, num_episodes):
    def map_evaluate_policy(params, rng, env_state):
        def evaluate_policy(params, rng, env_state):

            old_norm_stats = extract_norm_stats(env_state)

            env, env_params = make_env()

            def get_action_dim(action_space):
                if isinstance(action_space, spaces.Discrete):
                    return action_space.n
                elif isinstance(action_space, spaces.Box):
                    return action_space.shape[0]
                else:
                    raise ValueError(
                        f"Unsupported action space type: {type(action_space)}"
                    )

            network = ActorCritic(
                get_action_dim(env.action_space(env_params)), config=config, activation=activation
            )
            rng, _rng = jax.random.split(rng)
            rng_reset = jax.random.split(_rng, num_episodes)
            obs, env_state = env.reset(rng_reset, env_params)

            env_state = apply_norm_stats(env_state, old_norm_stats)

            ever_done = jnp.zeros(num_episodes, dtype=bool)
            total_reward = jnp.zeros(num_episodes)

            def step_fn(carry, _):
                rng, obs, env_state, ever_done, total_reward = carry
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, num_episodes)

                pi, value = network.apply(params, obs)
                if config.get("CONTINUOUS", False):
                    action = pi.loc
                else:
                    action = pi.mode()

                next_obs, next_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params
                )
                total_reward = (
                    info["returned_episode_returns"] * (1 - ever_done) + total_reward
                )
                ever_done = jnp.logical_or(done, ever_done)
                return (rng, next_obs, next_state, ever_done, total_reward), None

            if hasattr(env_params, "max_steps_in_episode"):
                max_steps = env_params.max_steps_in_episode
            elif hasattr(env_params, "max_timesteps"):
                max_steps = env_params.max_timesteps
            else:
                max_steps = env.max_steps_in_episode

            (rng, obs, state, ever_done, total_reward), _ = jax.lax.scan(
                step_fn,
                (rng, obs, env_state, ever_done, total_reward),
                length=max_steps,
            )

            return total_reward

        return jax.vmap(evaluate_policy, in_axes=(0, None, 0))(params, rng, env_state)

    return map_evaluate_policy


if __name__ == "__main__":

    rng = jax.random.PRNGKey(30)
    lrs = jnp.linspace(config["LR"], config["LR"] * 10, 10)

    def run_tuner(config, base_rng, lrs, seeds):
        """
        Run LR tuning across multiple seeds and learning rates.

        Args:
            config: training config dict
            base_rng: PRNGKey for reproducibility
            lrs: jnp.ndarray of candidate learning rates [num_lrs]
            seeds: list/array of ints [num_seeds]
        Returns:
            metrics: pytree shaped [num_seeds, num_lrs, ...]
        """
        train = make_train(config)
        train_jit = jax.jit(train)

        # -----------------------
        # 1) Expand rng for seeds
        # Each seed gets its own rng
        rngs = jax.random.split(base_rng, seeds)

        # -----------------------
        # 2) vmap over lrs (inner axis)
        def run_one_seed(rng):
            # For fairness: split rng once per LR but *reuse same subkeys*
            sub_rngs = jax.random.split(rng, len(lrs))
            return jax.vmap(train_jit)(sub_rngs, lrs)

        # -----------------------
        # 3) vmap over seeds (outer axis)
        if jax.local_device_count() > 1:
            rngs = rngs.reshape([jax.local_device_count(), -1, 2])
            run_one_seed = jax.vmap(run_one_seed)
            run_one_seed = jax.pmap(run_one_seed)
        else:
            run_one_seed = jax.vmap(run_one_seed)

        metrics = run_one_seed(rngs)
        return metrics

    num_seeds = 8
    metrics = run_tuner(config, rng, lrs, num_seeds)

    # metrics has shape [num_seeds, num_lrs, num_updates]
    returns = metrics["metrics"]["mean_training_return"]
    len_returns = len(returns)
    returns = jnp.nanmean(
        returns[..., int(len(returns) * 0.95) :], axis=-1
    )  # compute return from the final 5% of training
    returns = returns.reshape([num_seeds, len(lrs)])

    mean_returns = returns.mean(axis=0)  # average over seeds per LR
    std_returns = returns.std(axis=0)

    for lr, m, s in zip(lrs, mean_returns, std_returns):
        print(
            f"LR={lr:.4f} -> training return (without eval policy) ={m:.4f} ± {s:.4f}"
        )

    best_idx = int(jnp.argmax(mean_returns))  # convert index to Python int
    best_mean = float(mean_returns[best_idx])  # convert scalar
    best_std = float(std_returns[best_idx])

    print(
        f"Best LR in training: {lrs[best_idx]:.4f} with avg training return {best_mean:.4f}"
    )
    agent_params = metrics["runner_state"][0].params

    env_state = metrics["runner_state"][1]

    evaluate_policy = make_eval(config, 16)

    if jax.local_device_count() > 1:
        evaluate_policy = jax.vmap(evaluate_policy, in_axes=(0, None, 0))
        evaluate_policy = jax.pmap(evaluate_policy, in_axes=(0, None, 0))
    else:
        evaluate_policy = jax.vmap(evaluate_policy, in_axes=(0, None, 0))

    eval_rng = jax.random.PRNGKey(42)

    eval_returns = evaluate_policy(agent_params, eval_rng, env_state)
    eval_returns = eval_returns.reshape(num_seeds, *eval_returns.shape[-2:]).swapaxes(
        0, 1
    )
    eval_returns = eval_returns.mean(-1)
    eval_mean_returns = eval_returns.mean(axis=-1)
    eval_std_returns = eval_returns.std(axis=-1)

    for lr, m, s in zip(lrs, eval_mean_returns, eval_std_returns):
        print(f"LR={lr:.4f} -> eval_return={m:.4f} ± {s:.4f}")

    best_eval_idx = int(jnp.argmax(eval_mean_returns))  # convert index to Python int
    best_eval_mean = float(eval_mean_returns[best_eval_idx])  # convert scalar
    best_eval_std = float(eval_std_returns[best_eval_idx])

    print(
        f"Best LR in evaluation: {lrs[best_eval_idx]:.4f} with avg return {best_eval_mean:.4f}"
    )
    return_out = {"return_mean": best_eval_mean, "return_std": best_eval_std}
    print(json.dumps(return_out))
