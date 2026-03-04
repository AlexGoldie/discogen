from sfl_types import Transition
from typing import Callable
import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
from sfl_types import Level, Environment, EnvParams
import chex


def compute_learnability(
    successes: jnp.ndarray, total_episodes: jnp.ndarray, do_correction: bool = False
) -> jnp.ndarray:
    """Computes learnability score from successes and total episodes.
        Learnability is defined as p * (1 - p) where p is the success rate.
        The correction is applied to remove the bias in the estimator.
    Args:
        The shape of successes and total_episodes should be (num_envs,).
    Returns:
        learnability score of shape (num_envs,).
    """
    success_p = successes / jnp.maximum(1, total_episodes)
    learn = success_p * (1 - success_p)
    correction = total_episodes / (total_episodes + 1)
    assert successes.shape == total_episodes.shape
    assert correction.shape == total_episodes.shape
    if do_correction:
        learn = learn * correction

    return learn


def make_sample_and_score_fn(
    config,
    sample_random_levels: Callable[[jax.Array, int], Level],
    env: Environment,
    env_params: EnvParams,
    initialize_carry_fn: Callable[[int], chex.ArrayTree],
):
    """This returns a function that takes in a train state, and rng key and returns a set of sampled envs and their scores.
    In particular, it returns a tuple of train_state, (env_instances, env_scores, extra_dict).
    Levels with high scores are selected to train the agent on.

    Args:
        config (_type_): OmegaConf
        sample_random_levels (Callable[[jax.Array, int], Level]): function that takes in a rng key and batch size and returns a batch of envs
        env (Environment):
        env_params (EnvParams):

    Returns:
        Callable[[Any, jax.Array], Tuple[Any, Tuple[Level, jax.Array, dict]]]: function that takes in a train state and rng key and returns a tuple of train_state, (env_instances, env_scores, extra_dict)
    """
    batch_size = config["batch_size"]

    @jax.jit
    def sample_and_score_fn(train_state, rng):

        # This steps the environment
        def _env_step(runner_state, _):
            env_state, start_state, last_obs, last_done, hstate, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            obs_to_use = last_obs

            ac_in = (
                jax.tree.map(lambda x: x[np.newaxis, :], obs_to_use),
                last_done[np.newaxis, :],
            )
            hstate, pi, value = train_state.apply_fn(train_state.params, hstate, ac_in)
            action = pi.sample(seed=_rng).squeeze()
            log_prob = pi.log_prob(action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, batch_size)
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None, 0))(
                rng_step, env_state, action, env_params, start_state
            )

            transition = Transition(
                done,
                done,
                action.squeeze(),
                value.squeeze(),
                reward,
                log_prob.squeeze(),
                last_obs,
                info,
            )
            runner_state = (env_state, start_state, obsv, done, hstate, rng)
            return runner_state, transition

        @partial(jax.vmap, in_axes=(None, 1, 1, 1))
        @partial(jax.jit, static_argnums=(0,))
        def compute_agent_metrics(max_steps: int, dones, returns, info):
            idxs = jnp.arange(max_steps)

            @partial(jax.vmap, in_axes=(0, 0))
            def _ep_outcomes(start_idx, end_idx):
                mask = (idxs > start_idx) & (idxs <= end_idx) & (end_idx != max_steps)
                r = jnp.sum(returns * mask)
                goal_r = info["GoalR"]
                success = jnp.sum(goal_r * mask)
                l = end_idx - start_idx
                return r, success, l

            done_idxs = jnp.argwhere(dones, size=50, fill_value=max_steps).squeeze()
            mask_done = jnp.where(done_idxs == max_steps, 0, 1)
            indices_to_use = jnp.concatenate([jnp.array([-1]), done_idxs[:-1]])
            ep_return, success, length = _ep_outcomes(
                indices_to_use,
                done_idxs,
            )

            return {
                "ep_return": ep_return.mean(where=mask_done),
                "num_episodes": mask_done.sum(),
                "success_rate": success.mean(where=mask_done),
                "ep_len": length.mean(where=mask_done),
                "done_sums": dones.sum(),
                "should_be_zero": ((info["GoalR"] * (1 - dones)).sum()),
                "num_successes": success.sum(),
                "total_episodes": mask_done.sum(),
            }

        # sample envs
        rng, _rng, _rng2 = jax.random.split(rng, 3)

        env_instances = sample_random_levels(_rng2, batch_size)

        def _single_episode(rng, _):
            rng, _rng, __rng = jax.random.split(rng, 3)
            obsv, env_state = jax.vmap(env.reset, in_axes=(0, None, 0))(
                jax.random.split(_rng, batch_size), env_params, env_instances
            )

            init_hstate = initialize_carry_fn(
                batch_size,
            )

            runner_state = (env_state, env_state, obsv, jnp.zeros((batch_size), dtype=bool), init_hstate, __rng)
            # Get the trajectory data
            _, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["rollout_steps"])
            # And here, compute the success / failure statistics
            o = compute_agent_metrics(config["rollout_steps"], traj_batch.done, traj_batch.reward, traj_batch.info)
            return rng, (
                o["num_successes"].reshape(batch_size),
                o["total_episodes"].reshape(batch_size),
            )

        rng, (total_successes, total_episodes) = jax.lax.scan(_single_episode, rng, None, 1)
        # Sum over the scan dimension
        num_episodes_by_env = total_episodes.sum(axis=0)
        total_successes_by_env = total_successes.sum(axis=0)

        # Compute learnability
        learnability_by_env = compute_learnability(
            total_successes_by_env, num_episodes_by_env, config["learnability_correction"]
        )

        extra_dict = {}
        return train_state, (env_instances, learnability_by_env, extra_dict)

    return sample_and_score_fn
