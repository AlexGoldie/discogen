from sfl_types import Transition
from typing import Callable
import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
from sfl_types import Level, Environment, EnvParams
import chex


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
                return {"returns": r}

            done_idxs = jnp.argwhere(dones, size=50, fill_value=max_steps).squeeze()
            mask_done = jnp.where(done_idxs == max_steps, 0, 1)
            indices_to_use = jnp.concatenate([jnp.array([-1]), done_idxs[:-1]])
            outs = _ep_outcomes(
                indices_to_use,
                done_idxs,
            )

            outs = jax.tree.map(lambda x: x.mean(where=mask_done), outs)

            return {
                "outs": outs,
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
            return rng, o

        rng, outs = jax.lax.scan(_single_episode, rng, None, 1)
        # Sum over the scan dimension
        outs = jax.tree.map(lambda x: x.sum(axis=0), outs)
        returns = outs["outs"]["returns"].flatten()

        # This, for instance, scores all levels equally. You must do something else that properly scores levels.
        level_scores = jnp.ones_like(returns)

        extra_dict = {}
        return train_state, (env_instances, level_scores, extra_dict)

    return sample_and_score_fn
