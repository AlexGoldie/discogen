import jax
import jax.numpy as jnp
from sfl_types import Transition
import numpy as np
import make_network
def make_train_step_fn(
    config,
    network,
    env,
    env_params,
    sample_random_levels,

):
    """This is the train loop. You are given the current runner state (which includes env state, observations, dones, train_state, etc.) and you need to return the updated runner state after collecting new trajectories and performing learning updates.
    """
    # TRAIN LOOP
    @jax.jit
    def train_step(carry, unused):
        rng, runner_state, instances = carry
        rng2, rng = jax.random.split(rng)

        # COLLECT TRAJECTORIES
        runner_state = (*runner_state[:-1], rng)
        num_env_instances = config['num_to_save']

        def _env_step(runner_state, unused):
            train_state, env_state, start_state, last_obs, last_done, extra, hstate, update_steps, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            obs_to_use = last_obs
            ac_in = (
                jax.tree.map(lambda x: x[np.newaxis, :], obs_to_use),
                last_done[np.newaxis, :],
            )
            hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
            action = pi.sample(seed=_rng).squeeze()
            log_prob = pi.log_prob(action)
            env_act = action

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["num_train_envs"])
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None, 0))(
                rng_step, env_state, env_act, env_params, start_state
            )
            done_batch = done
            transition = Transition(
                done,
                last_done,
                action.squeeze(),
                value.squeeze(),
                reward,
                log_prob.squeeze(),
                last_obs,
                info,
            )
            runner_state = (train_state, env_state, start_state, obsv, done_batch, extra, hstate, update_steps, rng)
            return runner_state, (transition)

        initial_hstate = runner_state[-3]
        runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["num_steps"])

        train_state, env_state, start_state, last_obs, last_done, extra, hstate, update_steps, rng = runner_state

        # Update your agent here using the collected traj_batch

        # SAMPLE NEW ENVS
        def _sample_new_envs(rng):
            rng, _rng, _rng2 = jax.random.split(rng, 3)
            rng_reset = jax.random.split(_rng, config["num_envs_randomly_generated"])

            new_levels = sample_random_levels(_rng2, config["num_envs_randomly_generated"])
            obsv_gen, env_state_gen = jax.vmap(env.reset, in_axes=(0, None, 0))(rng_reset, env_params, new_levels)

            rng, _rng, _rng2 = jax.random.split(rng, 3)

            # This could be used to sample proportionally to the scores.
            # learn_scores = extra["scores"]
            # p_to_sample = learn_scores / (learn_scores.sum() + 1e-6)
            # sampled_env_instances_idxs = jax.random.choice(
            #     _rng, jnp.arange(num_env_instances), (config["num_envs_from_buffer"],), p=p_to_sample, replace=True
            # )
            # whereas this is uniform over the top k
            sampled_env_instances_idxs = jax.random.randint(
                _rng, (config["num_envs_from_buffer"],), 0, num_env_instances
            )
            sampled_env_instances = jax.tree.map(lambda x: x.at[sampled_env_instances_idxs].get(), instances)
            myrng = jax.random.split(_rng2, config["num_envs_from_buffer"])
            obsv_sampled, env_state_sampled = jax.vmap(env.reset, in_axes=(0, None, 0))(
                myrng, env_params, sampled_env_instances
            )

            obsv = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), obsv_gen, obsv_sampled)
            env_state = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), env_state_gen, env_state_sampled)

            return env_state, env_state, obsv, sampled_env_instances_idxs

        rng, _rng = jax.random.split(rng)
        env_state, start_state, obsv, sampled_env_instances_idxs = _sample_new_envs(_rng)

        update_steps = update_steps + 1
        runner_state = (
            train_state,
            env_state,
            start_state,
            obsv,
            jnp.zeros((config["num_train_envs"]), dtype=bool),
            extra,
            make_network.initialize_carry(config["num_train_envs"]),
            update_steps,
            rng,
        )
        metric = {}
        return (rng2, runner_state, instances), metric

    return train_step
