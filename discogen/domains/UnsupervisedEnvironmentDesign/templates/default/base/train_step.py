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

        # CALCULATE ADVANTAGE
        train_state, env_state, start_state, last_obs, last_done, extra, hstate, update_steps, rng = runner_state
        obs_to_use = last_obs
        _, _, last_val = network.apply(
            train_state.params,
            hstate,
            (
                jax.tree.map(lambda x: x[np.newaxis, :], obs_to_use),
                last_done[np.newaxis, :],
            ),
        )
        last_val = last_val.squeeze()

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition: Transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.next_done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + config["gamma"] * next_value * (1 - done) - value
                gae = delta + config["gamma"] * config["gae_lambda"] * (1 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(train_state, batch_info):
                init_hstate, traj_batch, advantages, targets = batch_info

                def _loss_fn_masked(params, init_hstate, traj_batch, gae, targets):

                    # RERUN NETWORK
                    obs_to_use = traj_batch.obs

                    (_, pi, value) = network.apply(
                        params,
                        jax.tree.map(lambda x: x.transpose(), init_hstate),
                        (obs_to_use, traj_batch.done),
                    )

                    metrics = {}
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                        -config["clip_eps"], config["clip_eps"]
                    )
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped)
                    critic_loss = config["vf_coef"] * value_loss.mean()

                    # CALCULATE ACTOR LOSS
                    logratio = log_prob - traj_batch.log_prob
                    ratio = jnp.exp(logratio)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config["clip_eps"],
                            1.0 + config["clip_eps"],
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    entropy = pi.entropy().mean()

                    approx_kl = jax.lax.stop_gradient(((ratio - 1) - logratio).mean())
                    clipfrac = jax.lax.stop_gradient((jnp.abs(ratio - 1) > config["clip_eps"]).mean())

                    total_loss = loss_actor + critic_loss - config["ent_coef"] * entropy
                    return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clipfrac, metrics)

                grad_fn = jax.value_and_grad(_loss_fn_masked, has_aux=True)
                total_loss, grads = grad_fn(train_state.params, init_hstate, traj_batch, advantages, targets)
                total_loss, grads = jax.lax.pmean((total_loss, grads), axis_name="devices")
                train_state = train_state.apply_gradients(grads=grads)

                return train_state, total_loss

            (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            ) = update_state
            rng, _rng = jax.random.split(rng)

            init_hstate = jax.tree.map(lambda x: jnp.reshape(x, (256, config["num_train_envs"])), init_hstate)
            batch = (
                init_hstate,
                traj_batch,
                advantages.squeeze(),
                targets.squeeze(),
            )
            permutation = jax.random.permutation(_rng, config["num_train_envs"])

            shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=1), batch)

            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(
                    jnp.reshape(
                        x,
                        [x.shape[0], config["num_minibatches"], -1] + list(x.shape[2:]),
                    ),
                    1,
                    0,
                ),
                shuffled_batch,
            )

            train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            return update_state, total_loss

        init_hstate = jax.tree.map(lambda x: x[None, :].squeeze().transpose(), initial_hstate)
        update_state = (
            train_state,
            init_hstate,
            traj_batch,
            advantages,
            targets,
            rng,
        )
        update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["update_epochs"])
        train_state, rng = update_state[0], update_state[-1]
        metric = jax.tree.map(
            lambda x: x.reshape((config["num_steps"], config["num_train_envs"])),
            traj_batch.info,
        )

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
        return (rng2, runner_state, instances), metric

    return train_step
