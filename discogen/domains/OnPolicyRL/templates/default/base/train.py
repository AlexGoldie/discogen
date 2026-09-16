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
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.environments import environment, spaces
from loss import loss_actor_and_critic
from make_env import make_env
from networks import ActorCritic
from optim import make_optimizer
from activation import get_activation
from targets import get_targets
from schedule import make_schedule_fn


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env, env_params = make_env()

    def train(rng, lr):
        # multiply lr by -1, since we focus on gradient *descent* and scale_by_optimizer is implemented for gradient *ascent*
        lr = -1 * lr

        def get_action_dim(action_space):
            if isinstance(action_space, spaces.Discrete):
                return action_space.n
            elif isinstance(action_space, spaces.Box):
                return action_space.shape[0]
            else:
                raise ValueError(f"Unsupported action space type: {type(action_space)}")

        # INIT NETWORK
        network = ActorCritic(
            get_action_dim(env.action_space(env_params)), config=config, activation=get_activation(config)
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)

        if config.get("SCHEDULE_LR", True):
            scale_by_optimizer = make_optimizer(config)
            schedule_fn = make_schedule_fn(config, lr)
            tx = optax.chain(
                scale_by_optimizer(),
                optax.scale_by_schedule(schedule_fn),
            )
        else:
            scale_by_optimizer = make_optimizer(config)
            constant_schedule = optax.linear_schedule(
                init_value=lr, end_value=lr, transition_steps=0
            )
            tx = optax.chain(
                scale_by_optimizer(),
                optax.scale_by_schedule(constant_schedule),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params
                )

                bootstrap_truncation = jnp.asarray(info["truncated"]) & ~jnp.asarray(
                    info["terminated"]
                )

                def get_final_value(_):
                    _, final_value = network.apply(
                        train_state.params, info["final_observation"]
                    )
                    return final_value

                final_value = jax.lax.cond(
                    jnp.any(bootstrap_truncation),
                    get_final_value,
                    lambda _: jnp.zeros_like(value),
                    operand=None,
                )
                reward = jnp.where(
                    bootstrap_truncation,
                    reward + config["GAMMA"] * jax.lax.stop_gradient(final_value),
                    reward,
                )
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            advantages, targets = get_targets(traj_batch, last_val, config=config)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    grad_fn = jax.value_and_grad(loss_actor_and_critic, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params,
                        traj_batch,
                        advantages,
                        targets,
                        network,
                        config,
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                # Batching and Shuffling
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                # Mini-batch Updates
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            # Updating Training State and Metrics:
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info

            finished_episodes_mask = metric["returned_episode"]

            finished_returns = jnp.where(
                finished_episodes_mask,
                metric["returned_episode_returns"],
                0.0
            )
            total_return = finished_returns.sum()

            total_episodes_finished = jnp.sum(finished_episodes_mask)

            mean_return = jax.lax.cond(
                total_episodes_finished == 0,
                lambda: jnp.nan, # Return NaN if no episodes finished
                lambda: total_return / total_episodes_finished,
            )

            # Create the final metric dictionary to be returned
            metric = {"mean_training_return": mean_return}

            rng = update_state[-1]

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train
