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
from loss import loss_actor_and_critic
from make_env import make_env
from networks import ActorCritic
from optim import scale_by_optimizer
from activation import activation
from targets import get_targets
from gymnax.environments import environment, spaces

def make_train(config):

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
            get_action_dim(env.action_space(env_params)), config=config, activation=activation
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)

        opt = scale_by_optimizer()
        def schedule(lr):
            return schedule_logic * lr

        # opt provides the scale_by_optimizer logic, but use full_opt in train state to combine opt, learning rate and other transformations
        full_opt = optax.chain(
            ...,
            scale_by_optimizer(),
            optax.scale_by_schedule(schedule)
        )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=full_opt,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        # STEP ENV
        obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)

        # CALCULATE THE GRADIENTS
        # Calculate advantages and targets
        advantages, targets = get_targets(traj_batch, last_val, config)

        grad_fn = jax.value_and_grad(loss_actor_and_critic, has_aux=True)
        total_loss, grads = grad_fn(
            train_state.params,
            # relevant inputs
        )
        train_state = train_state.apply_gradients(grads=grads)

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

        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train
