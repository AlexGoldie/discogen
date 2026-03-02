import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from networks import ActorCritic
from optim import scale_by_optimizer
from loss import loss_actor_and_critic
from make_env import make_env
from activation import activation
from targets import get_targets
from jaxmarl.environments import spaces

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    ac_in: tuple

def batchify(x: dict, agent_list, num_actors):
    max_dim = max([x[a].shape[-1] for a in agent_list])
    def pad(z):
        return jnp.concatenate([z, jnp.zeros(z.shape[:-1] + (max_dim - z.shape[-1],))], -1)

    x = jnp.stack([x[a] if x[a].shape[-1] == max_dim else pad(x[a]) for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_train(config):
    env = make_env()
    max_dim = jnp.argmax(jnp.array([env.observation_space(a).shape[-1] for a in env.agents]))
    init_x = jnp.zeros(env.observation_space(env.agents[max_dim]).shape)

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )




    def train(rng, lr):
        # multiply lr by -1, since we focus on gradient *descent* and scale_by_optimizer is implemented for gradient *ascent*
        lr = -1 * lr

        def linear_anneal(count):
            frac = (
                1.0
                - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
                / config["NUM_UPDATES"]
            )
            return lr * frac

        def get_action_dim(action_space):
            if isinstance(action_space, spaces.Discrete):
                return action_space.n
            elif isinstance(action_space, spaces.Box):
                return action_space.shape[0]
            else:
                raise ValueError(f"Unsupported action space type: {type(action_space)}")

        action_dim = get_action_dim(env.action_space(env.agents[0]))
        dummy_avail_actions = jnp.ones((config["NUM_ACTORS"], action_dim))
        # INIT NETWORK
        network = ActorCritic(
            action_dim,
            config=config,
            activation=activation,
        )

        rng, rng_init = jax.random.split(rng)
        network_params = network.init(rng_init, (init_x, dummy_avail_actions))
        schedule_fn = optax.linear_schedule(
            init_value=lr, end_value=lr, transition_steps=0
        )
        if config.get("ANNEAL_LR", True):
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                scale_by_optimizer(),
                optax.scale_by_schedule(linear_anneal),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                scale_by_optimizer(),
                optax.scale_by_schedule(schedule_fn),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, update_count, rng = runner_state

                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])

                if config.get("GET_AVAIL_ACTIONS", False):
                    avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                    avail_actions = jax.lax.stop_gradient(
                        batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                    )
                    ac_in = (obs_batch, avail_actions)
                else:
                    ac_in = (obs_batch, dummy_avail_actions)

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                pi, value = network.apply(train_state.params, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                # unbatchify always adds a trailing dim; squeeze it for discrete (1D) actions
                if action.ndim == 1:
                    env_act = {k: v.squeeze(-1) for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step, env_state, env_act,
                )

                # Broadcast env-level info to each agent, then flatten to NUM_ACTORS.
                info = jax.tree_map(
                    lambda x: jnp.broadcast_to(x, (env.num_agents,) + x.shape).reshape(
                        (config["NUM_ACTORS"],) + x.shape[1:]
                    ),
                    info,
                )
                transition = Transition(
                    batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    info,
                    ac_in,
                )
                runner_state = (train_state, env_state, obsv, update_count, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, update_count, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            last_ac_in = (last_obs_batch, dummy_avail_actions)
            _, last_val = network.apply(train_state.params, last_ac_in)

            advantages, targets = get_targets(traj_batch, last_val, config)

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

                    loss_info = {
                        "total_loss": total_loss[0],
                        "actor_loss": total_loss[1][1],
                        "critic_loss": total_loss[1][0],
                        "entropy": total_loss[1][2],
                        "ratio": total_loss[1][3],
                    }

                    return train_state, loss_info

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, loss_info = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, loss_info

            def callback(info):
                print(f'Env Step {info["env_step"]} Return: {info["returned_episode_returns"]}')


            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            update_count = update_count + 1
            r0 = {"ratio0": loss_info["ratio"][0,0].mean()}
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            metric = jax.tree.map(lambda x: x.mean(), metric)
            # For LR tuner: use env return metric
            _ret = metric.get("returned_episode_returns", jnp.nan)
            metric["mean_training_return"] = _ret["__all__"] if isinstance(_ret, dict) else _ret
            metric["update_step"] = update_count
            metric["env_step"] = update_count * config["NUM_STEPS"] * config["NUM_ENVS"]
            metric = {**metric, **loss_info, **r0}
            if config.get("DEBUG", False):
                jax.experimental.io_callback(callback, None, metric)
            metric_return = {"mean_training_return": metric["mean_training_return"]}
            runner_state = (train_state, env_state, last_obs, update_count, rng)
            return runner_state, metric_return

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, 0, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train
