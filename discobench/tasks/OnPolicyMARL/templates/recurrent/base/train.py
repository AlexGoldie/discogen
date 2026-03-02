import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import NamedTuple
from flax.training.train_state import TrainState
from networks import ActorCritic, RecurrentModule
from loss import loss_actor_and_critic
from targets import get_targets
from activation import activation
from make_env import make_env
from jaxmarl.environments import spaces
from optim import scale_by_optimizer

class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    env = make_env()
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
        network = ActorCritic(action_dim, config=config, activation=activation)
        dummy_avail_actions = jnp.ones((config["NUM_ACTORS"], action_dim))
        rng, _rng = jax.random.split(rng)
        if config.get("GET_AVAIL_ACTIONS", False):
            init_x = (
                jnp.zeros(
                    (1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])
                ),
                jnp.zeros((1, config["NUM_ENVS"])),
                jnp.zeros((1, config["NUM_ENVS"], get_action_dim(env.action_space(env.agents[0])))),
            )
        else:
            init_x = (
                jnp.zeros(
                    (1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])
                ),
                jnp.zeros((1, config["NUM_ENVS"])),
            )
        init_hstate = RecurrentModule.initialize_carry(config["NUM_ENVS"])
        network_params = network.init(_rng, init_hstate, init_x)
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
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_hstate = RecurrentModule.initialize_carry(config["NUM_ACTORS"])
        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state

                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                # check if env has attribute get_avail_actions
                if config.get("GET_AVAIL_ACTIONS", False):
                    avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                    avail_actions = jax.lax.stop_gradient(
                        batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                    )
                    ac_in = (
                        obs_batch[np.newaxis, :],
                        last_done[np.newaxis, :],
                        avail_actions,
                    )
                else:
                    avail_actions = dummy_avail_actions
                    ac_in = (
                        obs_batch[np.newaxis, :],
                        last_done[np.newaxis, :],
                    )


                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.squeeze() if isinstance(env.action_space(k), spaces.Discrete) else v for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                _info = {}
                _info["returned_episode_returns"] = info["returned_episode_returns"]
                _info["returned_episode"] = info["returned_episode"]
                if "returned_won_episode" in info:
                    _info["returned_won_episode"] = info["returned_won_episode"]
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), _info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(0),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    avail_actions,
                )
                runner_state = (train_state, env_state, obsv, done_batch, hstate, rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            if config.get("GET_AVAIL_ACTIONS", False):
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )
                ac_in = (
                    last_obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    avail_actions,
                )
            else:
                avail_actions = dummy_avail_actions
                ac_in = (
                    last_obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze()

            advantages, targets = get_targets(traj_batch, last_val, config)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    grad_fn = jax.value_and_grad(loss_actor_and_critic, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params,
                        init_hstate,
                        traj_batch,
                        advantages,
                        targets,
                        network,
                        config,
                    )
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

                # adding an additional "fake" dimensionality to perform minibatching correctly
                init_hstate = jnp.reshape(
                    init_hstate, (1, config["NUM_ACTORS"], -1)
                )
                batch = (
                    init_hstate,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate.squeeze(),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]


            metric = {}
            metric["returned_episode_returns"] = traj_batch.info["returned_episode_returns"]
            metric["returned_episode"] = traj_batch.info["returned_episode"]
            if "returned_won_episode" in traj_batch.info:
                metric["returned_won_episode"] = traj_batch.info["returned_won_episode"]
            metric = jax.tree.map(
                lambda x: x.reshape(
                        (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)
                ),
                metric,
            )

            ratio_0 = loss_info[1][3].at[0,0].get().mean()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            metric["loss"] = {
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "ratio": loss_info[1][3],
                "ratio_0": ratio_0,
                "approx_kl": loss_info[1][4],
                "clip_frac": loss_info[1][5],
            }
            # For LR tuner: use env return metric (jaxmarl often uses returned_episode_returns)
            _ret = metric.get("returned_episode_returns", jnp.nan)
            if isinstance(_ret, dict):
                _ret = _ret.get("__all__", jnp.nan)
            metric["mean_training_return"] = jnp.nanmean(_ret) if hasattr(_ret, "shape") else _ret

            rng = update_state[-1]

            def callback(metric):
                returns = metric["returned_episode_returns"][:, :, 0][
                    metric["returned_episode"][:, :, 0]
                ].mean()
                if "returned_won_episode" in metric:
                    win_rate = metric["returned_won_episode"][:, :, 0][
                        metric["returned_episode"][:, :, 0]
                    ].mean()
                else:
                    win_rate = None
                steps = metric["update_steps"]
                num_updates = config["NUM_UPDATES"]
                print(f"returns: {returns}, win_rate: {win_rate}, step: {steps} / {num_updates}")


            metric["update_steps"] = update_steps
            if config.get("DEBUG", False):
                jax.experimental.io_callback(callback, None, metric)
            metric_return = {"mean_training_return": metric["mean_training_return"]}
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return (runner_state, update_steps), metric_return

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            init_hstate,
            _rng,
        )
        (runner_state, _), metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train
