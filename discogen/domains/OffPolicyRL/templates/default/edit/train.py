import jax
import jax.numpy as jnp
import chex
import flax
import optax
from flax.training.train_state import TrainState
from networks import QNetwork
from rb import get_replay_buffer
from optim import scale_by_optimizer
from policy import explore
from q_update import q_loss_fn
from make_env import make_env

@chex.dataclass(frozen=True)
class TimeStep:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array

class CustomTrainState(TrainState):
    target_network_params: flax.core.FrozenDict
    timesteps: int
    n_updates: int

def make_train(config):

    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"]

    env, env_params, basic_env = make_env()

    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

    network = QNetwork(action_dim=env.action_space(env_params).n, width=config["WIDTH"], depth=config["DEPTH"])

    def train(rng):

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        init_obs, env_state = vmap_reset(config["NUM_ENVS"])(_rng)

        # INIT BUFFER
        buffer = get_replay_buffer(config)

        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )

        rng = jax.random.PRNGKey(0)  # use a dummy rng here
        _action = basic_env.action_space(env_params).sample(rng)
        _, _env_state = env.reset(rng, env_params)
        _obs, _, _reward, _done, _ = env.step(rng, _env_state, _action, env_params)
        _timestep = TimeStep(obs=_obs, action=_action, reward=_reward, done=_done)
        buffer_state = buffer.init(_timestep)

        # INIT NETWORK AND OPTIMIZER
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)

        lr = config["LR"]*-1
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

        # TRAINING LOOP
        def _update_step(runner_state, unused):
            """
            Fill in your training loop logic here.
            Inputs:
            - runner_state: the current runner state.
            - unused: unused.

            Returns:
            - runner_state: the updated runner state. (make sure to include the timesteps, n_updates in the train_state)
            - metrics: the metrics.

            You must:
            - Sample an action from the current observation using the logic defined in policy.py
            - Store and sample experience from the replay buffer defined in rb.py
            - Calculate the loss using the loss function defined in q_update.py
            - Update the network parameters using the optimizer defined in optim.py
            """

            runner_state = ... # fill in your runner state here

            loss = ... # fill in your loss here

            metrics = {
                "timesteps": train_state.timesteps,
                "updates": train_state.n_updates,
                "loss": loss.mean(),
                "returns": info["returned_episode_returns"].mean(),
            }

            return runner_state, metrics

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, buffer_state, env_state, init_obs, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train
