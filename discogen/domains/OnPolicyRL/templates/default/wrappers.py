from functools import partial
from typing import Any, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
from brax import envs as brax_envs
from brax.envs.base import Wrapper as BraxWrapper
from flax import struct
from gymnax.environments import environment, spaces


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0.0, 0, 0.0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info = dict(info)
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info


class FlattenObservationWrapper(GymnaxWrapper):
    """Flatten the observations of the environment."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def observation_space(self, params) -> spaces.Box:
        assert isinstance(
            self._env.observation_space(params), spaces.Box
        ), "Only Box spaces are supported for now."
        return spaces.Box(
            low=self._env.observation_space(params).low,
            high=self._env.observation_space(params).high,
            shape=(np.prod(self._env.observation_space(params).shape),),
            dtype=self._env.observation_space(params).dtype,
        )

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, state = self._env.reset(key, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state, reward, done, info


class GymnaxNoAutoResetWrapper(GymnaxWrapper):
    """Expose Gymnax transitions before its built-in auto-reset."""

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, key, state, action, params=None):
        obs, state, reward, step_done, info = self._env.step_env(
            key, state, action, params
        )

        if hasattr(self._env, "is_terminated") and hasattr(
            self._env, "is_truncated"
        ):
            terminated = self._env.is_terminated(state, params)
            truncated = self._env.is_truncated(state, params)
        else:
            truncated = state.time >= params.max_steps_in_episode
            # Older Gymnax versions include the time limit in step_done.
            terminated = jnp.asarray(step_done) & ~jnp.asarray(truncated)

        done = jnp.asarray(terminated) | jnp.asarray(truncated)
        info = dict(info)
        info["terminated"] = terminated
        info["truncated"] = truncated
        return obs, state, reward, done, info


class CraftaxTerminationWrapper(GymnaxWrapper):
    """Separate Craftax's natural terminals from its configured time limit."""

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, key, state, action, params=None):
        obs, state, reward, _, info = self._env.step(key, state, action, params)

        truncated = state.timestep >= params.max_timesteps
        state_without_timeout = state.replace(
            timestep=jnp.minimum(state.timestep, params.max_timesteps - 1)
        )
        terminated = self._env.is_terminal(state_without_timeout, params)
        done = jnp.asarray(terminated) | jnp.asarray(truncated)

        info = dict(info)
        info["terminated"] = terminated
        info["truncated"] = truncated
        return obs, state, reward, done, info


class AutoResetEnvWrapper(GymnaxWrapper):
    """Provides standard auto-reset functionality, providing the same behaviour as Gymnax-default."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):

        rng, _rng = jax.random.split(rng)
        obs_st, state_st, reward, done, info = self._env.step(
            _rng, state, action, params
        )

        rng, _rng = jax.random.split(rng)
        obs_re, state_re = self._env.reset(_rng, params)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree.map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return obs, state

        obs, state = auto_reset(done, state_re, state_st, obs_re, obs_st)

        info = dict(info)
        info["final_observation"] = obs_st

        return obs, state, reward, done, info


class CaptureFinalObservation(BraxWrapper):
    """Record Brax observations before its cached auto-reset replaces them."""

    def reset(self, rng):
        state = self.env.reset(rng)
        # Keep the state info pytree identical between reset and step.
        state.info["final_observation"] = state.obs
        return state

    def step(self, state, action):
        state = self.env.step(state, action)
        state.info["final_observation"] = state.obs
        return state


class BraxGymnaxWrapper:
    def __init__(self, env_name):

        env = brax_envs.get_environment(env_name=env_name, backend="positional")
        env = brax_envs.wrappers.training.EpisodeWrapper(
            env, episode_length=1000, action_repeat=1
        )
        env = CaptureFinalObservation(env)
        env = brax_envs.wrappers.training.AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)
        self.max_steps_in_episode = env.episode_length

    def reset(self, key, params=None):
        # print(f'bye: {key}')
        state = self._env.reset(key)
        return state.obs, state

    def step(self, key, state, action, params):
        next_state = self._env.step(state, action)
        done = next_state.done > 0.5
        truncated = next_state.info["truncation"] > 0.5
        terminated = done & ~truncated
        info = {
            "terminated": terminated,
            "truncated": truncated,
            "final_observation": next_state.info["final_observation"],
        }
        return next_state.obs, next_state, next_state.reward, done, info

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )


@struct.dataclass
class NormalizeObsEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    env_state: environment.EnvState


class NormalizeObservation(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        state = NormalizeObsEnvState(
            mean=jnp.zeros_like(obs[0]),
            var=jnp.ones_like(obs[0]),
            count=1e-4,
            env_state=state,
        )
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=state.env_state,
        )

        return (obs - state.mean) / jnp.sqrt(state.var + 1e-8), state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )

        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=env_state,
        )
        scale = jnp.sqrt(state.var + 1e-8)
        info = dict(info)
        final_observation = info.get("final_observation", obs)
        info["final_observation"] = (final_observation - state.mean) / scale
        return (
            (obs - state.mean) / scale,
            state,
            reward,
            done,
            info,
        )


@struct.dataclass
class NormalizeRewEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float
    env_state: environment.EnvState


class NormalizeReward(GymnaxWrapper):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        batch_count = obs.shape[0]
        state = NormalizeRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        return_val = state.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )
        return obs, state, reward / jnp.sqrt(state.var + 1e-8), done, info


class ClipAction(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, key, state, action, params=None):
        action = jnp.clip(action, -1, 1)
        return self._env.step(key, state, action, params)


class VecEnv(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))
