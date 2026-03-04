from jaxued.environments import Maze, MazeRenderer
from jaxued.environments.maze.env import EnvParams as MazeEnvParams
from jaxued.wrappers import AutoReplayWrapper
from jaxued.environments.maze import Level, make_level_generator, make_level_mutator_minimax
import jax
from gymnax.environments.environment import Environment, EnvState
from jaxued.environments import UnderspecifiedEnv
import chex
import jax.numpy as jnp
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union
from flax import struct
from kinetix.environment.wrappers import LogWrapper


@struct.dataclass
class EnvParams(MazeEnvParams):
    max_timesteps: int = 250


def unwrapped(env):
    return env._env._env


def _level_to_env_state(env, env_params, rng, level):
    _, state = unwrapped(env).reset_env_to_level(rng, level, env_params)
    return state


class MinigridToKinetixWrapper(Environment):
    """
    This wrapper follows the same interface as the KinetixEnvironment, i.e., allowing an optional parameter to be passed to reset or step.
    """

    def __init__(self, env: UnderspecifiedEnv):
        self._env = env

    @property
    def default_params(self) -> EnvParams:
        p = self._env.default_params

        return EnvParams(max_timesteps=p.max_steps_in_episode)

    # Overridden functions from Gymnax
    def step_env(self, rng, state, action: jnp.ndarray, env_params):
        return self._env.step(
            rng,
            state,
            action,
            env_params,
        )

    def reset_env(self, rng, env_params, override_reset_state: EnvState):
        assert override_reset_state is not None, "Must provide override_reset_state"
        env_state = override_reset_state
        return self._env.get_obs(env_state), env_state

    # Copied from gymnax so we could add kwargs
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        env_params: Optional[EnvParams] = None,
        override_reset_state: Optional[EnvState] = None,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        if env_params is None:
            env_params = self.default_params
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(key, state, action, env_params)

        info["GoalR"] = reward > 0

        obs_re, state_re = self.reset_env(key_reset, env_params, override_reset_state=override_reset_state)

        # Auto-reset environment based on termination
        state = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), state_re, state_st)
        obs = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st)

        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: chex.PRNGKey,
        env_params: Optional[EnvParams] = None,
        override_reset_state: Optional[EnvState] = None,
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        # Use default env parameters if no others specified
        if env_params is None:
            env_params = self.default_params

        obs, state = self.reset_env(key, env_params, override_reset_state=override_reset_state)
        return obs, state

    def action_space(self, params: EnvParams) -> Any:
        return self._env.action_space(params)


def make_envs(config):

    env = Maze(max_height=13, max_width=13, agent_view_size=config["agent_view_size"], normalize_obs=True)

    eval_env = env
    env = LogWrapper(MinigridToKinetixWrapper(env))
    eval_env = LogWrapper(MinigridToKinetixWrapper(eval_env))

    all_eval_levels = Level.load_prefabs(config["eval_levels"])
    all_eval_levels = jax.vmap(
        lambda level: _level_to_env_state(env, env.default_params, jax.random.PRNGKey(0), level)
    )(all_eval_levels)
    return env, eval_env, all_eval_levels, env.default_params, None


def init_level_samplers(config, env_params, static_env_params, env):
    sample_random_level = make_level_generator(unwrapped(env).max_height, unwrapped(env).max_width, config["n_walls"])

    def sample_random_state(rng):
        rng, _rng, __rng = jax.random.split(rng, 3)
        level = sample_random_level(_rng)
        state = _level_to_env_state(env, env_params, __rng, level)
        return state

    def sample_random_levels(rng, n):
        rngs = jax.random.split(rng, n)
        levels = jax.vmap(sample_random_state)(rngs)
        return levels

    return sample_random_levels
