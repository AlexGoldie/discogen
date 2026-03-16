import wrappers
from config import config
from craftax.craftax_env import make_craftax_env_from_name


def make_env():
    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", False)
    env_params = env.default_params
    env = wrappers.FlattenObservationWrapper(env)
    env = wrappers.LogWrapper(env)
    env = wrappers.AutoResetEnvWrapper(env)
    env = wrappers.VecEnv(env)
    return env, env_params
