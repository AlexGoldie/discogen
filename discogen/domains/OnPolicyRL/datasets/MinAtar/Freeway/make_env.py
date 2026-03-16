import gymnax
import wrappers


def make_env():
    env, env_params = gymnax.make("Freeway-MinAtar")
    env = wrappers.FlattenObservationWrapper(env)
    env = wrappers.LogWrapper(env)
    env = wrappers.VecEnv(env)
    return env, env_params
