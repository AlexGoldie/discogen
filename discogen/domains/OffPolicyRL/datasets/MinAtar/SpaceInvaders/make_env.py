import gymnax
import wrappers


def make_env():
    basic_env, env_params = gymnax.make("SpaceInvaders-MinAtar")
    env = wrappers.FlattenObservationWrapper(basic_env)
    env = wrappers.LogWrapper(env)
    return env, env_params, basic_env
