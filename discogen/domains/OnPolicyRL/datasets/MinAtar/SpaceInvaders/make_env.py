import gymnax
import wrappers


def make_env():
    env, env_params = gymnax.make("SpaceInvaders-MinAtar")
    env = wrappers.GymnaxNoAutoResetWrapper(env)
    env = wrappers.FlattenObservationWrapper(env)
    env = wrappers.LogWrapper(env)
    env = wrappers.AutoResetEnvWrapper(env)
    env = wrappers.VecEnv(env)
    return env, env_params
