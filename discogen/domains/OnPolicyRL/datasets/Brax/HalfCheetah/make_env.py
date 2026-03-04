import wrappers


def make_env():
    env, env_params = wrappers.BraxGymnaxWrapper(env_name="halfcheetah"), None
    env = wrappers.ClipAction(env)
    env = wrappers.LogWrapper(env)
    env = wrappers.VecEnv(env)
    env = wrappers.NormalizeObservation(env)
    env = wrappers.NormalizeReward(env, 0.99)
    return env, env_params
