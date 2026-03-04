import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper

def make_env():
    env = jaxmarl.make("humanoid_9|8", **{"homogenisation_method":"max"})
    env = LogWrapper(env)
    return env
