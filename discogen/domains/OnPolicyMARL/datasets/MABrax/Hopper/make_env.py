import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper

def make_env():
    env = jaxmarl.make("hopper_3x1", **{"homogenisation_method": "max"})
    env = LogWrapper(env)
    return env
