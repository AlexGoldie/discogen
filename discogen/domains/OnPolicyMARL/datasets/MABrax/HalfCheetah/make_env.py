import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper

def make_env():
    env = jaxmarl.make("halfcheetah_6x1", **{"homogenisation_method": "max"})
    env = LogWrapper(env)
    return env
