import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper

def make_env():
    env = jaxmarl.make("ant_4x2")
    env = LogWrapper(env)
    return env
