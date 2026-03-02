import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper

def make_env():
    env = jaxmarl.make("walker2d_2x3")
    env = LogWrapper(env)
    return env
