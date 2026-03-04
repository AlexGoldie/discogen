import jaxmarl
from jaxmarl.wrappers.baselines import MPELogWrapper

def make_env():
    env = jaxmarl.make("MPE_simple_spread_v3")
    env = MPELogWrapper(env)
    return env
