from kinetix.models.actor_critic import ActorCriticSymbolicRNN

from kinetix.models import make_network_from_config

def make_network(config, env_params, env):
    network = make_network_from_config(env, env_params, config)
    return network

def initialize_carry(batch_size):
    return ActorCriticSymbolicRNN.initialize_carry(batch_size)
