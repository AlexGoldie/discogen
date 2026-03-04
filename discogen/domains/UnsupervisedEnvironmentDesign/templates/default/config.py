from variable_config import config as variable_config
from env_specific_config import config as env_specific_config
FIXED_CONFIG = {
    "group": "test",
    "checkpoint_save_freq": -1
}

def _merge(a, b):
    for k, v in b.items():
        if k in a:
            for subkey, inner_val in v.items():
                if subkey in a[k]:
                    raise ValueError(f"Key {k}.{subkey} found in both fixed and variable config.")
                a[k][subkey] = inner_val
        else:
            a[k] = v
    return a


config = _merge(variable_config, env_specific_config)
config = _merge(config, FIXED_CONFIG)
