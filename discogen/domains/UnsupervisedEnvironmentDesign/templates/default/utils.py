import datetime
import logging
from kinetix.util import normalise_config, generate_params_from_config
from omegaconf import OmegaConf
from flax.serialization import to_state_dict
import jax
from omegaconf import OmegaConf
import copy


def get_logger():
    """Creates a formatted logger"""

    # ANSI escape codes for colors
    COLOR_CODES = {
        "DEBUG": "\033[94m",  # Blue
        "INFO": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[1;91m",  # Bold Red
    }
    RESET = "\033[0m"

    class ColorFormatter(logging.Formatter):
        def format(self, record):
            log_color = COLOR_CODES.get(record.levelname, "")
            log_time = datetime.datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
            formatted_message = f"{log_time} | {record.levelname}: {record.getMessage()}"
            return f"{log_color}{formatted_message}{RESET}"

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = ColorFormatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False  # ✅ Prevent messages from being passed to the root logger
    return logger


def normalise_minigrid_config(config):
    old_config = copy.deepcopy(config)
    keys = ["env", "learning", "model", "misc", "eval", "ued", "env_size", "train_levels"]
    for k in keys:
        if k not in config:
            config[k] = {}
        small_d = config[k]
        del config[k]
        for kk, vv in small_d.items():
            assert kk not in config, kk
            config[kk] = vv

    config["num_eval_levels"] = len(config["eval_levels"])

    steps = config["num_steps"] * config.get("outer_rollout_steps", 1) * config["num_train_envs"]
    config["num_updates"] = int(config["total_timesteps"]) // steps

    return config


def wrangle_config(config):
    if config["env_name"] == "Minigrid":
        config = normalise_minigrid_config(config)
    else:
        config = normalise_config(
            config, "SFL" if config["ued"]["ratio_from_buffer"] > 0 else "SFL-DR", save_config=False
        )
        env_params, static_env_params = generate_params_from_config(config)

        config["env_params_dict"] = to_state_dict(env_params)
        config["static_env_params_dict"] = to_state_dict(static_env_params)
        config["env_params"] = env_params
        config["static_env_params"] = static_env_params
    config["rollout_episodes"] = 1

    config["num_gpus"] = jax.device_count()

    assert config["num_batches"] % config["num_gpus"] == 0, f"num_batches {config['num_batches']} must be divisible by num_gpus {config['num_gpus']}"
    config["num_updates"] = config["num_updates"] // config["num_gpus"]
    config["num_to_save"] = config["num_to_save"] // config["num_gpus"]
    config["num_train_envs"] = config["num_train_envs"] // config["num_gpus"]
    config["num_batches"] = config["num_batches"] // config["num_gpus"]

    config["num_envs_from_buffer"] = int(config["num_train_envs"] * config["ratio_from_buffer"])
    config["num_envs_randomly_generated"] = int(config["num_train_envs"] * (1 - config["ratio_from_buffer"]))
    assert (config["num_envs_from_buffer"] + config["num_envs_randomly_generated"]) == config["num_train_envs"]

    config["total_timesteps"] = config["num_updates"] * config["num_steps"] * config["num_train_envs"]
    config["minibatch_size"] = config["num_train_envs"] * config["num_steps"] // config["num_minibatches"]

    return OmegaConf.create(config)
