from kinetix.environment import (
    LogWrapper,
    make_kinetix_env,
    make_reset_fn_from_config,
    make_vmapped_filtered_level_sampler,
)

from kinetix.util import load_evaluation_levels, get_eval_level_groups
from kinetix.environment.env import EnvParams, StaticEnvParams


def make_envs(config):
    env_params, static_env_params = EnvParams(**config["env_params"]), StaticEnvParams(**config["static_env_params"])

    def make_env(static_env_params):
        env = LogWrapper(
            make_kinetix_env(
                config["action_type"],
                config["observation_type"],
                None,
                env_params,
                static_env_params,
            )
        )
        return env

    env = make_env(static_env_params)
    num_eval_levels = len(config["eval_levels"])
    all_eval_levels, eval_static_env_params = load_evaluation_levels(config["eval_levels"])
    eval_group_indices = get_eval_level_groups(config["eval_levels"])

    eval_env = make_env(eval_static_env_params)

    return env, eval_env, all_eval_levels, env_params, static_env_params


def init_level_samplers(config, env_params, static_env_params, env):

    sample_random_level = make_reset_fn_from_config(
        config, env_params, static_env_params, physics_engine=env.physics_engine
    )
    sample_random_levels = make_vmapped_filtered_level_sampler(
        sample_random_level, env_params, static_env_params, config, env=env
    )

    return sample_random_levels
