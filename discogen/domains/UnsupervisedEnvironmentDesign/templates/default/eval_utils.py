from kinetix.util import general_eval
import chex
from flax.training.train_state import TrainState

def make_eval_fns(config, env, eval_env, env_params, all_eval_levels, DR_EVAL_LEVELS, initialize_carry_fn):
    def eval(rng: chex.PRNGKey, train_state_extra: tuple[TrainState, dict], keep_states=True):
        """
        This evaluates the current policy on the set of evaluation levels specified by config["eval_levels"].
        It returns (states, cum_rewards, episode_lengths), with shapes (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)
        """
        num_levels = len(config["eval_levels"])

        return general_eval(
            rng,
            eval_env,
            env_params,
            train_state_extra[0],
            all_eval_levels,
            env_params.max_timesteps,
            num_levels,
            keep_states=keep_states,
            return_trajectories=True,
            initialize_carry_fn=initialize_carry_fn
        )

    def eval_on_dr_levels(rng: chex.PRNGKey, train_state_extra: tuple[TrainState, dict], keep_states=False):
        return general_eval(
            rng,
            env,
            env_params,
            train_state_extra[0],
            DR_EVAL_LEVELS,
            env_params.max_timesteps,
            config["num_dr_eval_levels"],
            keep_states=keep_states,
            initialize_carry_fn=initialize_carry_fn
        )

    return eval, eval_on_dr_levels
