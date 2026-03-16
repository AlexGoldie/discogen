import os
import time

import jax
import jax.experimental
import jax.numpy as jnp
import optax
from flax.jax_utils import replicate, unreplicate
from flax.training.train_state import TrainState
import make_network

import json

# modules
import utils
import make_env
import sample_levels
import train_step
import eval_utils
from config import config

os.environ["WANDB_DISABLE_SERVICE"] = "True"


logger = utils.get_logger()


def get_top_scoring_levels(config, sample_random_levels, env, env_params, rng, train_state, initialize_carry_fn):
    batch_step = sample_levels.make_sample_and_score_fn(
        config, sample_random_levels, env, env_params, initialize_carry_fn
    )

    rngs = jax.random.split(rng, config["num_batches"])
    _, (levels, scores, extra) = jax.lax.scan(batch_step, (train_state), rngs, config["num_batches"])

    flat_env_instances = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), levels)
    scores = scores.flatten()
    top_level_indices = jnp.argsort(scores)[-config["num_to_save"] :]

    top_instances = jax.tree.map(lambda x: x.at[top_level_indices].get(), flat_env_instances)
    top_learn = scores.at[top_level_indices].get()
    return top_instances, top_learn


def main(config):
    config = utils.wrangle_config(config)
    rng = jax.random.PRNGKey(config["seed"])

    # Get Env
    env, eval_env, all_eval_levels, env_params, static_env_params = make_env.make_envs(config)

    # Get level sampler
    sample_random_levels = make_env.init_level_samplers(config, env_params, static_env_params, env)

    sample_random_level = lambda rng: jax.tree.map(lambda x: x[0], sample_random_levels(rng, 1))

    # Sample DR evaluation levels
    rng, _rng_reset = jax.random.split(rng)

    key_to_sample_dr_eval_set = jax.random.PRNGKey(100)
    DR_EVAL_LEVELS = sample_random_levels(key_to_sample_dr_eval_set, config["num_dr_eval_levels"])

    # Get the eval functions
    eval_unseen_levels_fn, eval_random_levels_fn = eval_utils.make_eval_fns(
        config,
        env,
        eval_env,
        env_params,
        all_eval_levels,
        DR_EVAL_LEVELS,
        initialize_carry_fn=make_network.initialize_carry,
    )

    network = make_network.make_network(config, env_params, env)

    def linear_schedule(count):
        count = count // (config["num_minibatches"] * config["update_epochs"])
        frac = 1.0 - count / config["num_updates"]
        return config["lr"] * frac

    # INIT NETWORK
    rng, _rng_reset = jax.random.split(rng)
    train_envs = 32  # arbitrary
    obs, _ = env.reset(rng, env_params, sample_random_level(rng))
    obs = jax.tree.map(
        lambda x: jnp.repeat(jnp.repeat(x[None, ...], train_envs, axis=0)[None, ...], 256, axis=0),
        obs,
    )
    init_x = (obs, jnp.zeros((256, train_envs)))
    init_hstate = make_network.initialize_carry(train_envs)
    network_params = {"params": network.init(_rng_reset, init_hstate, init_x)["params"]}

    lr_to_use = linear_schedule if config["anneal_lr"] else config["lr"]
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=lr_to_use, eps=1e-5),
        ),
    )
    rng, _rng_reset = jax.random.split(rng)

    # INIT ENV
    rng, _rng_reset, _rng_samples = jax.random.split(rng, 3)
    rng_resets = jax.random.split(_rng_reset, config["num_train_envs"])

    new_levels = sample_random_levels(_rng_samples, config["num_train_envs"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None, 0))(rng_resets, env_params, new_levels)

    start_state = env_state
    init_hstate = make_network.initialize_carry(config["num_train_envs"])

    get_new_train_buffer = lambda rng, ts, extra: get_top_scoring_levels(
        config,
        sample_random_levels,
        env,
        env_params,
        rng,
        ts,
        initialize_carry_fn=make_network.initialize_carry,
    )

    def _eval_step(runner_state_instances, rng):
        update_count_in_eval = runner_state_instances[0][-2]
        test_metrics = {}
        rng, _rng = jax.random.split(rng)

        # EVAL
        _rng, rng_eval, rng_eval_dr, rng_eval_buffer = jax.random.split(_rng, 4)
        (states, cum_rewards, _, episode_lengths, eval_infos), (eval_dones, eval_rewards) = jax.vmap(
            eval_unseen_levels_fn, (0, None)
        )(
            jax.random.split(rng_eval, config["eval_num_attempts"]),
            (
                runner_state_instances[0][0],
                runner_state_instances[0][-4],
            ),
        )
        all_eval_eplens = episode_lengths

        # Collect Metrics
        eval_returns = cum_rewards.mean(axis=0)  # (num_eval_levels,)
        mask = jnp.arange(env_params.max_timesteps)[None, ..., None] < episode_lengths[:, None, :]
        eval_solves = (eval_infos["returned_episode_solved"] * eval_dones * mask).sum(axis=1) / jnp.maximum(
            1, (eval_dones * mask).sum(axis=1)
        )
        eval_solves = eval_solves.mean(axis=0)
        # just grab the first run
        states, episode_lengths = jax.tree_util.tree_map(
            lambda x: x[0], (states, episode_lengths)
        )  # (num_steps, num_eval_levels, ...), (num_eval_levels,)
        # And one attempt
        states = jax.tree_util.tree_map(lambda x: x[:, :], states)
        episode_lengths = episode_lengths[:]

        test_metrics["update_count"] = runner_state[-2]
        test_metrics["eval_returns"] = eval_returns
        test_metrics["eval_ep_lengths"] = episode_lengths

        # Eval on sampled
        dr_states, dr_cum_rewards, _, dr_episode_lengths, dr_infos = jax.vmap(eval_random_levels_fn, (0, None))(
            jax.random.split(rng_eval_dr, config["eval_num_attempts"]),
            (
                runner_state_instances[0][0],
                runner_state_instances[0][-4],
            ),
        )

        eval_dr_returns = dr_cum_rewards.mean(axis=0).mean()
        eval_dr_eplen = dr_episode_lengths.mean(axis=0).mean()

        mask_dr = jnp.arange(env_params.max_timesteps)[None, ..., None] < dr_episode_lengths[:, None, :]
        my_eval_dones = dr_infos["returned_episode"]
        eval_dr_solves = (
            (
                (dr_infos["returned_episode_solved"] * my_eval_dones * mask_dr).sum(axis=1)
                / jnp.maximum(1, (my_eval_dones * mask_dr).sum(axis=1))
            )
            .mean(axis=0)
            .mean()
        )

        test_metrics["eval/mean_eval_return_sampled"] = eval_dr_returns
        test_metrics["eval/mean_eval_eplen_sampled"] = eval_dr_eplen
        test_metrics["eval/mean_eval_solve_sampled"] = eval_dr_solves

        # Collect Metrics
        eval_returns = cum_rewards.mean(axis=0)  # (num_eval_levels,)

        log_dict = {}
        log_dict["to_remove"] = {
            "eval_return": eval_returns,
            "eval_solve_rate": eval_solves,
            "eval_eplen": all_eval_eplens,
        }
        for i, name in enumerate(config["eval_levels"]):
            log_dict[f"eval_info/eval_avg_return/{name}"] = eval_returns[i]
            log_dict[f"eval_info/eval_avg_solve_rate/{name}"] = eval_solves[i]
        log_dict.update({"eval/mean_eval_return": eval_returns.mean()})
        log_dict.update({"eval/mean_eval_solve_rate": eval_solves.mean()})
        log_dict.update({"eval/mean_eval_eplen": all_eval_eplens.mean()})
        jax.debug.print("Eval returns {} {}", eval_returns.shape, eval_returns.mean())

        test_metrics.update(log_dict)
        return test_metrics

    def single_device_train_step(runner_state_instances):
        train_step_fn = train_step.make_train_step_fn(
            config,
            network,
            env,
            env_params,
            sample_random_levels,
        )
        return jax.lax.scan(train_step_fn, runner_state_instances, None, config["eval_freq"])

    pmapped_train_step_fn = jax.pmap(single_device_train_step, axis_name="devices")
    pmapped_get_buffer_fn = jax.pmap(get_new_train_buffer, axis_name="devices")

    def train_and_eval_step(runner_state_instances, eval_rng):
        time_dic = {}
        time_start = time.time()
        runner_state, instances, carry = runner_state_instances

        (
            learnability_rng,
            eval_singleton_rng,
            eval_tl_rng,
            carry_learnability_rng,
            log_learnability_rng,
        ) = jax.random.split(eval_rng, 5)

        update_step = runner_state[-2]

        train_state_replicate = replicate(runner_state[0], jax.local_devices())
        extra_replicate = replicate(runner_state[-4])

        def _update_buffer(instances, learnability_rng):
            def _new_buffer(learnability_rng):
                rngs = jax.random.split(learnability_rng, jax.local_device_count())
                instances, learnabilty_scores = pmapped_get_buffer_fn(rngs, train_state_replicate, extra_replicate)
                instances, learnabilty_scores = jax.tree.map(
                    lambda x: x.reshape(x.shape[0] * x.shape[1], *x.shape[2:]),
                    (instances, learnabilty_scores),
                )

                return instances, learnabilty_scores

            should_get_new_buffer = jnp.array(update_step % config["buffer_update_frequency"] == 0, dtype=bool)

            scores = runner_state[-4]["scores"]
            if config["eval_freq"] == config["buffer_update_frequency"]:
                # This is much faster
                instances, learnabilty_scores = _new_buffer(learnability_rng)
            else:
                # this is actually a lot of wasted compute but is likely better
                instances_new, learnabilty_scores_new = _new_buffer(learnability_rng)

                def _do_new():
                    return (
                        instances_new,
                        learnabilty_scores_new,
                    )

                def _do_old():
                    return (
                        instances,
                        scores,
                    )

                instances, learnabilty_scores = jax.lax.cond(should_get_new_buffer, _do_new, _do_old)
            return instances, learnabilty_scores

        instances, learnabilty_scores = _update_buffer(instances, learnability_rng)

        extra = runner_state[-4]
        extra["scores"] = learnabilty_scores
        # add extra back to runner state
        runner_state = runner_state[:-4] + (extra,) + runner_state[-3:]

        time_dic["timing/get_buffer"] = t = time.time() - time_start
        logger.info(f"Timing:: Getting Buffer {t:.2f}s")
        time_curr = time.time()

        time_dic["timing/logging_buffer"] = t = time.time() - time_curr
        logger.info(f"Timing:: Logging Buffer (1) {t:.2f}s")
        time_curr = time.time()

        # TRAIN
        runner_state_instances = (
            jax.random.split(runner_state[-1], jax.local_device_count()),
            replicate(runner_state, jax.local_devices()),
            replicate(instances, jax.local_devices()),
        )

        runner_state_instances, _ = pmapped_train_step_fn(runner_state_instances)
        runner_state_instances = (
            unreplicate(runner_state_instances[0]),
            unreplicate(runner_state_instances[1]),
            unreplicate(runner_state_instances[2]),
        )
        runner_state_instances = runner_state_instances[1:]

        time_dic["timing/training"] = t = time.time() - time_curr
        logger.info(f"Timing:: Training {t:.2f}s")
        time_curr = time.time()

        time_dic["timing/log_buffer2"] = t = time.time() - time_curr
        logger.info(f"Timing:: Logging Buffer (2) {t:.2f}s")
        time_curr = time.time()

        # EVAL
        test_metrics = {}
        test_metrics.update(_eval_step(runner_state_instances, eval_singleton_rng))
        runner_state, _ = runner_state_instances
        test_metrics["update_count"] = runner_state[-2]

        time_dic["timing/eval"] = t = time.time() - time_curr
        logger.info(f"Timing:: Getting Buffer {t:.2f}s")
        time_curr = time.time()
        time_dic["timing/total_iteration"] = t = time_curr - time_start
        logger.info(f"Timing:: Total Iteration {t:.2f}s")
        test_metrics.update(time_dic)

        return (runner_state, instances, carry), test_metrics

    extra = {
        "scores": jnp.zeros((config["num_to_save"] * config["num_gpus"])),
    }
    rng, _rng_reset = jax.random.split(rng)
    runner_state = (
        train_state,
        env_state,
        start_state,
        obsv,
        jnp.zeros((config["num_train_envs"]), dtype=bool),
        extra,
        init_hstate,
        0,
        _rng_reset,
    )

    instances = sample_random_levels(rng, config["num_to_save"] * config["num_gpus"])
    carry = []

    # eval at the start

    for eval_step in range(int(config["num_updates"] * config["num_gpus"] // config["eval_freq"])):
        rng, eval_rng = jax.random.split(rng)
        runner_state_instances, metrics = train_and_eval_step((runner_state, instances, carry), eval_rng)
        runner_state, instances, carry = runner_state_instances

    return_out = {"solve_rate": metrics["eval/mean_eval_solve_rate"].item()}
    print(json.dumps(return_out))


if __name__ == "__main__":
    main(config)
