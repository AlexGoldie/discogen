import json
import jax
import jax.numpy as jnp
from activation import activation
from jaxmarl.environments import spaces
from make_env import make_env
from networks import RecurrentModule, ActorCritic
from train import make_train, batchify, unbatchify
from config import config

# Evaluation loop
def make_eval(config, num_episodes):
    env = make_env()
    max_dim = jnp.argmax(jnp.array([env.observation_space(a).shape[-1] for a in env.agents]))
    def get_action_dim(action_space):
        if isinstance(action_space, spaces.Discrete):
            return action_space.n
        elif isinstance(action_space, spaces.Box):
            return action_space.shape[0]
        else:
            raise ValueError(f"Unsupported action space type: {type(action_space)}")

    network = ActorCritic(
        get_action_dim(env.action_space(env.agents[0])),
        config=config,
        activation=activation,
    )

    def eval(params, rng):
        def single_episode(rng):
            rng, rng_reset = jax.random.split(rng)
            obs, env_state = env.reset(rng_reset)
            init_hstate = RecurrentModule.initialize_carry(env.num_agents)

            def _cond_fn(runner_state):
                _, _, _, _, _, done = runner_state
                return jnp.logical_not(done)

            def _env_step(runner_state):
                rng, last_obs, h_state, env_state, total_reward, done = runner_state

                rng, rng_step = jax.random.split(rng)

                last_obs = jax.tree.map(lambda x: x[jnp.newaxis, :], last_obs)
                obs_batch = batchify(last_obs, env.agents, env.num_agents)
                if config.get("GET_AVAIL_ACTIONS", False):
                    avail_actions = env.get_avail_actions(env_state.env_state)
                    avail_actions = jax.tree.map(lambda x: x[jnp.newaxis, :], avail_actions)
                    avail_actions = batchify(avail_actions, env.agents, env.num_agents)
                    ac_in = (obs_batch[jnp.newaxis, :], jnp.zeros((1, env.num_agents)), avail_actions)
                else:
                    ac_in = (obs_batch[jnp.newaxis, :], jnp.zeros((1, env.num_agents)))

                next_h_state, pi, _ = network.apply(params, h_state, ac_in)

                if config.get("CONTINUOUS", False):
                    action = pi.loc
                else:
                    action = pi.mode()

                env_act = unbatchify(action, env.agents, 1, env.num_agents)
                env_act = {k: v[0].squeeze() if isinstance(env.action_space(k), spaces.Discrete) else v[0] for k, v in env_act.items()}

                next_obs, next_env_state, reward, done, info = env.step(
                    rng_step, env_state, env_act,
                )
                if '__all__' in reward:
                    reward = reward['__all__']
                else:
                    _reward = 0
                    for agent in env.agents:
                        _reward = _reward + reward[agent]
                    reward = _reward

                total_reward = total_reward + reward
                done = done['__all__']
                return (rng, next_obs, next_h_state, next_env_state, total_reward, done)

            runner_state = (rng, obs, init_hstate, env_state, 0, False)
            runner_state = jax.lax.while_loop(
                _cond_fn,
                _env_step,
                runner_state,
            )
            return runner_state[-2]

        rngs = jax.random.split(rng, num_episodes)
        total_rewards = jax.vmap(single_episode)(rngs)
        return total_rewards.mean()

    return eval

if __name__ == "__main__":
    rng = jax.random.PRNGKey(30)
    use_vmap = config.get("VMAP", True)

    lrs = jnp.linspace(config["LR"], config["LR"] * 10, 10)
    num_seeds = 4

    def run_tuner(config, base_rng, lrs, seeds):
        """
        Run LR tuning across multiple seeds and learning rates.

        Args:
            config: training config dict
            base_rng: PRNGKey for reproducibility
            lrs: jnp.ndarray of candidate learning rates [num_lrs]
            seeds: int, number of seeds
        Returns:
            metrics: pytree shaped [num_seeds, num_lrs, ...]
        """
        train = make_train(config)
        train_jit = jax.jit(train)

        rngs = jax.random.split(base_rng, seeds)

        if not use_vmap:
            # Sequential runs to avoid memory blow-up (no vmap/pmap)
            results = []
            for s in range(seeds):
                row = []
                for i in range(len(lrs)):
                    out = train_jit(rngs[s], lrs[i])
                    row.append(out)
                results.append(row)
            # Stack into [num_seeds, num_lrs, ...] to match vmap output structure
            rs_flat = [
                results[s][l]["runner_state"]
                for s in range(seeds)
                for l in range(len(lrs))
            ]
            ms_flat = [
                results[s][l]["metrics"]
                for s in range(seeds)
                for l in range(len(lrs))
            ]
            # runner_state is (train_state, env_state, last_obs, last_done, hstate, rng)
            def stack_reshape(seeds_n, lrs_n, *xs):
                stacked = jnp.stack(xs)
                return stacked.reshape((seeds_n, lrs_n) + stacked.shape[1:])

            # Stack params from train_states so agent_params[s,l] works for eval
            params_list = [rs_flat[i][0].params for i in range(seeds * len(lrs))]
            params_stacked = jax.tree_util.tree_map(
                lambda *ps: stack_reshape(seeds, len(lrs), *ps),
                *params_list,
            )
            first_train_state = rs_flat[0][0]
            train_state_stacked = first_train_state.replace(params=params_stacked)
            # Stack remaining runner_state leaves for consistent structure
            env_state_stacked = jax.tree_util.tree_map(
                lambda *xs: stack_reshape(seeds, len(lrs), *xs),
                *[r[1] for r in rs_flat],
            )
            rest_stacked = [
                jax.tree_util.tree_map(
                    lambda *xs: stack_reshape(seeds, len(lrs), *xs),
                    *[r[i] for r in rs_flat],
                )
                for i in range(2, len(rs_flat[0]))
            ]
            runner_state_stacked = (
                train_state_stacked,
                env_state_stacked,
                *rest_stacked,
            )
            metrics_stacked = jax.tree_util.tree_map(
                lambda *xs: stack_reshape(seeds, len(lrs), *xs),
                *ms_flat,
            )
            metrics = {
                "runner_state": runner_state_stacked,
                "metrics": metrics_stacked,
            }
        else:
            def run_one_seed(rng):
                sub_rngs = jax.random.split(rng, len(lrs))
                return jax.vmap(train_jit)(sub_rngs, lrs)

            if jax.local_device_count() > 1:
                rngs = rngs.reshape([jax.local_device_count(), -1, 2])
                run_one_seed = jax.vmap(run_one_seed)
                run_one_seed = jax.pmap(run_one_seed)
            else:
                run_one_seed = jax.vmap(run_one_seed)

            metrics = run_one_seed(rngs)
        return metrics

    metrics = run_tuner(config, rng, lrs, num_seeds)

    # metrics has shape [num_seeds, num_lrs, num_updates]
    returns = metrics["metrics"]["mean_training_return"]
    len_returns = returns.shape[-1]
    returns = jnp.nanmean(
        returns[..., int(len_returns * 0.95) :], axis=-1
    )  # compute return from the final 5% of training
    returns = returns.reshape([num_seeds, len(lrs)])

    mean_returns = returns.mean(axis=0)  # average over seeds per LR
    std_returns = returns.std(axis=0)

    for lr, m, s in zip(lrs, mean_returns, std_returns):
        print(
            f"LR={lr:.4f} -> training return (without eval policy) ={m:.4f} ± {s:.4f}"
        )

    best_idx = int(jnp.argmax(mean_returns))
    best_mean = float(mean_returns[best_idx])
    best_std = float(std_returns[best_idx])

    print(
        f"Best LR in training: {lrs[best_idx]:.4f} with avg training return {best_mean:.4f}"
    )
    agent_params = metrics["runner_state"][0].params

    # Evaluate all seeds of the best LR only (from training)
    best_lr_params = jax.tree_util.tree_map(
        lambda leaf: leaf[:, best_idx], agent_params
    )  # shape [num_seeds, ...]

    evaluate_policy = make_eval(config, 16)

    if not use_vmap:
        eval_rng = jax.random.PRNGKey(42)
        eval_returns = jnp.zeros(num_seeds)
        for s in range(num_seeds):
            rng_s, eval_rng = jax.random.split(eval_rng)
            params_s = jax.tree_util.tree_map(lambda leaf: leaf[s], best_lr_params)
            eval_returns = eval_returns.at[s].set(evaluate_policy(params_s, rng_s))
    else:
        if jax.local_device_count() > 1:
            evaluate_policy = jax.vmap(evaluate_policy, in_axes=(0, None))
            evaluate_policy = jax.pmap(evaluate_policy, in_axes=(0, None))
        else:
            evaluate_policy = jax.vmap(evaluate_policy, in_axes=(0, None))

        eval_rng = jax.random.PRNGKey(42)
        eval_returns = evaluate_policy(best_lr_params, eval_rng)

    best_eval_mean = float(eval_returns.mean())
    best_eval_std = float(eval_returns.std())

    print(
        f"Best LR (from training): {lrs[best_idx]:.4f} -> eval return {best_eval_mean:.4f} ± {best_eval_std:.4f} over {num_seeds} seeds"
    )
    return_out = {"return_mean": best_eval_mean, "return_std": best_eval_std}
    print(json.dumps(return_out))
