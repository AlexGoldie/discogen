import jax
from config import config
from train import make_train
import os
from config import config
from make_env import make_env
from networks import QNetwork
from policy import exploit
import json
from functools import partial
import jax.numpy as jnp

def eval(params, rng, config, num_episodes):
    env, env_params, _ = make_env()

    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

    network = QNetwork(action_dim=env.action_space(env_params).n, width=config["WIDTH"], depth=config["DEPTH"])

    rng, rng_init = jax.random.split(rng)
    init_obs, env_state = vmap_reset(num_episodes)(rng_init)

    max_steps = env_params.max_steps_in_episode

    def _env_step(runner_state, unused):
        last_obs, env_state, rng = runner_state

        rng, rng_a, rng_s = jax.random.split(rng, 3)
        q_val = network.apply(params, last_obs)
        action = exploit(rng_a, q_val, 0, config)

        obs, env_state, reward, done, info = vmap_step(num_episodes)(
            rng_s, env_state, action)

        runner_state = (obs, env_state, rng)
        metrics = info
        return runner_state, metrics


    runner_state = (init_obs, env_state, rng)
    runner_state, metrics = jax.lax.scan(
        _env_step, runner_state, None, max_steps
    )
    return {"metrics": metrics}

if __name__ == "__main__":

    num_seeds = 8
    rng = jax.random.PRNGKey(42)
    rngs = jax.random.split(rng, num_seeds)
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    results = jax.block_until_ready(train_vjit(rngs))

    returns = results['metrics']['returns']
    params = results['runner_state'][0].params

    rng = jax.random.PRNGKey(42)
    rngs = jax.random.split(rng, num_seeds)
    num_episodes = 100

    eval_fn = partial(eval, config=config, num_episodes=num_episodes)

    vmap_eval = jax.vmap(eval_fn, in_axes=(0, 0))

    all_eval_results = jax.jit(vmap_eval)(params, rngs)

    def get_seed_mean_return(eval_metrics_one_seed):

        mask = eval_metrics_one_seed['returned_episode']
        returns = eval_metrics_one_seed['returned_episode_returns']

        valid_returns = jnp.where(mask, returns.astype(float), jnp.nan)

        mean_return = jnp.nanmean(valid_returns)

        return jnp.nan_to_num(mean_return, nan=0.0)

    seed_mean_returns = jax.vmap(get_seed_mean_return)(all_eval_results['metrics'])

    agg_return_mean = seed_mean_returns.mean()
    agg_return_std = seed_mean_returns.std()
    return_out = {"return_mean": float(agg_return_mean), "return_std": float(agg_return_std)}
    print(json.dumps(return_out))
