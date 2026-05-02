import jax
import json
import jax.numpy as jnp
import os

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"
os.environ.pop("LD_LIBRARY_PATH", None)
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

from surrogate import Surrogate, fit_posterior
from next_queries import next_queries
from sampler import build_sampler
from domain import ParamSpace
from acq_fn import acq_fn
from surrogate_optimizer import build_surrogate_optimizer
from acq_optimizer import build_acq_fn_gradient_optimizer, acq_fn_optimizer
from make_env import make_env
from config import config
from functools import partial


def build_jit_step(surrogate: Surrogate,
                   surrogate_optimizer,
                   acq_fn_gradient_optimizer,
                   config: dict,
                   obs_dim: int):
    """
    Builds a JIT-compiled optimization step function.
    Args:
        surrogate: Surrogate model.
        surrogate_optimizer: Optimizer for the surrogate model.
        acq_fn_gradient_optimizer: Optimizer for acquisition function gradient optimization.
        config: Configuration dictionary.
        obs_dim: Number of observed dimensions.
    Returns:
        JIT-compiled step function.
    """

    @partial(jax.jit, static_argnums=())
    def jit_step(obs_array: jnp.ndarray,
                 obs_values: jnp.ndarray,
                 mask: jnp.ndarray,
                 surrogate_params: dict,
                 candidate_samples: jnp.ndarray,
                 key: jnp.ndarray):
        """
        Single Bayesian optimization step with fixed-shape arrays.
        Args:
            obs_array: Padded observation array [max_size, obs_dim].
            obs_values: Padded observation values [max_size].
            mask: Binary mask (1 for valid, 0 for padded) [max_size].
            surrogate_params: Current surrogate model parameters.
            candidate_samples: Candidate points for acquisition optimization [n_candidates, obs_dim].
            key: JAX random key.
        Returns:
            Tuple of (new_query, updated_surrogate_params, acq_fn_vals).
        """
        # --- fit surrogate model to current data ---
        surrogate_params = fit_posterior(
            y=obs_values,
            X=obs_array,
            mask=mask,
            surrogate=surrogate,
            init_surrogate_params=surrogate_params,
            optimizer=surrogate_optimizer,
            config=config
        )

        # --- compute acquisition function values for candidate points ---
        partial_acq_fn = partial(acq_fn, X=obs_array, y=obs_values, mask=mask, surrogate=surrogate, surrogate_params=surrogate_params, config=config)
        acq_vals = partial_acq_fn(candidate_samples)

        # --- find best N points and optimize acquisition function locally ---
        top_idxs = jnp.argsort(acq_vals)[-config['acq_top_n_samples']:]
        init_points = candidate_samples[top_idxs]

        # --- local gradient-based optimization of top candidates ---
        def gradient_optimize_single(sample_point: jnp.ndarray) -> jnp.ndarray:
            from acq_optimizer import gradient_acq_fn_optimizer
            return gradient_acq_fn_optimizer(
                sample_point[None, :],
                partial_acq_fn,
                acq_fn_gradient_optimizer,
                config=config
            ).squeeze(0)

        opt_candidate_samples = jax.vmap(gradient_optimize_single)(init_points)
        opt_acq_vals = partial_acq_fn(opt_candidate_samples)

        # --- concatenate and find best overall ---
        all_candidates = jnp.concatenate([candidate_samples, opt_candidate_samples], axis=0)
        all_acq_vals = jnp.concatenate([acq_vals, opt_acq_vals], axis=0)

        # --- select best query ---
        new_query = next_queries(
            obs_samples=obs_array,
            obs_values=obs_values,
            candidate_samples=all_candidates,
            candidate_acq_fn_vals=all_acq_vals,
            remaining_budget=config['fixed_budget'],  # or pass actual remaining
            config=config
        )

        return new_query, surrogate_params, all_acq_vals

    return jit_step


if __name__ == "__main__":

    # --- create the environment: define the objective function to maximise, and its parameter space ---
    obj_fn, domain = make_env()
    obs_dim = len(domain)

    # --- list of maximum values for each seed ---
    maximum_values = []
    for seed in range(config['num_seeds']):
        config['seed'] = seed
        sampler = build_sampler(obs_dim=obs_dim, seed=config['seed'], config=config)
        param_space = ParamSpace(domain, seed=config['seed'], sampler=sampler)
        key = jax.random.PRNGKey(config['seed'])

        # --- compute total size for pre-allocation ---
        total_size = config['fixed_num_initial_samples'] + config['fixed_budget']

        # --- pre-allocate padded arrays ---
        obs_array = jnp.zeros((total_size, obs_dim))
        obs_values = jnp.full((total_size,), -jnp.inf)  # use -inf for padding so max works correctly
        mask = jnp.zeros((total_size,))

        # --- sample and evaluate initial points ---
        init_obs_dict = param_space.sample_params(config['fixed_num_initial_samples'])
        init_obs_values = jax.vmap(obj_fn)(init_obs_dict)
        init_obs_array = param_space.to_array(init_obs_dict)

        # --- fill in initial observations ---
        n_init = config['fixed_num_initial_samples']
        obs_array = obs_array.at[:n_init].set(init_obs_array)
        obs_values = obs_values.at[:n_init].set(init_obs_values.ravel())
        mask = mask.at[:n_init].set(1.0)

        # --- initialise surrogate model, surrogate optimiser, acquisition function and acquisition function gradient optimiser ---
        surrogate = Surrogate(config, obs_dim=obs_dim)
        surrogate_optimizer = build_surrogate_optimizer(config)
        acq_fn_gradient_optimizer = build_acq_fn_gradient_optimizer(config)

        # --- initialise surrogate parameters ---
        init_key, key = jax.random.split(key)
        surrogate_params = surrogate.init(init_key, X=obs_array, y=obs_values, mask=mask, method="neg_log_likelihood")

        # --- build JIT-compiled step function ---
        jit_step = build_jit_step(
            surrogate=surrogate,
            surrogate_optimizer=surrogate_optimizer,
            acq_fn_gradient_optimizer=acq_fn_gradient_optimizer,
            config=config,
            obs_dim=obs_dim
        )

        # --- pre-sample all candidate points for the entire budget (Sobol is CPU-only) ---
        total_candidates_needed = config['fixed_budget'] * config['acq_sample_size']
        all_raw_candidates = sampler.random(total_candidates_needed)

        # --- current index for inserting new observations ---
        current_idx = n_init

        # --- iteratively sample and evaluate points until budget is exhausted ---
        for i in range(config['fixed_budget']):

            # --- split key ---
            key = jax.random.fold_in(key, i)

            # --- get pre-sampled candidates for this iteration ---
            start_idx = i * config['acq_sample_size']
            end_idx = start_idx + config['acq_sample_size']
            raw_candidates = all_raw_candidates[start_idx:end_idx]

            # --- transform raw [0,1] samples to parameter space (Python side, outside JIT) ---
            candidate_dict = {k: domain[k].inverse_transform(jnp.array(raw_candidates[:, j]))
                              for j, k in enumerate(domain)}
            candidate_samples = param_space.to_array(candidate_dict)

            # --- run JIT-compiled optimization step ---
            new_query, surrogate_params, _ = jit_step(
                obs_array,
                obs_values,
                mask,
                surrogate_params,
                candidate_samples,
                key
            )

            # --- convert next query back to parameter dictionary and evaluate ---
            query_dict = param_space.to_dict(new_query[:1])
            query_value = obj_fn(query_dict)

            # --- update padded arrays in-place ---
            obs_array = obs_array.at[current_idx].set(new_query[0])
            obs_values = obs_values.at[current_idx].set(query_value.ravel()[0])
            mask = mask.at[current_idx].set(1.0)
            current_idx += 1

        # --- append maximum value to list ---
        valid_obs_values = obs_values[:current_idx]
        maximum_values.append(jnp.max(valid_obs_values).item())
        print(f"Max sampled point for seed {seed}: {jnp.max(valid_obs_values).item()}")

    # --- output maximum value at the end of optimization ---
    output = {"maximum_value_mean": jnp.array(maximum_values).mean().item(), "maximum_value_std": jnp.array(maximum_values).std().item()}
    print(json.dumps(output))
