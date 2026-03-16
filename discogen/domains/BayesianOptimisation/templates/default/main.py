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


if __name__ == "__main__":

    # --- create the environment: define the objective function to maximise, and its parameter space ---
    obj_fn, domain = make_env()

    # --- list of maximum values for each seed (cannot vmap since samplers are mainly on CPU - have not been implemented for GPU)---
    maximum_values = []
    for seed in range(config['num_seeds']):
        config['seed'] = seed
        sampler = build_sampler(obs_dim=len(domain), seed=config['seed'], config=config)
        param_space = ParamSpace(domain, seed=config['seed'], sampler=sampler)
        key = jax.random.PRNGKey(config['seed'])

        # --- sample and evaluate initial points---
        obs_dict = param_space.sample_params(config['fixed_num_initial_samples'])
        obs_values = jax.vmap(obj_fn)(obs_dict)

        # --- initialise surrogate model, surrogate optimiser, acquisition function and acquisition function gradient optimiser ---
        surrogate = Surrogate(config, obs_dim=len(domain))
        surrogate_optimizer = build_surrogate_optimizer(config)
        acq_fn_gradient_optimizer = build_acq_fn_gradient_optimizer(config)
        remaining_budget = config['fixed_budget']

        # --- iteratively sample and evaluate points until budget is exhausted ---
        while remaining_budget > 0:

            # --- split key ---
            i = config['fixed_budget'] - remaining_budget
            key = jax.random.fold_in(key, i)

            # --- convert current observations to array ---
            obs_array = param_space.to_array(obs_dict)

            # --- fit surrogate model to current data ---
            init_surrogate_params = surrogate.init(key, X=obs_array, y=obs_values, method="neg_log_likelihood")
            surrogate_params = fit_posterior(y=obs_values, X=obs_array, surrogate=surrogate, init_surrogate_params=init_surrogate_params, optimizer=surrogate_optimizer, config=config)

            # --- sample candidate points for acquisition function optimization---
            candidate_samples = param_space.to_array(param_space.sample_params(n_samples=config['acq_sample_size']))

            # --- potentially optimize candidate points, and evaluate acquisition function values at these points---
            candidate_samples, acq_fn_vals = acq_fn_optimizer(candidate_samples=candidate_samples, acq_fn=acq_fn, acq_fn_gradient_optimizer=acq_fn_gradient_optimizer, X=obs_array, y=obs_values, surrogate=surrogate, surrogate_params=surrogate_params, config=config)

            # --- choose next query(ies) based on acquisition function values ---
            queries = next_queries(obs_samples=obs_array, obs_values=obs_values, candidate_samples=candidate_samples, candidate_acq_fn_vals=acq_fn_vals, remaining_budget=remaining_budget, config=config)

            # [--- ensure budget is not exceeded ---]
            if len(queries) > remaining_budget:
                queries = queries[:remaining_budget]
            remaining_budget -= len(queries)

            # --- convert next query(ies) back to parameter dictionary ---
            queries_dict = param_space.to_dict(queries)
            queries_values = jax.vmap(obj_fn)(queries_dict)

            # --- update observations ---
            obs_dict = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *[obs_dict, queries_dict])
            obs_values = jnp.concatenate([obs_values, queries_values.reshape(-1)])

        # --- append maximum value to list ---
        maximum_values.append(jnp.max(obs_values).item())
        print(f"Max sampled point for seed {seed}: {jnp.max(obs_values).item()}")

    # --- output maximum value at the end of optimization ---
    output = {"maximum_value_mean": jnp.array(maximum_values).mean().item(), "maximum_value_std": jnp.array(maximum_values).std().item()}
    print(json.dumps(output))
