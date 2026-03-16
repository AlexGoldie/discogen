"""Main entry point for Neural Cellular Automata training and evaluation."""

import json

import jax
import jax.numpy as jnp
from flax import nnx

from config import config
from make_dataset import sample_state
from test_loss import eval_loss
from nca import NCA
from optimiser import create_optimizer
from perceive import create_perceive
from pool import Pool
from train import create_train_step
from update import create_update
from visualize import visualize


def create_nca(rngs: nnx.Rngs) -> NCA:
    """Create an NCA system with perceive and update modules."""
    perceive = create_perceive(rngs)
    update = create_update(rngs)
    return NCA(perceive, update)


def initialize_pool(key: jax.Array) -> Pool:
    """Initialize the state pool with seed states and targets."""
    pool_size = config["train"]["pool_size"]
    keys = jax.random.split(key, pool_size)
    states, targets = jax.vmap(sample_state, in_axes=(None, 0))(config, keys)
    return Pool.create({"state": states, "target": targets})

def evaluate(nca: NCA, key: jax.Array) -> dict:
    """Evaluate the trained NCA."""
    eval_config = config["eval"]
    num_seeds = eval_config["num_eval_seeds"]
    num_steps = eval_config["eval_num_steps"]

    keys = jax.random.split(key, num_seeds)
    state_init, targets = jax.vmap(sample_state, in_axes=(None, 0))(config, keys)

    state_axes = nnx.StateAxes({nnx.RngState: 0, ...: None})
    state_final = nnx.split_rngs(splits=num_seeds)(
        nnx.vmap(
            lambda nca, state: nca(state, num_steps=num_steps),
            in_axes=(state_axes, 0),
        )
    )(nca, state_init)

    losses = jax.vmap(eval_loss)(state_final, targets)
    return {
        "loss_mean": float(jnp.mean(losses)),
        "loss_std": float(jnp.std(losses)),
        "state_final": state_final,
        "targets": targets,
    }


def main():
    """Main training and evaluation loop."""
    train_config = config["train"]
    key = jax.random.key(train_config["seed"])
    rngs = nnx.Rngs(params=train_config["seed"], dropout=train_config["seed"])

    # Create NCA and optimizer
    nca = create_nca(rngs)
    optimizer = create_optimizer(nca)

    # Count parameters
    params = nnx.state(nca, nnx.Param)
    num_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"Number of parameters: {num_params}")

    # Initialize pool
    key, subkey = jax.random.split(key)
    pool = initialize_pool(subkey)
    print(f"Pool size: {pool.size}")

    # Create training step
    train_step = create_train_step(nca, optimizer)

    # Training loop
    num_train_steps = train_config["num_train_steps"]
    print(f"Training for {num_train_steps} steps...")
    losses = []
    print_interval = 128

    for step in range(num_train_steps):
        key, subkey = jax.random.split(key)
        loss, pool = train_step(nca, optimizer, pool, subkey)
        losses.append(float(loss))

        if step % print_interval == 0 or step == num_train_steps - 1:
            avg_loss = sum(losses[-print_interval:]) / len(losses[-print_interval:])
            print(f"Step {step}: avg_loss = {avg_loss:.6f}")

    # Evaluation
    print("\nEvaluating...")
    key, subkey = jax.random.split(key)
    eval_results = evaluate(nca, subkey)

    print(f"Evaluation loss: {eval_results['loss_mean']:.6f} +/- {eval_results['loss_std']:.6f}")
    eval_metrics = {"loss_mean": eval_results["loss_mean"], "loss_std": eval_results["loss_std"]}

    print("\nGenerating visualizations...")
    viz_metrics = visualize(eval_results["state_final"], eval_results["targets"], config)

    if viz_metrics:
        eval_metrics.update(viz_metrics)
        if "accuracy" in viz_metrics:
            print(f"Classification accuracy: {viz_metrics['accuracy']:.2%}")

    print(json.dumps(eval_metrics))


if __name__ == "__main__":
    main()
