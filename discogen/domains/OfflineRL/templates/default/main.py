import json
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import jax
import jax.numpy as jnp
import numpy as np

from config import config
from make_env import make_env_and_datasets
from networks import Actor, Value
from optim import create_optimizer
from train import make_train_step, sample_actions
from datasets import Dataset
from evaluation import evaluate


def main():
    """Run offline RL training."""

    # --- Setup ---
    rng = jax.random.PRNGKey(config['seed'])
    np.random.seed(config['seed'])

    # --- Create environment and dataset ---
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        config['env_name'],
        action_clip_eps=config.get('action_clip_eps', 1e-5),
    )

    # Wrap dataset
    train_dataset = Dataset.create(**train_dataset)
    train_dataset.return_next_actions = True

    if val_dataset is not None:
        val_dataset = Dataset.create(**val_dataset)
        val_dataset.return_next_actions = True

    # --- Initialize networks ---
    example_batch = train_dataset.sample(1)
    obs_dim = example_batch['observations'].shape[-1]
    action_dim = example_batch['actions'].shape[-1]

    rng, actor_rng, critic_rng = jax.random.split(rng, 3)

    # Create actor network
    actor_def = Actor(
        hidden_dims=config['actor_hidden_dims'],
        action_dim=action_dim,
        layer_norm=config.get('actor_layer_norm', False),
        tanh_squash=config['tanh_squash'],
        const_std=True,
        final_fc_init_scale=config.get('actor_fc_scale', 0.01),
    )
    actor_params = actor_def.init(actor_rng, example_batch['observations'])['params']

    # Create critic network (ensemble)
    critic_def = Value(
        hidden_dims=config['value_hidden_dims'],
        layer_norm=config['layer_norm'],
        num_ensembles=2,
    )
    critic_params = critic_def.init(
        critic_rng,
        example_batch['observations'],
        example_batch['actions']
    )['params']

    # Create optimizers
    actor_opt = create_optimizer(config['lr'])
    critic_opt = create_optimizer(config['lr'])

    # Initialize train state
    train_state = {
        'actor_params': actor_params,
        'actor_opt_state': actor_opt.init(actor_params),
        'critic_params': critic_params,
        'critic_opt_state': critic_opt.init(critic_params),
        'target_actor_params': actor_params,  # Initialize targets = current
        'target_critic_params': critic_params,
        'step': 0,
    }

    # Create training step function
    train_step_fn = make_train_step(config, actor_def, critic_def, actor_opt, critic_opt)

    # --- Training loop ---
    print(f"Starting training for {config['offline_steps']} steps...")
    for step in range(1, config['offline_steps'] + 1):
        # Sample batch
        batch = train_dataset.sample(config['batch_size'])

        # Training step (delayed actor updates)
        rng, step_rng = jax.random.split(rng)
        full_update = (step % config['actor_freq'] == 0)
        train_state, train_info = train_step_fn(train_state, batch, step_rng, full_update)

        # Evaluation
        if config['eval_interval'] > 0 and step % config['eval_interval'] == 0:
            rng, eval_rng = jax.random.split(rng)
            eval_info, trajs, _ = evaluate(
                actor_def=actor_def,
                actor_params=train_state['actor_params'],
                env=eval_env,
                config=config,
                num_episodes=config['eval_episodes'],
                rng=eval_rng,
            )

            eval_metrics = {f'evaluation/{k}': v for k, v in eval_info.items()}

            print(f"Evaluation at step {step}: return={eval_info.get('episode.return', 0):.2f}, "
                  f"normalized={eval_info.get('episode.normalized_return', 0):.2f}")

    # --- Final evaluation ---
    print("\nRunning final evaluation...")
    rng, eval_rng = jax.random.split(rng)
    final_eval_info, trajs, _ = evaluate(
        actor_def=actor_def,
        actor_params=train_state['actor_params'],
        env=eval_env,
        config=config,
        num_episodes=config.get('final_eval_episodes', 50),
        rng=eval_rng,
    )

    print(f"\nFinal Results:")
    print(f"  Return: {final_eval_info.get('episode.return', 0):.2f}")
    print(f"  Normalized Return: {final_eval_info.get('episode.normalized_return', 0):.2f}")

    eval_keys = ["episode.return", "episode.length", "success"]

    return_out = {k: float(v) for k, v in final_eval_info.items() if k in eval_keys}
    print(json.dumps(return_out))

    return final_eval_info


if __name__ == "__main__":
    main()
