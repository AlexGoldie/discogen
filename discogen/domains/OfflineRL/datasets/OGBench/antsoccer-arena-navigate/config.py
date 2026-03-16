"""Configuration for antsoccer-arena-navigate-singletask-task1-v0."""

config = {
    # --- Environment ---
    'env_name': 'antsoccer-arena-navigate-singletask-task1-v0',
    'seed': 0,

    # --- Training ---
    'offline_steps': 1_000_000,
    'batch_size': 256,
    'lr': 3e-4,

    # --- Network architecture ---
    'actor_hidden_dims': (512, 512, 512, 512),
    'value_hidden_dims': (512, 512, 512, 512),
    'layer_norm': True,
    'actor_layer_norm': False,
    'tanh_squash': True,
    'actor_fc_scale': 0.01,

    'discount': 0.995,
    'tau': 0.005,
    'actor_freq': 2,
    'actor_noise': 0.2,
    'actor_noise_clip': 0.5,
    'alpha_actor': 0.01,
    'alpha_critic': 0.01,

    # --- Evaluation ---
    'eval_interval': 100_000,
    'eval_episodes': 50,
    'final_eval_episodes': 100,

    # --- Data processing ---
    'action_clip_eps': 1e-5,
}
