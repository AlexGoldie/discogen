config = {
    "nca": {
        "channel_size": 32,
        "cell_dropout_rate": 0.5,
        "use_alive_masking": False,
        "alive_threshold": 0.1,
        "preserve_channels": 0,
    },
    "target": {
        "target_size": 28,
        "pad_width": 0,
    },
    "train": {
        "seed": 0,
        "num_steps": 128,
        "pool_size": 1024,
        "batch_size": 8,
        "num_train_steps": 8192,
    },
    "optimizer": {
        "learning_rate": 2e-3,
        "lr_decay_factor": 0.1,
        "lr_decay_steps": 2000,
        "max_grad_norm": 1.0,
    },
    "eval": {
        "num_eval_seeds": 16,
        "eval_num_steps": 128,
    },
}
