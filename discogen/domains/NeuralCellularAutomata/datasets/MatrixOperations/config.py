config = {
    "nca": {
        "channel_size": 32,
        "cell_dropout_rate": 0.5,
        "use_alive_masking": False,
        "alive_threshold": 0.1,
        "preserve_channels": 6,  # channels 26-31: 26: A, 27: B, 28-31: operation one-hot
    },
    "matrix": {
        "size": 8,
        "input_channels": [26, 27],  # A, B
        "output_channel": 0,  # output matrix
        "operations": ["transpose", "negate", "add", "multiply"],
    },
    "target": {
        "target_size": 8,
        "pad_width": 0,
    },
    "train": {
        "seed": 0,
        "num_steps": 64,
        "pool_size": 1024,
        "batch_size": 16,
        "num_train_steps": 8192,
    },
    "optimizer": {
        "learning_rate": 1e-3,
        "lr_decay_factor": 0.1,
        "lr_decay_steps": 4000,
        "max_grad_norm": 1.0,
    },
    "eval": {
        "num_eval_seeds": 64,
        "eval_num_steps": 64,
    },
}
