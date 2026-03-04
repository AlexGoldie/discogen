config = {
    "train": {
        "seed": 42,
        "output_dir": "./results",
        "max_epochs": 15,
        "batch_size": 512,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "log_every_n_steps": 1,
        "early_stopping": {
            "min_delta": 0.00,
            "patience": 10,
        }
    },
    "model": {
        "input_dim": 23,
        "model_dim": 100,
        "kernel_size": 3,
        "padding": 1,
        "dropout_rate": 0.5,
        "lstm_layers": 2,
        "batch_norm": False,
        "bi_directional": False,
    },
    "dataset": {
        "name": "LibriBrainSherlock5",
        "batch_size": 32,
        "num_workers": 2,
        "tmin": 0.0,
        "tmax": 0.8,
        "seed": 784,
    },
    "loss": {

    },
    "optimizer": {
        "learning_rate": 0.001,
        "weight_decay": 0.01,
    },
}
