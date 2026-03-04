num_classes = 196
num_channels = 3

config = {
    "train": {
        "seed": 42,
        "output_dir": "./results",
        "num_train_epochs": 30,
        "per_device_train_batch_size": 64,
        "per_device_eval_batch_size": 64,
        "weight_decay": 0.01,
        "logging_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "disable_tqdm": True,
        "remove_unused_columns": False,
        "eval_strategy": "epoch",
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "early_stopping_patience": 5,
        "early_stopping_threshold": 0.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "max_grad_norm": 1.0,
    },
    "dataset": {
        "name": "StanfordCars",
        "raw_image_size": (),   # Uncertain numbers
        "resized_image_size": (256, 256),
        "input_image_size": (224, 224),
        "norm_mean": [0.485,0.456,0.406],  # ImageNet mean
        "norm_std": [0.229,0.224,0.225],  # ImageNet std
        "num_channels": num_channels,
        "num_classes": num_classes,
        "image_key": "image",
        "label_key": "label",
        # Available keys: "train", "test", "contrast", "gaussian_noise",
        # "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
        # "spatter"
        "train_keys": ["train"],
        "test_keys": ["test"],
        "horizontal_flip_prob": 0.5,
    },
    "model": {
        "num_classes": num_classes,
        "num_channels": num_channels,
        "dropout": 0.2,
        "channels": [64, 128, 256],
        "blocks_per_stage": 2,
        "use_residual": True,
    },
    "optimizer": {
        "learning_rate": 0.001,
        "weight_decay": 0.01,
    },
}
