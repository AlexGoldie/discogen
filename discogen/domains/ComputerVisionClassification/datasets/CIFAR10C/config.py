num_classes = 10
num_channels = 3

config = {
    "train": {
        "seed": 42,
        "output_dir": "./results",
        "num_train_epochs": 30,
        "per_device_train_batch_size": 512,
        "per_device_eval_batch_size": 2048,
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
        "name": "CIFAR10C",
        "raw_image_size": (32, 32),
        "resized_image_size": (32, 32),
        "input_image_size": (32, 32),
        "norm_mean": [0.4914, 0.4822, 0.4465],  # STANDARD CIFAR-10 MEAN
        "norm_std": [0.2470, 0.2435, 0.2616],  # STANDARD CIFAR-10 STD
        "num_channels": num_channels,
        "num_classes": num_classes,
        "image_key": "image",
        "label_key": "label",
        # Available keys: "clean", "brightness", "contrast", "defocus_blur",
        #  "elastic_transform", "fog", "frost", "gaussian_blur", "gaussian_noise",
        #  "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur",
        #  "pixelate", "saturate", "shot_noise", "snow", "spatter", "speckle_noise",
        #  "zoom_blur"
        "train_keys": ["clean"],
        "test_keys": ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur",
                      "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur",
                      "pixelate", "saturate", "shot_noise", "snow", "spatter", "speckle_noise",
                      "zoom_blur"],
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
