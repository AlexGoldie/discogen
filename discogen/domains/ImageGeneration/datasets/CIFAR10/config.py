num_classes = 10
num_channels = 3
transformed_image_size = 32
timesteps = 1000
image_key = "img"
label_key = "label"
dataset = "CIFAR10"

config = {
    "meta": {"name": dataset, "seed": 1},
    "train": {
        "validation_split": 0.2,
        "steps": 200_000,
        "per_device_batch_size": 512,
        "num_classes": num_classes,
        "optimizer": {"lr": 2e-4, "weight_decay": 1e-4},
        "early_stopping": {"n_fake_samples": 10_000, "interval": 10_000, "patience": 5},
    },
    "eval": {
        "per_device_batch_size": 1024,
        "num_classes": num_classes,
        "dataset_name": dataset,
        "channels": num_channels,
        "n_real_samples": 50_000,
        "n_fake_samples": 10_000,
    },
    "preprocess": {
        "transformed_image_size": (transformed_image_size, transformed_image_size),
        "image_key": image_key,
        "label_key": label_key,
        "channels": num_channels,
    },
    "model": {
        "dim": 64,
        "dim_mults": (1, 2, 4, 8),
        "num_classes": num_classes,
        "channels": num_channels,
        "timesteps": timesteps,
        "image_size": transformed_image_size,
        "dropout": 0.1,
    },
}
