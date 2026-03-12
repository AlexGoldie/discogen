num_classes = 1000
num_channels = 3
transformed_image_size = 32
timesteps = 1000
image_key = "image"
label_key = "label"
dataset = "ImageNet-1k"

config = {
    "train": {
        "seed": 3,
        "epoch": 64,
        "per_device_batch_size": 512,
        "num_classes": num_classes,
        "optimizer": {
            "lr": 1e-4, 
            "weight_decay": 1e-4,
        },
        "image_key": image_key,
        "label_key": label_key,
    },
    "eval": {
        "per_device_batch_size": 1024,
        "num_classes": num_classes,
        "dataset_name": dataset,
        "channels": num_channels,
        "n_real_samples": 50_000,
        "n_fake_samples": 10_000,
    },
    "dataset": {
        "name": dataset,
        "transformed_image_size": (transformed_image_size, transformed_image_size),
        "channels": num_channels,
        "num_classes": num_classes,
        "image_key": image_key,
        "label_key": label_key,
    },
    "model": {
        "dim": 64,
        "dim_mults": (1, 2, 4, 8),
        "num_classes": num_classes,
        "channels": num_channels,
        "dropout": 0.2,
        "timesteps": timesteps,
        "image_size": transformed_image_size,
    },
}
