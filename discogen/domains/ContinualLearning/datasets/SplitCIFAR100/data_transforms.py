from torchvision import transforms


def build_transforms(image_size: int, train: bool):
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    ops = [transforms.Resize(image_size)]
    if train and image_size >= 32:
        ops += [transforms.RandomHorizontalFlip()]
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)
