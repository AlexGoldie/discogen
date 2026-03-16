from torchvision import transforms


def build_transforms(image_size: int, train: bool):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    ops = [transforms.Resize(image_size)]
    if train and image_size >= 64:
        ops += [transforms.RandomHorizontalFlip()]
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)
