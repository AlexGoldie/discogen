from torchvision import transforms


def build_transforms(image_size: int, train: bool):
    # Input is tensor already; ensure 3 channels and resize as needed
    ops = [
        transforms.Lambda(lambda x: x.expand(3, *x.shape[-2:])),
        transforms.Resize(image_size),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
    ]
    return transforms.Compose(ops)
