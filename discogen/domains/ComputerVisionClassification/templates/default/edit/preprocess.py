from torchvision import transforms
from typing import Dict
import torch


def build_transforms(config: Dict):
    """Build data augmentation and preprocessing transforms.
    Args:
        config (Dict): Configuration dictionary containing transform parameters.
            - name (str): Name of the dataset.
            - num_channels (int): Number of input channels.
            - raw_image_size (List[int]): Original size of the images. It might not be accurate for dataset with varying image sizes.
            - resized_image_size (List[int]): Size to resize the shorter side of the image.
            - input_image_size (List[int]): Size for cropping the image.
            - horizontal_flip_prob (float): Probability of applying horizontal flip.
            - norm_mean (list): Mean values for normalization.
            - norm_std (list): Standard deviation values for normalization.
            - image_key (str): Key for image data in the dataset.
            - label_key (str): Key for label data in the dataset.
    Returns:
        train_transforms, eval_transforms: Composed transforms for training and evaluation.
    """

    train_transforms = transforms.Compose(
        [
            ...
        ]
    )
    eval_transforms = transforms.Compose(
        [
            ...
        ]
    )

    return train_transforms, eval_transforms
