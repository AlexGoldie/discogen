DESCRIPTION
CelebA-HQ is a high-quality version of the CelebFaces Attributes dataset, designed for research in computer vision tasks such as face generation, facial attribute recognition, and image synthesis. It contains high-resolution, carefully aligned celebrity face images with minimal compression artifacts. CelebA-HQ improves upon the original CelebA dataset by providing significantly higher image quality and consistent preprocessing, making it widely used in generative modeling research (e.g., GANs and diffusion models).

OBSERVATION
Each observation is a full-color (RGB) facial image with a fixed resolution of 256 × 256 pixels. Images are aligned and cropped so that facial landmarks are consistently positioned, which simplifies training for deep learning models.

CLASSES
The dataset contains 2 categories:
0: female
1: male

DATASET STRUCTURE
Training and validation set: ~27,000 images (split can be tuned in config.py, default is 80:20 split)
Evaluation set: ~3,000 images
