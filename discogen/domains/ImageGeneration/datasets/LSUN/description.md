DESCRIPTION
This LSUN church train dataset is a subset of the large-scale image dataset LSUN. LSUN is designed for advancing research in scene understanding, object recognition, and generative modeling. It contains millions of labeled images across multiple scene and object categories collected from the internet. LSUN is commonly used for training and evaluating deep learning models, particularly generative models such as GANs and diffusion models, due to its large size and diverse visual content.

OBSERVATION
Each observation is a full-color (RGB) image representing a specific scene or object category. Image resolutions vary across the dataset, and images are resized 256×256 during preprocessing (can be modified in config.py) 

CLASSES
LSUN chuch train contains 16 evenly split categories, each with a large number of images.

DATASET STRUCTURE
Training and validation set: ~120,000 images (split can be tuned in config.py, default is 80:20 split)
Evaluation set: ~6,300 images
