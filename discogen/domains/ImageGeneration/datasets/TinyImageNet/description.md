DESCRIPTION
Tiny ImageNet is a subset of the ImageNet dataset designed for faster experimentation while maintaining classification complexity. It consists of 200 classes from the original ImageNet, with images downsampled to 64×64 pixels. The dataset provides a middle ground between small-scale datasets like CIFAR and full-scale ImageNet.

OBSERVATION SPACE
Each observation is a 64×64×3 RGB color image representing one of 200 object classes.

CLASSES
The dataset contains 200 evenly distributed classes selected from ImageNet, covering diverse object categories.

DATASET STRUCTURE
Training and Validation set: 100,000 images (split can be tuned in config.py, default is 80:20 split)
Evaluation set: 10,000 images
