DESCRIPTION
CIFAR-10-C is a robustness benchmark derived from CIFAR-10, designed to evaluate model performance under distribution shift. The dataset systematically applies 19 types of common corruptions to the original CIFAR-10 test images at 5 different severity levels, creating realistic perturbations that models may encounter in deployment.

OBSERVATION SPACE
Each observation is a 32×32×3 RGB image that has been corrupted, with the same class labels as CIFAR-10.

CORRUPTION TYPES
The dataset includes 19 corruption categories:

Noise: Gaussian, shot, impulse
Blur: defocus, glass, motion, zoom
Weather: snow, frost, fog, brightness
Digital: contrast, elastic, pixelate, JPEG compression, saturate, spatter

DATASET STRUCTURE
Total images: 950,000 (10,000 base images × 19 corruptions × 5 severity levels)
Severity levels: 1 (mild) to 5 (severe)
Same 10 classes as CIFAR-10
