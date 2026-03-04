DESCRIPTION
CIFAR-10-LT is a long-tailed variant of CIFAR-10 designed to study learning under class imbalance. The training set exhibits exponential decay in samples per class, simulating real-world scenarios where some categories have significantly more data than others. The test set remains balanced to evaluate performance across all classes.

OBSERVATION SPACE
Each observation is a 32×32×3 RGB image with pixel values in the range [0, 255].

CLASSES
The same 10 classes as CIFAR-10, but with imbalanced training distribution.

DATASET STRUCTURE
Training set: fewer than 50,000 images with exponentially decreasing samples per class
Test set: 10,000 balanced images (1,000 per class)
Imbalance ratio varies based on configuration (typical ratios: 10, 50, or 100)
