DESCRIPTION
MNIST is the foundational handwritten digit recognition dataset, consisting of grayscale images extracted from two NIST databases. The digits were written by Census Bureau employees and high school students, providing natural variation in writing styles. Despite its age, MNIST remains a standard benchmark for validating machine learning implementations.

OBSERVATION SPACE
Each observation is a 28×28 grayscale image representing a handwritten digit (0-9), with pixel values typically normalized to [0, 1] or kept in [0, 255] range.

CLASSES
The dataset contains 10 digit classes (0-9), with one class per digit.

DATASET STRUCTURE
Training set: 60,000 images (6,000 per class)
Test set: 10,000 images (1,000 per class)
Balanced distribution: each class contains exactly 7,000 images total
Equal representation of Census Bureau employees and high school students in both splits
