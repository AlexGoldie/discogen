DESCRIPTION
Fashion-MNIST is a drop-in replacement for the classic MNIST dataset, consisting of grayscale images of fashion items. Designed as a more challenging alternative to handwritten digits, it maintains the same data format and split structure, making it ideal for benchmarking machine learning algorithms on a more complex task while keeping computational requirements modest.

OBSERVATION SPACE
Each observation is a 28×28 grayscale image with pixel values in the range [0, 255], where higher values represent darker pixels. Total of 784 pixels per image.

CLASSES
The dataset contains 10 evenly divided fashion item categories:
0: T-shirt/top
1: Trouser
2: Pullover
3: Dress
4: Coat
5: Sandal
6: Shirt
7: Sneaker
8: Bag
9: Ankle boot

DATASET STRUCTURE
Training and validation set: 60,000 images (split can be tuned in config.py, default is 80:20 split)
Evaluation set: 10,000 images
