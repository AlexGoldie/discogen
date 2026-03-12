DESCRIPTION
CIFAR-10 is a widely-used image classification benchmark consisting of 60,000 color images across 10 common object categories. Each image is 32×32 pixels in RGB format. The dataset provides a balanced training set and a carefully curated test set for evaluating classification performance on natural images.

OBSERVATION SPACE
Each observation is a 32×32×3 RGB image with pixel values in the range [0, 255], representing one of 10 object classes.

CLASSES
The dataset contains 10 classes:

0: airplane
1: automobile
2: bird
3: cat
4: deer
5: dog
6: frog
7: horse
8: ship
9: truck

DATASET STRUCTURE
Training set: 50,000 images (5,000 per class)
Test set: 10,000 images (1,000 per class)
The training data is divided into 5 batches of 10,000 images each
Test set contains exactly 1,000 randomly-selected images per class
