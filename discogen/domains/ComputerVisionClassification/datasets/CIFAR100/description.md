DESCRIPTION
CIFAR-100 is an extension of CIFAR-10 with 100 fine-grained classes, providing a more challenging classification benchmark. Each image is 32×32 pixels in RGB format. The classes are grouped into 20 superclasses for hierarchical classification tasks, with 5 fine classes per superclass.

OBSERVATION SPACE
Each observation is a 32×32×3 RGB image with pixel values in the range [0, 255], representing one of 100 classes.

CLASSES
The dataset contains 100 fine-grained classes organized into 20 superclasses:

Aquatic mammals: beaver, dolphin, otter, seal, whale
Fish: aquarium fish, flatfish, ray, shark, trout
Flowers: orchids, poppies, roses, sunflowers, tulips
Food containers: bottles, bowls, cans, cups, plates
Fruit and vegetables: apples, mushrooms, oranges, pears, sweet peppers
Household electrical devices: clock, keyboard, lamp, telephone, television
Household furniture: bed, chair, couch, table, wardrobe
Insects: bee, beetle, butterfly, caterpillar, cockroach
Large carnivores: bear, leopard, lion, tiger, wolf
Large man-made outdoor things: bridge, castle, house, road, skyscraper
Large natural outdoor scenes: cloud, forest, mountain, plain, sea
Large omnivores and herbivores: camel, cattle, chimpanzee, elephant, kangaroo
Medium-sized mammals: fox, porcupine, possum, raccoon, skunk
Non-insect invertebrates: crab, lobster, snail, spider, worm
People: baby, boy, girl, man, woman
Reptiles: crocodile, dinosaur, lizard, snake, turtle
Small mammals: hamster, mouse, rabbit, shrew, squirrel
Trees: maple, oak, palm, pine, willow
Vehicles 1: bicycle, bus, motorcycle, pickup truck, train
Vehicles 2: lawn mower, rocket, streetcar, tank, tractor

DATASET STRUCTURE
Training set: 50,000 images (500 per class)
Test set: 10,000 images (100 per class)
