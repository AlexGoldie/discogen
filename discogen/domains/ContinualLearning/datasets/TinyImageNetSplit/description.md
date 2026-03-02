DESCRIPTION
TinyImageNetSplit is a class-incremental continual learning benchmark constructed from Tiny ImageNet by partitioning its 200 classes into disjoint task-specific groups. With higher resolution (64×64) and more classes than CIFAR-based benchmarks, it provides a more challenging testbed for continual learning algorithms operating on natural images with greater visual complexity and diversity.

TASK STRUCTURE
Base dataset: Tiny ImageNet (64×64 RGB images, 200 classes)
Number of tasks: Configurable (commonly 10 tasks × 20 classes each)
Classes per task: Disjoint subsets (e.g., 10, 20, or 40 classes per task)
Image resolution: 64×64×3 RGB
Total classes: 200 across all tasks

OBSERVATION SPACE
Each observation is a 64×64×3 RGB image representing one of 200 object classes from ImageNet, divided among sequential tasks.

EVALUATION
Standard continual learning evaluation includes:

Per-task accuracy: Classification accuracy on each task's class subset
Average accuracy: Mean performance across all learned classes
Forgetting metric: Degradation on earlier tasks after learning new ones
Final accuracy: Performance on all 200 classes after sequential training

CHARACTERISTICS
Class-incremental learning with disjoint class sets
Higher resolution than CIFAR-based benchmarks (64×64 vs 32×32)
Larger number of classes (200 total)
Greater visual complexity and within-class variation
Tests scalability of continual learning to more realistic scenarios
Unified classification across all tasks at test time
