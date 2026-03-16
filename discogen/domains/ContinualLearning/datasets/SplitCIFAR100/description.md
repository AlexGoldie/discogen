DESCRIPTION
SplitCIFAR100 is a class-incremental continual learning benchmark that partitions the 100 classes of CIFAR-100 into disjoint subsets, with each subset forming a separate task. The class assignment to tasks is deterministic based on a seed, creating a sequence where the model must learn new classes without forgetting previously learned ones. This benchmark tests class-incremental learning where both the input distribution and the set of relevant classes change over time.

TASK STRUCTURE
Base dataset: CIFAR-100 (32×32 RGB images, 100 classes)
Number of tasks: Configurable (commonly 10 tasks × 10 classes each)
Classes per task: Disjoint subsets (e.g., 10, 20, or 50 classes per task)
Class order: Deterministic based on seed
Total classes: 100 across all tasks

OBSERVATION SPACE
Each observation is a 32×32×3 RGB image from CIFAR-100, with classes divided among tasks.

EVALUATION
Standard continual learning metrics include:

Per-task accuracy: Performance on each task's classes after training
Average accuracy: Mean accuracy across all seen classes
Forgetting: Performance drop on earlier tasks' classes
Incremental accuracy curve: Accuracy trajectory as tasks are learned sequentially

CHARACTERISTICS
Class-incremental scenario (new classes introduced per task)
Disjoint class sets prevent overlap between tasks
Fixed input distribution (CIFAR-100) across tasks
Tests ability to expand classifier without catastrophic forgetting
No task boundaries provided at test time (unified classification)
