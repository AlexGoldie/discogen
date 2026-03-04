DESCRIPTION
PermutedMNIST is a continual learning benchmark that creates multiple tasks from the MNIST dataset by applying fixed, deterministic pixel permutations. Each task uses the same 10-digit classes (0-9) but reorders the 28×28 pixel grid according to a unique permutation derived from a base seed. This generates a sequence of tasks with identical label semantics but different input distributions, making it ideal for evaluating continual learning algorithms' ability to handle distribution shift without forgetting previously learned knowledge.

TASK STRUCTURE
Base dataset: MNIST (28×28 grayscale images)
Number of tasks: Configurable (typically 10-20 tasks)
Classes per task: 10 (digits 0-9, consistent across all tasks)
Permutation: Unique, deterministic pixel reordering per task based on seed
Task identity: Not provided at test time (unless explicitly stated)

OBSERVATION SPACE
Each observation is a permuted 28×28 grayscale image (784 pixels total) with pixel values in the range [0, 255] or normalized to [0, 1].

EVALUATION
The benchmark tests continual learning through several metrics:

Per-task accuracy: Classification accuracy on each individual task after training
Average accuracy: Mean accuracy across all tasks after sequential training
Forgetting: Degradation in performance on earlier tasks after learning new ones
Forward transfer: Ability to leverage prior knowledge for new tasks
Backward transfer: Improvement on past tasks from learning new ones

CHARACTERISTICS
Same label space across all tasks (no class incremental component)
Distribution shift through input transformation only
No correlation between task permutations (unless designed)
Tests pure robustness to catastrophic forgetting
Domain-incremental continual learning scenario
