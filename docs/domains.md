# Task Domains
Here, we describe the various task domains available in DiscoGen. We expect this to continue to grow as our benchmark scales.

## BayesianOptimisation
The agent must maximise randomly sampled variables using Bayesian Optimisation.

### Modules
`acq_fn`, `acq_optimizer`, `sampler`, `next_queries`, `surrogate`, `surrogate_optimizer`

### Datasets
`Ackley1d`, `Ackley2d`, `Branin2d`, `Bukin2d`, `Cosine8d`, `DropWave2d`, `EggHolder2d`, `Griewank5d`, `Hartmann6d`, `HolderTable2d`, `Levy6d`

## BrainSpeechDetection
The agent is tasked with training a speech detector based on brain MEG signals.

### Modules
`loss`, `networks`, `optim`

### Datasets
`LibriBrainSherlock1`, `LibriBrainSherlock2`, `LibriBrainSherlock3`, `LibriBrainSherlock4`, `LibriBrainSherlock5`, `LibriBrainSherlock6`, `LibriBrainSherlock7`

## ComputerVisionClassification
The agent must train an image classifier for a range of different image classification datasets, of varying difficulty.

### Modules
`loss`, `networks`, `optim`, `preprocess`

### Datasets
`CIFAR10`, `CIFAR10C`, `CIFAR10LT`, `CIFAR100`, `FashionMNIST`, `MNIST`, `OxfordFlowers`, `StanfordCars`, `TinyImageNet`

## ContinualLearning
The agent must train a model on different non-stationary continual learning tasks.

### Modules
`optim`, `regularizer`, `replay`, `sampler`, `scheduler`

### Datasets
`PermutedMNIST`, `SplitCIFAR100`, `TinyImageNetSplit`

## GreenhouseGasPrediction
The agent must train a model to predict the changing concentrations of different greenhouse gases in the atmosphere.

### Modules
`data_processing`, `model`

### Datasets
`CH4`, `CO2`, `N2O`, `SF6`

## LanguageModelling
The agent must pre-train a language model on different small-scale pretraining datasets.

### Modules
`loss`, `networks`, `optim`

### Datasets
`LMFineWeb`, `OPCFineWebCode`, `OPCFineWebMath`, `TinyStories`

## ModelUnlearning
The agent must unlearn certain behaviours of a pretrained model while maintaining others.

### Modules
`loss`

### Datasets
`muse`, `tofu`, `wmdp_cyber`

### Models
`gemma-7b-it`, `Llama-2-7b-chat-hf`, `Llama-2-7b-hf`, `Llama-2-13b-hf`, `Llama-3.1-8b-Instruct`, `Llama-3.2-1B-Instruct`, `Llama-3.2-3B-Instruct`, `phi-1_5`, `Phi-3.5-mini-instruct`, `Qwen2.5-1.5B-Instruct`, `Qwen2.5-3B-Instruct`, `Qwen-2.5-7B-Instruct`

### Installation
Please note, after installing the ModelUnlearning `requirements.txt`, you must install `flash-attn`. Please use:
```bash
pip install flash-attn==2.6.3 --no-build-isolation
```

## OffPolicyRL
The agent must train a value-based RL agent in game environments.

### Modules
`config`, `networks`, `optim`, `policy`, `q_update`, `rb`, `train`

### Datasets
`MinAtar/Asterix`, `MinAtar/Breakout`, `MinAtar/Freeway`, `MinAtar/SpaceInvaders`

## OnPolicyMARL
The agent must train multiple RL agents in cooperative and competitive multi-agent RL environments.

### Modules
`loss`, `networks`, `optim`, `train`, `activation`, `targets`

### Datasets
`MABrax/Ant`, `MABrax/HalfCheetah`, `MABrax/Hopper`, `MABrax/Humanoid`, `MABrax/Walker`, `MPE/Spread`, `SMAX/2s3z`, `SMAX/3s_vs_5z`, `SMAX/3s5z`, `SMAX/3s5z_vs_3s6z`, `SMAX/5m_vs_6m`, `SMAX/6h_vs_8z`, `SMAX/10m_vs_11m`, `SMAX/27m_vs_30m`, `SMAX/smacv2_5_units`, `SMAX/smacv2_10_units`, `SMAX/smacv2_20_units`


## OnPolicyRL
The agent must train an on-policy RL agent in game and robotics environments.

### Modules
`loss`, `networks`, `optim`, `train`

### Datasets
`Brax/Ant`, `Brax/HalfCheetah`, `Brax/Hopper`, `Brax/Humanoid`, `Brax/Pusher`, `Brax/Reacher`, `Brax/Walker2D`, `Craftax/Craftax`, `Craftax/Craftax-Classic`, `MinAtar/Asterix`, `MinAtar/Breakout`, `MinAtar/Freeway`, `MinAtar/SpaceInvaders`

## UnsupervisedEnvironmentDesign
The agent must develop level sampling methods for an on-policy RL agent.

### Modules
`sample_levels`, `train_step`, `variable_config`

### Datasets
`Kinetix/Large`, `Kinetix/Medium`, `Kinetix/Small`, `Minigrid`
