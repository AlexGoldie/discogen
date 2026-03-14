# DiscoGen

<!-- [![Release](https://img.shields.io/github/v/release/AlexGoldie/discogen)](https://img.shields.io/github/v/release/AlexGoldie/discogen)
[![Build status](https://img.shields.io/github/actions/workflow/status/AlexGoldie/discogen/main.yml?branch=main)](https://github.com/AlexGoldie/discogen/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/AlexGoldie/discogen/branch/main/graph/badge.svg)](https://codecov.io/gh/AlexGoldie/discogen)
[![Commit activity](https://img.shields.io/github/commit-activity/m/AlexGoldie/discogen)](https://img.shields.io/github/commit-activity/m/AlexGoldie/discogen)
[![License](https://img.shields.io/github/license/AlexGoldie/discogen)](https://img.shields.io/github/license/AlexGoldie/discogen) -->

**DiscoGen** is a procedural generator of algorithm discovery tasks in machine learning.

## What is DiscoGen?

DiscoGen is a procedural generator of algorithm discovery tasks in machine learning, which can be used for optimising automated algorithm discovery and AI scientist systems. DiscoGen has a modular setup, an emphasis on discovering algorithms that transfer, and a huge diversity of tasks (over a billion)! We hope DiscoGen helps drive the frontier of research in algorithm discovery by providing a large-scale, open-ended landscape for optimising and evaluation AI research agents!

### Key Features

- **Sampling Tasks**: Continuously sample random, different tasks.
- **Modular Architecture**: Break down ML algorithms into composable components
- **Multiple Domains**: Support for reinforcement learning, language modeling, computer vision, Bayesian optimization, and more
- **Flexible Configuration**: Easy switching between baseline and experimental implementations
- **LLM-Ready**: Designed for automated algorithm discovery using AI agents
- **Extensible**: Simple framework for adding new tasks and domains
- **Principled Evaluation**: Algorithms are evaluated on held-out *meta-test* datasets to prevent overfitting.

## Quick Start

### Installation

Install from source:

```bash
git clone git@github.com:AlexGoldie/discogen.git
cd discogen
make install
```
or install from pip:
```bash
pip install discogen
```
### Basic Usage

List available domains:
```bash
uv run discogen get-domains
```

Create a full task-domain codebase (with baseline implementations):
```bash
uv run discogen create-task --task-domain OnPolicyRL
```

Create an example task for algorithm discovery:
```bash
uv run discogen create-task --task-domain OnPolicyRL --example
```

See the full [Usage Guide](usage.md) for detailed instructions.

## Available Domains

DiscoGen currently supports the following task domains:

- **[OnPolicyRL](domains.md#onpolicyrl)**: On-policy reinforcement learning (PPO-style algorithms)
- **[OffPolicyRL](domains.md#offpolicyrl)**: Off-policy reinforcement learning (DQN-style algorithms)
- **[LanguageModelling](domains.md#languagemodelling)**: Pre-training language models
- **[ComputerVisionClassification](domains.md#computervisionclassification)**: Image classification tasks
- **[BayesianOptimisation](domains.md#bayesianoptimisation)**: Black-box optimization
- **[BrainSpeechDetection](domains.md#brainspeechdetection)**: Neural signal analysis
- **[ModelUnlearning](domains.md#modelunlearning)**: LLM unlearning tasks
- **[UnsupervisedEnvironmentDesign](domains.md#unsupervisedenvironmentdesign)**: Environment curriculum learning
- **[ContinualLearning](domains.md#continuallearning)**: Learning under non-stationarity
- **[GreenhouseGasPrediction](domains.md#greenhousegasprediction)**: Predicting atmospheric greenhouse gas concentrations
- **[OnPolicyMARL](domains.md#onpolicymarl)**: On-policy Multi-Agent reinforcement learning (IPPO-style algorithms)
- **[OfflineRL](domains.md#offlinerl)**: Offline reinforcement learning from fixed datasets.
- **[TrajectoryPrediction](domains.md#trajectoryprediction)**: Trajectory prediction of vehicles for self-driving.
- **[NeuralCellularAutomata](domains.md#neuralcellularautomata)**: Evolving neural cellular automata for open-ended tasks.

See the [Domains](domains.md) page for detailed information about each domain.

## How It Works

### 1. Modular Components

Each task domain is decomposed into modules. For example, OnPolicyRL includes:
- `loss.py`: Objective function (e.g., PPO loss)
- `networks.py`: Neural network architectures
- `optim.py`: Optimization algorithms
- `train.py`: Training loop logic

### 2. Base and Edit Implementations

Each module has two versions:
- **Base**: Fully implemented, tested baseline
- **Edit**: Template with function signatures for customization

### 3. Configuration-Driven

Control which modules use baseline vs. custom implementations via YAML config:

```yaml
change_optim: true   # Use custom optimizer
change_loss: false   # Use baseline loss
change_networks: false
change_train: false
```

### 4. Task Generation

DiscoGen builds the configuration into a complete, runnable task in `task_src/`:

```bash
discogen create-task --task-domain OnPolicyRL # Build the default configuration (no editable files)
cd task_src/OnPolicyRL
python run_main.py # Run OnPolicyRL training for *all* enviornments.
```

### 5. Sample Random Tasks

```bash
discogen sample-task-config --config-dest task_config.yaml # Saves a random task config and prints the task domain
discogen create-task --task-domain <task_domain> --config-path task_config.yaml # Create the random task
cd task_src
# Run algorithm discovery agent for created task
discogen create-task --task-domain <task_domain> --config-path task_config.yaml --test # Create the meta-test part of the task
cd task_src
python run_main.py # Evaluate the agent's code on held out datasets
```

## Documentation

### For Users
- **[Usage Guide](usage.md)**: CLI commands, Python API, and workflows
- **[Domains](domains.md)**: Available task domains and their modules

### For Contributors
- **[Contributing Overview](how_to/overview.md)**: How to add new domains to DiscoGen
- **[Dataset Integration](how_to/dataset_integration.md)**: Adding new datasets to tasks

## Example Use Cases

### Optimising Algorithm Discovery Agents

Use DiscoGen to sample algorithm discovery tasks for optimising agents:
1. Randomly sample new task
2. Algorithm Discovery Agent iterates on implementations for editable modules
3. Measure meta-train (in-distribution) algorithm performance
4. Measure meta-test (new, out-of-distribution dataset) algorithm performance and update the agent
5. Repeat to optimise the agent for developing generalist algorithms

### Evaluating Algorithm Discovery Agents

Use the DiscoBench task suite to benchmark algorithm discovery agents:
1. Loop through all DiscoBench tasks
2. Generate meta-train part of DiscoBench task, and allow agent to develop implementations for editable modules
3. Measure meta-train (in-distribution) algorithm performance
4. Generate meta-test part of DiscoBench task
5. Evaluate discovered algorithms for generalisation

## Project Structure

```
discogen/
├── discobench_configs/   # All DiscoBench task configurations
├── domains/              # Task domain implementations
│   ├── OnPolicyRL/
│   ├── LanguageModelling/
│   └── ...
├── utils/              # Core utilities
├── create_task.py      # Task generation logic
├── create_config.py    # Configuration utilities
└── cli.py              # Command-line interface

task_src/               # Generated task files (after running create-task)
```

## Contributing

We welcome contributions! DiscoGen grows stronger with more domains, datasets and evaluation types.

- Found a bug? [Open an issue](https://github.com/AlexGoldie/discogen/issues)
- Want to add a task? See the [Contributing Guide](how_to/overview.md)
- Adding datasets? Check the [Dataset Integration Guide](how_to/dataset_integration.md)

## Citation

If you use DiscoGen in your research, please cite:

```bibtex
@article{goldie2026discogen,
  title={DiscoGen: Procedural Generation of Algorithm Discovery Tasks in Machine Learning},
  author={Alexander D. Goldie and Zilin Wang and Adrian Hayler and Deepak Nathani and Edan Toledo and Ken Thampiratwong and Aleksandra Kalisz and Michael Beukman and Alistair Letcher and Shashank Reddy and Clarisse Wibault and Theo Wolf and Charles O'Neill and Uljad Berdica and Nicholas Roberts and Saeed Rahmani and Hannah Erlebach and Roberta Raileanu and Shimon Whiteson and Jakob N. Foerster},
  year={2026}
}
```

## Links

- **GitHub Repository**: [https://github.com/AlexGoldie/discogen](https://github.com/AlexGoldie/discogen)
- **Documentation**: [https://AlexGoldie.github.io/discogen](https://AlexGoldie.github.io/discogen)
- **Blog**: [https://alexgoldie.github.io/discogen-blog/](https://alexgoldie.github.io/discogen-blog/)
- **PyPI Package**: Coming soon

## License

This project is licensed under the terms specified in the [LICENSE](https://github.com/AlexGoldie/discogen/blob/main/LICENSE) file.
