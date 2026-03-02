<h1 align="center">DiscoBench: An Open-Ended Benchmark For Algorithm Discovery</h1>

<p align="center">
  <img src="docs/assets/discobench.png" alt="DiscoBench Logo" width="100%">
</p>

This repository contains code for the DiscoBench modular benchmark for automated algorithm discovery.

- **Github repository**: <https://github.com/AlexGoldie/discobench/>
- **Documentation**: <https://AlexGoldie.github.io/discobench/>
- **Blog**: <https://alexgoldie.github.io/discobench-blog/>

## Quick Start

Install DiscoBench:

```bash
pip install discobench
```

List available domains:
```bash
discobench get-domains
```

Create a task:
```bash
discobench create-task --task-domain OnPolicyRL
```

See the [full documentation](https://AlexGoldie.github.io/discobench/) for detailed usage. Please note that each task_domain has its own set of requirements which may need to be installed.

Every domain includes references in `discobench/domains/<task_domain>/utils/_reference.txt`.

## Task Domains

| Task Domain | Modules | Datasets | Description |
| :--- | :--- | :--- | :--- |
| **BayesianOptimisation** | acq_fn, acq_optimizer, sampler, next_queries, surrogate, surrogate_optimizer | Ackley1D, Ackley2D, Branin2d, Bukin2d, Cosine8d, DropWave2d, EggHolder2d, Griewank5d, Hartmann6d, HolderTable2d, Levy6d. | Optimization of black-box functions using surrogate models to find global minima/maxima. |
| **BrainSpeechDetection** | loss, networks, optim | 7 LibriBrainSherlock tasks. | Detecting speech features directly from brain activity data. |
| **ComputerVisionClassification** | loss, networks, optim, preprocess | CIFAR10, CIFAR10C, CIFAR10LT, CIFAR100, FashionMNIST, MNIST, OxfordFlowers, StanfordCars, TinyImageNet. | Image classification on a range of datasets. |
| **ContinualLearning** | optim, regularizer, replay, sampler, scheduler | PermutedMNIST, SplitCIFAR100, TinyImageNetSplit. | Training a model on continually changing data, such that it can adapt to new data without losing old capabilities. |
| **GreenhouseGasPrediction** | data_processing, model | 4 Mauna Loa Time-series (CO2, N2O, SF6, CH4). | Time-series forecasting of atmospheric greenhouse gas concentrations. |
| **LanguageModelling** | loss, networks, optim | OPCFineWebCode, OPCFineWebMath, LMFineWeb, TinyStories. | Training transformer-based models on code, mathematics, and narrative text. |
| **ModelUnlearning** | loss | MUSE, TOFU, WMDP_Cyber. | Fine-tuning pretrained models to remove specific knowledge or data points while retaining others. |
| **OffPolicyRL** | q_update, policy, networks, optim, rb, train, config | 4 MinAtar. | Value-based RL for training an agent in MinAtar. |
| **OnPolicyMARL** | activation, loss, networks, optim, targets, train | 5 MABrax, MPE Spread, 11 SMAX | Training multiple on-policy RL agents in different multi-agent environments. |
| **OnPolicyRL** | loss, networks, optim, train | 4 MinAtar, 7 Brax, 2 Craftax. | Training an RL agent in a range of different RL environments using PPO-style algorithms. |
| **UnsupervisedEnvironmentDesign** | sample_levels, train_step, variable_config | 3 Kinetix sizes, Minigrid. | Generating and curating training environments/levels to improve RL agent generalization. |

## Development Setup

### 1. Set Up Your Development Environment

Install the environment and the pre-commit hooks with:

```bash
make install
```

This will also generate your `uv.lock` file.

## Contributing

We welcome contributions! DiscoBench grows stronger with more tasks and domains.

- **Found a bug?** [Open an issue](https://github.com/AlexGoldie/discobench/issues)
- **Want to add a task?** See our [Contributing Guide](https://AlexGoldie.github.io/discobench/how_to/overview/)
- **Adding datasets?** Check the [Dataset Integration Guide](https://AlexGoldie.github.io/discobench/how_to/dataset_integration/)

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed development guidelines.

## Citation

If you use DiscoBench in your research, please cite:

```bibtex
@article{goldie2025discobench,
  title={DiscoBench: An Open-Ended Benchmark For Algorithm Discovery},
  author={Alexander D. Goldie and Zilin Wang and Adrian Hayler and Deepak Nathani and Edan Toledo and Ken Thampiratwong and Aleksandra Kalisz and Michael Beukman and Alistair Letcher and Shashank Reddy and Clarisse Wibault and Theo Wolf and Charles O'Neill and Jakob N. Foerster and Shimon Whiteson and Roberta Raileanu},
  year={2025}
}
```

## License

DiscoBench is released under the [MIT License](LICENSE).
