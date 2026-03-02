# Using DiscoBench

This guide covers how to use DiscoBench for algorithm discovery tasks.

## Installation

Install DiscoBench using pip (once published) or from source:

### From Source

```bash
git clone https://github.com/AlexGoldie/discobench.git
cd discobench
make install
```

This will:
- Create a virtual environment using uv
- Install all dependencies
- Set up pre-commit hooks

## Quick Start

### 1. List Available Domains

See all available task domains:

```bash
discobench get-domains
```

### 2. View Modules for Each Domain

See which modules are available for each domain:

```bash
discobench get-modules
```

### 3. Create a Task

Create task files for a specific domain:

```bash
discobench create-task --task-domain OnPolicyRL
```

This creates a training task with default configuration. The generated files will appear in the `task_src/` directory.

## CLI Reference

DiscoBench provides three main commands:

### `create-task`

Create task source files for algorithm discovery.

**Usage:**
```bash
discobench create-task --task-domain DOMAIN [OPTIONS]
```

**Required Options:**
- `--task-domain TEXT`: The task domain to create (e.g., OnPolicyRL, LanguageModelling)

**Optional Flags:**
- `--test`: Create test task instead of training task
- `--config-path PATH`: Path to custom task_config.yaml (defaults to built-in config)
- `--example`: Create example task using prebuilt example configs
- `--use-base`: Initialises modules in codebase to baseline implementations, rather than using the less structured `edit` files. Using this may make exploration harder, but will start the task from a reasonably performant and working implementation.
- `--no-data`: Creates codebase without downloading or copying any data. This will not run in a production setting, but can be useful for quickly understanding how a codebase looks before running experiments (without having to wait for data download).
- `--eval-type EVAL_TYPE`: The type of evaluation to use. This defaults to `performance`, meaning the agent must maximise the performance of the algorithm. We also support `energy`, where the agent must minimise the energy used to train a model with the algorithm, and `time`, where the agent must minimise the time used to train a model with the algorithm.

**Examples:**

Create a meta-training task for OnPolicyRL (note, this will use the baseline codebase which does *not* include editable modules):
```bash
discobench create-task --task-domain OnPolicyRL
```

Create a meta-test task (note, this will use the baseline codebase which does *not* include editable modules):
```bash
discobench create-task --task-domain OnPolicyRL --test
```

Use a custom configuration:
```bash
discobench create-task --task-domain LanguageModelling --config-path my_config.yaml
```

Use the example configuration:
```bash
discobench create-task --task-domain LanguageModelling --example
```

### `get-domains`

List all available task domains in DiscoBench.

**Usage:**
```bash
discobench get-domains
```

**Output:**
Shows a list of all available domains like OnPolicyRL, LanguageModelling, BayesianOptimisation, etc.

### `get-modules`

List all available modules for each domain.

**Usage:**
```bash
discobench get-modules
```

**Output:**
Shows which modular components are available in each domain (e.g., loss, networks, optim, train).

## Python API

You can also use DiscoBench programmatically from Python:

### Creating Tasks

```python
from discobench import create_task

# Create a training task
create_task(task_domain="OnPolicyRL", test=False)

# Create a test task with custom config
create_task(
    task_domain="LanguageModelling",
    test=True,
    config_path="my_config.yaml"
)

# Create a task starting from baseline implementations
create_task(
    task_domain="OnPolicyRL",
    test=False,
    use_base=True
)
```

### Getting Domain Information

```python
from discobench import get_domains, get_modules

# Get list of all domains
domains = get_domains()
print(domains)

# Get modules for each domain
modules = get_modules()
for domain, module_list in modules.items():
    print(f"{domain}: {module_list}")
```

### Creating Custom Configurations

```python
from discobench import create_config

# Get default config for a domain
config = create_config(task_domain="OnPolicyRL")

# Modify the config
config["change_optim"] = True
config["change_loss"] = False

# Use it to create a task
create_task(
    task_domain="OnPolicyRL",
    test=False,
    config_dict=config
)
```

## Configuration Files

Task behavior is controlled by `task_config.yaml` files. Here would be an example:

```yaml
train_task_id: [MinAtar/Breakout, MinAtar/Freeway]
test_task_id: [MinAtar/Asterix, MinAtar/SpaceInvaders]

source_path: task_src/OnPolicyRL
template_backend: default

change_optim: true
change_loss: true
change_networks: false
change_train: false
```

**Key Fields:**

- `train_task_id`: Datasets/environments for training
- `test_task_id`: Datasets/environments for testing
- `source_path`: Where to create the task files (default: `task_src/`)
- `template_backend`: Which template variant to use (e.g., default, transformer, recurrent)
- `change_*`: Set to `true` to use editable module versions, `false` for baseline implementations

## Creating Tasks

### Workflow 1: Running a Default Task

```bash
# 1. Create the task
discobench create-task --task-domain OnPolicyRL

# 2. Navigate to the created task
cd task_src/OnPolicyRL

# 3. Run all task_ids in the task
# Note: this will only run if change_*=False for all *
# or you have completed module implementations!
python run_main.py
```

### Workflow 2: Using the example config

```bash
# 1. Create the task
discobench create-task --task-domain OnPolicyRL --example

# 2. Navigate to the created task
cd task_src/OnPolicyRL

# 3. Run your agent to develop new algorithms

# 4. Create the test task
discobench create-task --task-domain OnPolicyRL --example --test

# 5. Run evaluation
python run_main.py
```

### Workflow 3: Customizing Module Selection

1. Get the default config:
   ```python
   from discobench import create_config
   config = create_config("OnPolicyRL")
   ```

2. Modify which modules are editable:
   ```python
   config["change_optim"] = True  # Use editable optimizer
   config["change_loss"] = True   # Use editable loss
   ```

3. Create task with custom config:
   ```python
   from discobench import create_task
   create_task("OnPolicyRL", test=False, config_dict=config)
   ```

## Running DiscoBench

We provide all DiscoBench configs in `discobench/discobench_configs`. DiscoBench is a set of specific, hand-designed tasks for meta-meta-evaluation of algorithm discovery agents; in other words, these are tasks that exist within the support of DiscoGen, but should not be directly optimised on.

To run all DiscoBench tasks:
1. Get the list of discobench tasks:
```python
from discobench.utils import get_discobench_tasks
discobench_task_list = get_discobench_tasks()
```

2. Loop through creating all tasks:
```python
from discobench import create_discobench
for task in discobench_task_list:
    create_discobench(task, eval_type='performance')
    # Run agent on discobench task and report score
```

It is also possible to use different `eval_type`s for DiscoBench. For example, to benchmark algorithm discovery agents for developing energy efficient algorithms, use `create_discobench(task, eval_type='energy')`.

## Next Steps

- See [Domains](domains.md) for detailed information about available task domains
- See [Contributing Guide](how_to/overview.md) to add your own tasks
- See [Dataset Integration](how_to/dataset_integration.md) to add new datasets
