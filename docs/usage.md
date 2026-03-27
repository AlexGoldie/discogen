# Using DiscoGen

This guide covers how to use DiscoGen for generating algorithm discovery tasks, be it for agent optimisation or evaluation.

## Installation

Install DiscoGen using pip (once published) or from source:

### From Source

```bash
git clone https://github.com/AlexGoldie/discogen.git
cd discogen
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
discogen get-domains
```

### 2. View Modules for Each Domain

See which modules are available for each domain:

```bash
discogen get-modules
```

### 3. Create a Default Task

Create task files for a specific domain:

```bash
discogen create-task --task-domain OnPolicyRL
```

This creates a training task with default configuration. The generated files will appear in the `task_src/` directory. Note: the default task will not include any editable modules, so this will just show the domain's codebase structure.

## Python API Reference

You can use DiscoGen programmatically from Python, to automatically sample or create algorithm discovery tasks as part of your program:

### Creating Tasks

```python
from discogen import create_task

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

### Sampling Random Tasks

To benefit from procedural generation, a typical system would involve sampling random tasks and optimising the algorithm discovery agent for each one; for example, in the DiscoGen paper, we optimise the prompt of an algorithm discovery agent over a large number of randomly sampled DiscoGen tasks.

There are a number of possible options for sampling tasks. A typical pipeline may look like:

```python
from discogen import sample_task_config, create_task

task_domain, task_config = sample_task_config(
    p_edit=0.3,
    p_data = [0.4,0.4,0.2],
    p_use_base=0.,
    eval_type="random",
    use_backends= True,
    seed = 42
)

create_task(
    task_domain=task_domain,
    config_dict=task_config,
    baseline_scale=0.8
)

# Run agent to get final algorithm

create_task(
    task_domain=task_domain,
    config_dict=task_config,
    baseline_scale=0.8,
    test=True
)
```

### Getting Domain Information

```python
from discogen import get_domains, get_modules

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
from discogen import create_config

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


## CLI Reference

DiscoGen also provides a CLI for easy testing and use. The main commands are:

### `create-task`

Create task source files for algorithm discovery.

**Usage:**
```bash
discogen create-task --task-domain DOMAIN [OPTIONS]
```

**Required Options:**

- `--task-domain TEXT`: The task domain to create (e.g., OnPolicyRL, LanguageModelling).

**Optional Flags:**
- `--test`: Create test task instead of training task.

- `--config-path PATH`: Path to custom task_config.yaml (defaults to built-in config).

- `--example`: Create example task using prebuilt example configs.

- `--use-base`: Initialises modules in codebase to baseline implementations, rather than using the less structured `edit` files. Using this may make exploration harder, but will start the task from a reasonably performant and working implementation.

- `--no-data`: Creates codebase without downloading or copying any data. This will not run in a production setting, but can be useful for quickly understanding how a codebase looks before running experiments (without having to wait for data download).

- `--eval-type EVAL_TYPE`: The type of evaluation to use. This defaults to `performance`, meaning the agent must maximise the performance of the algorithm. We also support `energy`, where the agent must minimise the energy used to train a model with the algorithm, and `time`, where the agent must minimise the time used to train a model with the algorithm. Defaults to `performance`.

- `--baseline-scale BASELINE_SCALE`: If using `eval_type=energy` or `eval_type=time`, the agent's objective is to match a threshold score using minimum resources. Adjusting the baseline score can be used to make tasks easier (`scale<1`) or harder (`scale>1`). Defaults to `1`, meaning the agent must match the score of the baseline implementation.

- `--cache-root CACHE_ROOT`: Where downloaded data should be cached to.


**Examples:**

Create a meta-training task for OnPolicyRL (note, this will use the baseline codebase which does *not* include editable modules):
```bash
discogen create-task --task-domain OnPolicyRL
```

Create a meta-test task (note, this will use the baseline codebase which does *not* include editable modules):
```bash
discogen create-task --task-domain OnPolicyRL --test
```

Use a custom configuration:
```bash
discogen create-task --task-domain LanguageModelling --config-path my_config.yaml
```

Use the example configuration:
```bash
discogen create-task --task-domain LanguageModelling --example
```

Change the evaluation type:
```bash
discogen create-task --task-domain LanguageModelling --example --eval-type energy --baseline-scale 0.8
```

### `get-domains`

List all available task domains in DiscoBench.

**Usage:**
```bash
discogen get-domains
```

**Output:**
Shows a list of all available domains like OnPolicyRL, LanguageModelling, BayesianOptimisation, etc.

### `get-modules`

List all available modules for each domain.

**Usage:**
```bash
discogen get-modules
```

**Output:**
Shows which modular components are available in each domain (e.g., loss, networks, optim, train).

### `sample-task-config`

Randomly sample a new config and save it locally. This will uniformly sample from domains, and use weighted sampling according to command args to select editable modules, and meta-train and meta-test datasets.

NOTE: After sampling a config, you will then need to create a task using the sampled task domain. Here, you could also change the `baseline_scale` if desired. We implement these as two separate commands to enable independent creation of meta-train and meta-test parts of the task.

**Optional Flags**

- `--p-edit FLOAT`: The probability each module is marked as editable.

- `--p-data LIST`: A list of probabilities or weights for how each dataset is allocated. Can be a list of 2 values (`[p_meta_train, p_meta_test]`, which must be a set of probabilities with total <= 1), or a list of 3 values (`[w_meta_train, w_meta_test, w_exclude]`), which will be normalised into probabilities before sampling.

- `--p-use-base FLOAT`: The probability of starting each editable module off from a baseline implementation, rather than using the interface-only inputs.

- `--eval-type`: Which type of evaluation to use. Can be one of `["random", "performance", "time", "energy]`. `"random"` will randomly sample an `eval_type` dueing the config sampling process.

- `--no-backends`: If passed, will only sample tasks from the `default` backend.

- `--source-path PATH`: After creating the task (using the `create-task` command), where the task should be created.

- `--max-attempts INT`: How many attempts to take to try to sample a config, before raising an error. Due to rejection sampling, certain probabilities can sometimes be overly restrictive, such that tasks are hard to come by.

- `--seed INT`: A random seed for deterministic sampling of tasks.

- `--config-dest PATH`: Where to save the sampled config file. This will default to `task_config.yaml`.

### `create-config`

Create a config yaml file, to enable manual editing and testing of files.

**Required Options**

- `--task-domain`: The task domain to make the default config for.

**Optional Flags**

- `--save-dir`: Where to save the config to. Defaults to `./task_configs`.

## Creating Tasks

### Workflow 1: Using an example config

```bash
# 1. Create the task
discogen create-task --task-domain OnPolicyRL --example

# 2. Navigate to the created task
cd task_src/OnPolicyRL

# 3. Run your agent to develop new algorithms

# 4. Create the test task
discogen create-task --task-domain OnPolicyRL --example --test

# 5. Run evaluation
python run_main.py
```

### Workflow 2: Customizing Module Selection

1. Get the default config:
   ```python
   from discogen import create_config
   config = create_config("OnPolicyRL")
   ```

2. Modify which modules are editable:
   ```python
   config["change_optim"] = True  # Use editable optimizer
   config["change_loss"] = True   # Use editable loss
   ```

3. Create meta-train task with custom config:
   ```python
   from discogen import create_task
   create_task("OnPolicyRL", test=False, config_dict=config)
   ```

4. Run agent to discover algorithm

5. Create meta-test task with custom config:
   ```python
   from discogen import create_task
   create_task("OnPolicyRL", test=True, config_dict=config)
   ```

### Workflow 3: Sampling Random Tasks To Optimise Agent

```python
from discogen import sample_task_config, create_task

n_tasks = 50 # Run for some fixed number of tasks

for task_id in range(n_tasks):
    task_domain, task_config = sample_task_config(
        p_edit=0.3,
        p_data = [0.4,0.4,0.2],
        p_use_base=0.,
        eval_type="random",
        use_backends= True,
        seed = task_id
    ) # Sample a random task config

    create_task(
        task_domain=task_domain,
        config_dict=task_config,
        baseline_scale=0.8
    ) # Create the meta-train task

    run_agent(task) # Run agent to get final algorithm

    create_task(
        task_domain=task_domain,
        config_dict=task_config,
        baseline_scale=0.8,
        test=True
    ) # Create the meta-test task

    # Enter where you set the task to be made (in the config), and run `run_main.py` using (e.g.) subprocess.run

    optimise_agent(meta_test_scores)
```

## Running Agents

To run an agent on tasks generated by DiscoGen, you will need to ensure they are operating in the correct environment. For safety reasons, we recommend running all agents in a containerised environment.

When DiscoGen creates a task, it will create two files needed for installing packages etc.: `requirements.txt` and `install.sh`. We recommend using `install.sh`, since `requirements.txt` often does **NOT** include all necessary packages for running a task (say, if CUDA kernels need to be compiled). A typical pipeline might involve building a single docker image for EACH domain, when first sampled, and then caching that for every future time the domain is sampled.

If you prefer to not use `./install.sh`, or want to install additional packages, please check if the domain has a `README.md` which details additional package requirements. Some domains (e.g., ModelUnlearning) may also need environment variables set - for example, in that case, a HuggingFace API Key.


We do not currently provide code for any agents in DiscoGen. This is subject to change in the future!

## Running DiscoBench

We provide all DiscoBench configs in `discogen/discobench_configs`. DiscoBench is a set of specific, hand-designed tasks for meta-meta-evaluation of algorithm discovery agents; in other words, these are tasks that exist within the support of DiscoGen, but should not be directly optimised on.

To run all DiscoBench tasks:
1. Get the list of discobench tasks:
```python
from discogen.utils import get_discobench_tasks
discobench_task_list = get_discobench_tasks()
```

2. Loop through creating all tasks:
```python
from discogen import create_discobench
for task in discobench_task_list:
    create_discobench(task, eval_type='performance')
    # Run agent on discobench task and report score
```

It is also possible to use different `eval_type`s for DiscoBench. For example, to benchmark algorithm discovery agents for developing energy efficient algorithms, use `create_discobench(task, eval_type='energy')`.

## Next Steps

- See [Domains](domains.md) for detailed information about available task domains
- See [Contributing Guide](how_to/overview.md) to add your own tasks
- See [Dataset Integration](how_to/dataset_integration.md) to add new datasets
