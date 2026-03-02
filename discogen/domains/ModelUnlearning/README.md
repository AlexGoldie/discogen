# Unlearning Task

This guide provides instructions for setting up and running the Unlearning task in DiscoBench.

## Installation

After setting up the discobench environment (ie make install), install task-specific requirements:

```bash
pip install -r discobench/domains/Unlearning/utils/requirements.txt
pip install flash-attn==2.6.3 --no-build-isolation
```

## Configuration

### Add Hugging Face Token

Set your Hugging Face token as an environment variable:

```bash
export HF_TOKEN="your_token"
```

(Make sure your HF account has access to eg Llama models if one of your train/test datasets includes a Llama model.)

## Usage

### Select datasets

Specify train/test datasets in discobench/domains/ModelUnlearning/task_config.yaml. The list of available datasets is given by the folder names in discobench/domains/ModelUnlearning/datasets. Default: train = wmdp_cyber_qwen, test = tofu_qwen.

### Create Task

```bash
python discobench/create_task.py --task_domain Unlearning
```

### Run Task

```bash
cd task_src
python run_main.py
```


## For Mac users

The Unlearning task uses `flash-attention-2` by default. If running on Mac, where flash-attention is not available, run
```bash
python discobench/domains/ModelUnlearning/utils/toggle_attn.py --platform mac
```
to use `sdpa` instead. This will simply add a few lines (model_args: attn_implementation: sdpa) to the main_config.yaml files across all datasets. Run again with --platform server to remove the lines if need be.
