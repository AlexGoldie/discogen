# Unlearning Task

This guide provides instructions for setting up and running the ModelUnlearning task in DiscoGen.

## Installation

ModelUnlearning requires installing packages in two steps. Therefore, in addition to installing from `requirements.txt`, you should install `flash-attn`.

After creating the ModelUnlearning task, and entering the correct directory, use the following commands:
```bash
pip install -r requirements.txt
pip install flash-attn==2.6.3 --no-build-isolation
```

This is also handled by `install.sh`.

## Configuration

### Add Hugging Face Token

Set your Hugging Face token as an environment variable:

```bash
export HF_TOKEN="your_token"
```

(Make sure your HF account has access to eg Llama models if one of your train/test datasets includes a Llama model.)

# Additional Information

## For Mac users

The Unlearning task uses `flash-attention-2` by default. If running on Mac, where flash-attention is not available, run
```bash
python discogen/domains/ModelUnlearning/utils/toggle_attn.py --platform mac
```
to use `sdpa` instead. This will simply add a few lines (model_args: attn_implementation: sdpa) to the main_config.yaml files across all datasets. Run again with --platform server to remove the lines if need be.
