# LanguageModelling Task

This guide provides instructions for setting up and running the LanguageModelling task in DiscoGen.

## Installation

LanguageModelling requires installing packages in two steps. Therefore, in addition to installing from `requirements.txt`, you should install `ninja`, `causal-conv1d` and `mamba-ssm`.

After creating the LanguageModelling task, and entering the correct directory, use the following commands:
```bash
pip install -r requirements.txt
pip install ninja
pip install "causal-conv1d>=1.4.0" --no-build-isolation --no-cache-dir
pip install "mamba-ssm>=2.0.0" --no-build-isolation --no-cache-dir
```

This is also handled by `install.sh` (which will also ensure the necessary nvidia packages are installed and set).
