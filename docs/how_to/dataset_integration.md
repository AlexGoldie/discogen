# 🗂️ **Dataset Integration Guide**

This guide explains how to add new datasets to DiscoGen domains. This is particularly useful for task like ComputerVision or LanguageModelling, which use datasets instead of environments. We'll use FashionMNIST as our primary example throughout this guide.

---

## 🎯 Overview

Datasets in DiscoGen serve two main purposes:
1. **[Data Download](#download_datasetdest_loc-str)**: Provide raw data for training/evaluation
2. **[Data Loading](#load_datasetsrc_loc-str--data)**: Preprocess and format data for PyTorch models

Each dataset must implement specific functions that integrate with the DiscoGen framework.

---

## 📁 Dataset Structure

Every dataset lives in its own folder under `discogen/domains/{TASK_DOMAIN}/datasets/{DATASET_NAME}/`:

```
FashionMNIST/
├── make_dataset.py    # Required: download_dataset() and load_dataset()
├── config.py          # Optional: dataset-specific configurations
└── description.md     # Required: human-readable description
```

---

## 🔧 Required Functions

### `download_dataset(dest_loc: str)`

Downloads and saves the raw dataset to the specified location.

**FashionMNIST Example:**
```python
import datasets

def download_dataset(dest_loc: str):
    ds_dict = datasets.load_dataset("zalando-datasets/fashion_mnist")
    ds_dict.save_to_disk(dest_loc)
```

**Key Points:**
- **Called automatically** by `make_files.py` during task setup for each dataset
- **Destination path**: `dest_loc` is `task_src/{task_id}/data` where `task_id` is your dataset name (e.g., `task_src/FashionMNIST/data`)
- **Format**: Saves data in HuggingFace datasets format (`.arrow` files) for efficient loading
- **Caching behavior**: The first time a dataset is downloaded, it will be cached in .cache.
- **Integration timing**: Called once per dataset during the `make_files()` process, before training code is assembled

### `load_dataset(src_loc: str = "./data")`

Loads, preprocesses, and returns PyTorch-compatible datasets.

**Key Concept:** `load_dataset()` serves as the **interface between your specific dataset implementation and the general task code**. While you can implement it however you want, it must provide a consistent interface that the task's training and evaluation code can rely on.

**Key Points:**
- **Interface Role**: Adapts raw downloaded data to the format expected by the specific task domain
- **Task-Specific**: Different tasks expect different data formats (HuggingFace DatasetDict for CV, custom DataLoader for Language Modeling, etc.)
- **Consistency**: The same function name `load_dataset()` is used across datasets, but implementations vary by task requirements

### Example: FashionMNIST for Computer Vision Classification

For Computer Vision Classification tasks, the training code expects HuggingFace DatasetDict format:

```python
import datasets
import torch
import numpy as np

def load_dataset(src_loc: str = "./FashionMNIST/data"):
    # Load the raw downloaded data
    dataset = datasets.load_from_disk(src_loc)

    def preprocess_function(examples):
        # Convert to the expected format for CV classification
        images = torch.tensor(np.array(examples['image'])[:, None, :, :], dtype=torch.float32) / 255.0
        return {"pixel_values": images, "label": examples["label"]}

    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    processed_datasets.set_format("torch")
    return processed_datasets
```

**What this provides:**
- Returns a dictionary with `"train"` and `"test"` keys
- Each split contains datasets compatible with HuggingFace Trainer
- Standardized column names that the task code expects
- PyTorch tensors ready for training

**Interface Requirements for CV Classification:**
- `load_dataset()` → DatasetDict with `["train"]` and `["test"]` splits
- Each dataset has `"pixel_values"` (images) and `"label"` (targets) columns
- Data is in PyTorch tensor format

### What to put into `download_dataset` vs. `load_dataset`:

The output of `download_dataset` is cached on first run, so it is recommended to move any expensive overhead (if possible) into `download_dataset`. For example, for the Language Modelling task, `load_dataset` not only handles the downloading, but also the preprocessing into `*.bin` files, which is quite an expensive task in itself.

---

## 🎨 Integration with DiscoGen

### How Data Flows Through the System

1. **Task Creation** (`make_files.py`):
   ```python
   # For each dataset in task_config, automatically calls:
   dest_loc = Path("task_src") / dataset_name  # e.g., "task_src/FashionMNIST"
   download_dataset(dest_loc / "data")  # Downloads to "task_src/FashionMNIST/data"
   ```

2. **Training** (`train.py`):
   ```python
   from make_dataset import load_dataset

   processed_datasets = load_dataset()
   train_dataset = processed_datasets["train"]
   test_dataset = processed_datasets["test"]
   ```

3. **Evaluation** (`evaluate.py`):
   ```python
   # Currently loads test data directly for validation
   dataset = load_dataset("zalando-datasets/fashion_mnist", split="test")
   ```
