from pnpl.datasets import LibriBrainSpeech
from torch.utils.data import DataLoader
import random
import torch


run_keys = [
    ('0', '1', 'Sherlock2', '1'),
    ('0', '2', 'Sherlock2', '1'),
    ('0', '3', 'Sherlock2', '1'),
    ('0', '4', 'Sherlock2', '1'),
    ('0', '5', 'Sherlock2', '1'),
    ('0', '6', 'Sherlock2', '1'),
    ('0', '7', 'Sherlock2', '1'),
    ('0', '8', 'Sherlock2', '1'),
    ('0', '9', 'Sherlock2', '1'),
    ('0', '10', 'Sherlock2', '1'),
    ('0', '11', 'Sherlock2', '1'),
    ('0', '12', 'Sherlock2', '1'),
]


def download_dataset(dest_loc: str):

    LibriBrainSpeech(
      data_path=dest_loc,
      include_run_keys=run_keys,
      preload_files=True,
    )


class FilteredDataset(torch.utils.data.Dataset):
    """
    Parameters:
        dataset: LibriBrain dataset.
        limit_samples (int, optional): If provided, limits the length of the dataset to this
                          number of samples.
        speech_silence_only (bool, optional): If True, only includes segments that are either
                          purely speech or purely silence (with additional balancing).
        apply_sensors_speech_mask (bool, optional): If True, applies a fixed sensor mask to the sensor
                          data in each sample.
    """
    SENSORS_SPEECH_MASK = [18, 20, 22, 23, 45, 120, 138, 140, 142, 143, 145,
                       146, 147, 149, 175, 176, 177, 179, 180, 198, 271, 272, 275]

    def __init__(self,
                 dataset,
                 limit_samples=None,
                 disable=False,
                 apply_sensors_speech_mask=True):
        self.dataset = dataset
        self.limit_samples = limit_samples
        self.apply_sensors_speech_mask = apply_sensors_speech_mask
        self.sensors_speech_mask = self.SENSORS_SPEECH_MASK
        self.balanced_indices = list(range(len(dataset.samples)))
        self.balanced_indices = random.sample(self.balanced_indices, len(self.balanced_indices))

    def __len__(self):
        if self.limit_samples is not None:
            return self.limit_samples
        return len(self.balanced_indices)

    def __getitem__(self, index):
        original_idx = self.balanced_indices[index]
        if self.apply_sensors_speech_mask:
            sensors = self.dataset[original_idx][0][self.sensors_speech_mask]
        else:
            sensors = self.dataset[original_idx][0][:]
        label_from_the_middle_idx = self.dataset[original_idx][1].shape[0] // 2
        return [sensors, self.dataset[original_idx][1][label_from_the_middle_idx]]


def _split_run_keys(seed: int):
    """Deterministically select one run key for test using provided seed.

    Returns:
        (train_run_keys, test_run_keys)
    """
    rng = random.Random(seed)
    test_key = rng.choice(run_keys)
    train_keys = [rk for rk in run_keys if rk != test_key]
    return train_keys, [test_key]


def load_dataset(src_loc: str = "./LibriBrainSherlock2/data"):
    from config import config

    seed = config["dataset"]["seed"]
    train_run_keys, test_run_keys = _split_run_keys(seed)
    print(f"Seed {seed} selected test run key: {test_run_keys[0]}")

    train_data = LibriBrainSpeech(
      data_path=src_loc,
      include_run_keys=train_run_keys,
      tmin=config["dataset"]["tmin"],
      tmax=config["dataset"]["tmax"],
      preload_files=True,
      download=False,
    )

    test_data = LibriBrainSpeech(
      data_path=src_loc,
      include_run_keys=test_run_keys,
      tmin=config["dataset"]["tmin"],
      tmax=config["dataset"]["tmax"],
      preload_files=True,
      download=False,
    )

    print("Filtered dataset:")
    train_data_filtered = FilteredDataset(train_data)
    train_loader_filtered = DataLoader(train_data_filtered, batch_size=config["dataset"]["batch_size"], shuffle=True, num_workers=config["dataset"]["num_workers"])
    print(f"Train data contains {len(train_data_filtered)} samples.")

    test_data_filtered = FilteredDataset(test_data)
    test_loader_filtered = DataLoader(test_data_filtered, batch_size=config["dataset"]["batch_size"], shuffle=False, num_workers=config["dataset"]["num_workers"])
    print(f"Test data contains {len(test_data_filtered)} samples.")


    return {
        "train": train_loader_filtered,
        "test": test_loader_filtered
    }


def baselines(dataloader):
    labels = []
    for batch in dataloader:
        labels.append(batch[1])
    labels = torch.cat(labels, axis=0)

    total_count = labels.shape[0]
    positive_count = (labels == 1).sum().item()
    negative_count = (labels == 0).sum().item()

    # Implement baseline models for comparison
    # 1. Random baseline
    tp = positive_count * 0.5
    tn = negative_count * 0.5
    fp = negative_count * 0.5
    fn = positive_count * 0.5
    f1_pos = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    f1_neg = (2 * tn) / (2 * tn + fn + fp + 1e-8)
    random_f1_score = 0.5 * (f1_pos + f1_neg)

    # 2. Majority class baseline
    tp = positive_count
    tn = 0
    fp = total_count - positive_count
    fn = 0
    f1_pos = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    f1_neg = (2 * tn) / (2 * tn + fn + fp + 1e-8)
    majority_f1_score = 0.5 * (f1_pos + f1_neg)

    return {
        "random_f1_score": random_f1_score,
        "majority_f1_score": majority_f1_score,
    }
