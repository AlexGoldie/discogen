from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """Data parameters."""
    past_len: int = 21  # 2.1 seconds at 10Hz
    future_len: int = 60  # 6 seconds at 10Hz
    trajectory_sample_interval: int = 1
    max_agents: int = 15
    max_road_segs: int = 256
    max_points_per_lane: int = 20
    map_range: int = 100  # meters
    only_train_on_ego: bool = False
    store_data_in_memory: bool = False
    use_cache: bool = False
    overwrite_cache: bool = False
    cache_path: str = "./cache"


@dataclass
class ModelConfig:
    """Model architecture parameters."""
    hidden_size: int = 128
    num_modes: int = 6
    k_attr: int = 2  # trajectory attributes (x, y)
    map_attr: int = 2  # map point attributes (x, y)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 32
    learning_rate: float = 0.00075
    num_epochs: int = 100
    max_grad_norm: float = 5.0
    weight_decay: float = 0.01


@dataclass
class Config:
    """Main config combining data, model, and training settings."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Device
    device: str = "cuda"
    seed: int = 42

    def to_dict(self):
        """Convert config to flat dictionary for model initialization."""
        return {
            # Data
            'past_len': self.data.past_len,
            'future_len': self.data.future_len,
            'trajectory_sample_interval': self.data.trajectory_sample_interval,
            'max_agents': self.data.max_agents,
            'max_road_segs': self.data.max_road_segs,
            'max_points_per_lane': self.data.max_points_per_lane,
            'only_train_on_ego': self.data.only_train_on_ego,
            'store_data_in_memory': self.data.store_data_in_memory,
            'use_cache': self.data.use_cache,
            'overwrite_cache': self.data.overwrite_cache,
            'cache_path': self.data.cache_path,
            # Model
            'hidden_size': self.model.hidden_size,
            'num_modes': self.model.num_modes,
            'k_attr': self.model.k_attr,
            'map_attr': self.model.map_attr,
            # Training
            'batch_size': self.training.batch_size,
            'learning_rate': self.training.learning_rate,
            'num_epochs': self.training.num_epochs,
            'max_grad_norm': self.training.max_grad_norm,
            'weight_decay': self.training.weight_decay,
            # Device
            'device': self.device,
            'seed': self.seed,
        }


def get_config() -> Config:
    """Get default configuration."""
    return Config()
