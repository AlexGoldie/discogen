import torch
import torch.nn as nn
from typing import Dict, Any


class TrajectoryPredictionModel(nn.Module):

    def __init__(self, config: Dict[str, Any]):
        """Initialize the model with the given configuration.
        Args:
            config (dict): Configuration object containing model parameters.
                - hidden_size (int): Hidden dimension for the model
                - num_modes (int): Number of prediction modes (possible futures)
                - future_len (int): Number of future timesteps to predict
                - past_len (int): Number of past timesteps observed
                - k_attr (int): Feature dimension per agent timestep (typically 2 for x,y)
                - map_attr (int): Feature dimension per map point (typically 2 for x,y)
        """
        super().__init__()
        """Fill in your network architecture here."""

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass for trajectory prediction.
        Args:
            batch: Dictionary containing 'input_dict' with:
                - obj_trajs: (B, N, T_past, k_attr) agent past trajectories
                - obj_trajs_mask: (B, N, T_past) validity mask
                - map_polylines: (B, S, P, map_attr) road polyline points
                - map_polylines_mask: (B, S, P) road validity mask
                - track_index_to_predict: (B,) which agent to predict for
        Returns:
            Dictionary with:
                - 'predicted_trajectory': (B, K, T_future, 5)
                    Distribution parameters [mean_x, mean_y, std_x, std_y, correlation]
                - 'predicted_probability': (B, K)
                    Mode probabilities (must sum to 1)
        """
        """Fill in your forward pass here."""
        ...
