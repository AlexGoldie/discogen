"""Metrics helpers for AA, Forgetting, and Score."""

from typing import Dict
import torch


def compute_forgetting_matrix(acc_matrix: torch.Tensor) -> torch.Tensor:
    # acc_matrix shape: [T, T], where entry (i,t) is acc on task i after task t
    running_max, _ = torch.cummax(acc_matrix, dim=1)
    final = acc_matrix[:, -1].unsqueeze(1).expand_as(acc_matrix)
    return torch.clamp(running_max - final, min=0.0)


def final_metrics(acc_matrix: torch.Tensor, lambda_forgetting: float) -> Dict[str, float]:
    t = acc_matrix.shape[0]
    aa = acc_matrix[:, -1].mean().item()
    if t > 1:
        f = compute_forgetting_matrix(acc_matrix)[:, :-1].mean().item()
    else:
        f = 0.0
    return {"AA": aa, "F": f, "Score": aa - lambda_forgetting * f}
