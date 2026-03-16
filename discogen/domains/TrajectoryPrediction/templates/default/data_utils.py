import math
import numpy as np
import torch
from typing import Dict, Tuple, Optional


def rotate_points_along_z(points: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rotate points around Z-axis by angle (radians)."""
    cosa = np.cos(angle)
    sina = np.sin(angle)

    if points.shape[-1] == 2:
        rot_matrix = np.stack((
            cosa, sina,
            -sina, cosa
        ), axis=1).reshape(-1, 2, 2)

        points_rot = np.matmul(points, rot_matrix)
    else:
        # Handle 3D points (x, y, z, ...)
        rot_matrix = np.stack((
            cosa, sina, np.zeros_like(cosa),
            -sina, cosa, np.zeros_like(cosa),
            np.zeros_like(cosa), np.zeros_like(cosa), np.ones_like(cosa)
        ), axis=1).reshape(-1, 3, 3)

        points_rot = np.zeros_like(points)
        points_rot[..., :3] = np.matmul(points[..., :3], rot_matrix)
        points_rot[..., 3:] = points[..., 3:]

    return points_rot


def rotate_points_along_z_tensor(points: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate points around Z-axis (PyTorch version)."""
    cosa = torch.cos(angle)
    sina = torch.sin(angle)

    if points.shape[-1] == 2:
        rot_matrix = torch.stack((
            cosa, sina,
            -sina, cosa
        ), dim=1).view(-1, 2, 2).float()

        points_rot = torch.matmul(points, rot_matrix)
    else:
        ones = torch.ones_like(cosa)
        zeros = torch.zeros_like(cosa)
        rot_matrix = torch.stack((
            cosa, sina, zeros,
            -sina, cosa, zeros,
            zeros, zeros, ones
        ), dim=1).view(-1, 3, 3).float()

        points_rot = torch.zeros_like(points)
        points_rot[..., :3] = torch.matmul(points[..., :3], rot_matrix)
        points_rot[..., 3:] = points[..., 3:]

    return points_rot


def get_polyline_dir(polyline: np.ndarray) -> np.ndarray:
    """
    Compute direction vectors for polyline segments.

    Args:
        polyline: ndarray of shape (N, 2+) - N points

    Returns:
        Direction vectors of shape (N, 2+)
    """
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    polyline_pre[0] = polyline[0]
    diff = polyline - polyline_pre
    polyline_dir = diff / np.clip(np.linalg.norm(diff, axis=-1, keepdims=True), a_min=1e-6, a_max=1e9)
    return polyline_dir


def interpolate_polyline(polyline: np.ndarray, num_points: int = 20) -> np.ndarray:
    """Resample polyline to fixed number of points."""
    if len(polyline) == 0:
        return np.zeros((num_points, polyline.shape[-1] if len(polyline.shape) > 1 else 2))

    if len(polyline) == 1:
        return np.tile(polyline, (num_points, 1))

    # Compute cumulative distances
    distances = np.zeros(len(polyline))
    distances[1:] = np.cumsum(np.linalg.norm(np.diff(polyline[:, :2], axis=0), axis=1))

    if distances[-1] < 1e-6:
        return np.tile(polyline[0], (num_points, 1))

    # Normalize distances
    distances = distances / distances[-1]

    # Target distances
    target_distances = np.linspace(0, 1, num_points)

    # Interpolate each dimension
    interpolated = np.zeros((num_points, polyline.shape[-1]))
    for d in range(polyline.shape[-1]):
        interpolated[:, d] = np.interp(target_distances, distances, polyline[:, d])

    return interpolated


def generate_mask(past_len: int, total_len: int, sample_interval: int = 1) -> np.ndarray:
    """Generate validity mask for trajectory sampling."""
    mask = np.zeros(total_len)
    indices = np.arange(0, total_len, sample_interval)
    mask[indices] = 1.0
    return mask


def normalize_trajectory(
    trajectory: np.ndarray,
    center: np.ndarray,
    heading: float
) -> np.ndarray:
    """Transform trajectory to ego-centric coordinates (centered, heading-aligned)."""
    normalized = trajectory.copy()
    normalized[:, :2] = normalized[:, :2] - center

    # Rotate to align heading with x-axis
    cos_h = np.cos(-heading)
    sin_h = np.sin(-heading)
    x_rot = normalized[:, 0] * cos_h - normalized[:, 1] * sin_h
    y_rot = normalized[:, 0] * sin_h + normalized[:, 1] * cos_h
    normalized[:, 0] = x_rot
    normalized[:, 1] = y_rot

    return normalized


def denormalize_trajectory(
    trajectory: np.ndarray,
    center: np.ndarray,
    heading: float
) -> np.ndarray:
    """Transform trajectory from ego-centric back to world coordinates."""
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)

    denormalized = trajectory.copy()
    x_rot = denormalized[:, 0] * cos_h - denormalized[:, 1] * sin_h
    y_rot = denormalized[:, 0] * sin_h + denormalized[:, 1] * cos_h
    denormalized[:, 0] = x_rot
    denormalized[:, 1] = y_rot

    # Translate back to world position
    denormalized[:, :2] = denormalized[:, :2] + center

    return denormalized


def compute_ade(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Average Displacement Error between predicted and ground truth trajectories."""
    errors = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=-1)
    if mask is not None:
        errors = errors * mask
        return errors.sum() / (mask.sum() + 1e-6)
    return errors.mean()


def compute_fde(pred: np.ndarray, gt: np.ndarray) -> float:
    """Final Displacement Error at the last timestep."""
    return np.linalg.norm(pred[-1, :2] - gt[-1, :2])


def compute_min_ade_fde(
    pred_modes: np.ndarray,
    gt: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[float, float, int]:
    """Compute best ADE/FDE across K prediction modes. Returns (min_ade, min_fde, best_idx)."""
    num_modes = pred_modes.shape[0]
    ades = []
    fdes = []

    for k in range(num_modes):
        ades.append(compute_ade(pred_modes[k], gt, mask))
        fdes.append(compute_fde(pred_modes[k], gt))

    best_idx = np.argmin(fdes)
    return ades[best_idx], fdes[best_idx], best_idx
