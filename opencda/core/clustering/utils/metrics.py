"""Metric and scoring utilities for clustering algorithms."""

import numpy as np


def sigmoid(x: float, k: float = 1.0) -> float:
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-k * x))


def density_score(density: float, threshold: float) -> float:
    """Compute density score with sigmoid."""
    return sigmoid(density - threshold)


def compute_coalition_mean(values: list) -> float:
    """Compute mean of coalition values."""
    return np.mean(values) if values else 0.0


def compute_spatiotemporal_distance(pos1, vel1, pos2, vel2,
                                    alpha: float = 0.5) -> float:
    """Compute spatiotemporal distance between two vehicles."""
    spatial_dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
    velocity_diff = np.linalg.norm(np.array(vel1) - np.array(vel2))
    return alpha * spatial_dist + (1 - alpha) * velocity_diff


def calculate_cos(vec1, vec2) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / norm_product if norm_product > 0 else 0.0
