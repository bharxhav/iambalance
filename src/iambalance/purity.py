"""Purity calculation module for iambalance package."""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def calculate_distance_matrix(original: pd.DataFrame, oversampled: pd.DataFrame) -> np.ndarray:
    """Calculate the distance matrix between original and oversampled data points.

    Args:
        original: Original dataset.
        oversampled: Oversampled dataset.

    Returns:
        Distance matrix.
    """
    combined = pd.concat([original, oversampled])
    nn = NearestNeighbors(n_neighbors=len(original), metric='euclidean')
    nn.fit(combined)
    distances, _ = nn.kneighbors(oversampled)
    return distances


def calculate_neighborhood_purity(distances: np.ndarray, k: int = 5) -> List[float]:
    """Calculate the purity of each oversampled point's neighborhood.

    Args:
        distances: Distance matrix.
        k: Number of nearest neighbors to consider.

    Returns:
        List of purity scores for each oversampled point.
    """
    neighborhood_purities = []
    for point_distances in distances:
        k_nearest = point_distances[:k]
        # Assuming distance 0 means it's an original point
        purity = np.sum(k_nearest > 0) / k
        neighborhood_purities.append(purity)
    return neighborhood_purities


def calculate_global_purity(neighborhood_purities: List[float]) -> float:
    """Calculate the global purity score.

    Args:
        neighborhood_purities: List of neighborhood purity scores.

    Returns:
        Global purity score.
    """
    return np.mean(neighborhood_purities)


def calculate_purity(original: pd.DataFrame, oversampled: pd.DataFrame, k: int = 5) -> Tuple[float, List[float]]:
    """Calculate the purity of oversampled data.

    This function calculates both a global purity score and individual purity
    scores for each oversampled data point.

    Args:
        original: Original dataset.
        oversampled: Oversampled dataset.
        k: Number of nearest neighbors to consider for neighborhood purity.

    Returns:
        A tuple containing the global purity score and a list of individual purity scores.
    """
    distances = calculate_distance_matrix(original, oversampled)
    neighborhood_purities = calculate_neighborhood_purity(distances, k)
    global_purity = calculate_global_purity(neighborhood_purities)

    return global_purity, neighborhood_purities
