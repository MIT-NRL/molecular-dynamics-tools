"""Distance and shared-neighbor clustering calculations."""

from .distance import compute_by_distance
from .results import ClusterResult, SharedNeighborClusterResult
from .shared_neighbors import ALL_CONNECTIONS, compute_by_shared_neighbors

__all__ = [
    "ALL_CONNECTIONS",
    "ClusterResult",
    "SharedNeighborClusterResult",
    "compute_by_distance",
    "compute_by_shared_neighbors",
]
