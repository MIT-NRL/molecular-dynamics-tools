"""Public result containers for clustering calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(slots=True)
class ClusterResult:
    """Cluster-size distribution and metadata from one connectivity rule.

    ``cluster_distribution`` contains a ``cluster_size`` axis and probability
    columns. ``metadata`` records the definition and execution plan.
    """

    cluster_distribution: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(slots=True)
class SharedNeighborClusterResult(ClusterResult):
    """Cluster, sharing, and percolation results from one neighbor projection.

    The additional tables report shared-neighbor categories, per-frame network
    metrics, total and finite component counts, and aggregate percolation.
    """

    sharing_distribution: pd.DataFrame
    frame_summary: pd.DataFrame
    percolation_cluster_distribution: pd.DataFrame
    percolation_summary: pd.DataFrame


__all__ = ["ClusterResult", "SharedNeighborClusterResult"]
