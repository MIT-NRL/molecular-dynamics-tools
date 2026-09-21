"""Shared-neighbor clustering public API."""

from __future__ import annotations

from collections.abc import Sequence

from .._execution import Backend
from ..clusters.polyhedra import SharingMode, analyze_bridging_clusters
from ..trajectory import Trajectory
from .results import SharedNeighborClusterResult

ALL_CONNECTIONS: tuple[SharingMode, ...] = (
    "connected",
    "corner",
    "edge",
    "face",
)


def compute_by_shared_neighbors(
    trajectory: Trajectory,
    centers: str,
    neighbors: str,
    cutoff: float,
    *,
    connections: Sequence[SharingMode] = ALL_CONNECTIONS,
    include_isolated: bool = False,
    min_shared_neighbors: int = 1,
    max_cluster_size: int | None = None,
    frames: slice | Sequence[int] | None = None,
    ncore: int | None = 1,
    backend: Backend = "auto",
    show_progress: bool = False,
) -> SharedNeighborClusterResult:
    """Compute center clusters induced by shared coordinating neighbors.

    Every center pair is categorized from the same neighbor search. The
    default returns the aggregate connected network and the mutually exclusive
    corner-, edge-, and face-sharing networks. Percolation uses connections
    with at least ``min_shared_neighbors`` shared atoms, independently of the
    categories requested in ``connections``.

    Args:
        trajectory: Trajectory returned by :func:`load_trajectory`.
        centers: Species whose projected clusters are measured.
        neighbors: Coordinating species shared by center pairs.
        cutoff: Maximum center-neighbor distance.
        connections: Cluster networks to return. By default, return connected,
            corner-, edge-, and face-sharing networks.
        include_isolated: Include size-one center components in each requested
            network and in percolation component counts.
        min_shared_neighbors: Minimum shared-neighbor count for a percolation
            edge. This does not filter ``connections``.
        max_cluster_size: Largest size represented explicitly in distributions.
        frames: Frame slice or explicit frame indices.
        ncore: Total logical-CPU budget.
        backend: ``"auto"``, ``"serial"``, or ``"multiprocessing"``.
        show_progress: Display calculation progress when available.

    Returns:
        Cluster, sharing, per-frame, and percolation tables from one streamed
        neighbor calculation.
    """

    resolved_connections = tuple(connections)
    legacy = analyze_bridging_clusters(
        trajectory,
        center=str(centers),
        ligand=str(neighbors),
        r_center_ligand=float(cutoff),
        sharing_modes=resolved_connections,
        include_unshared=include_isolated,
        min_shared_ligands=min_shared_neighbors,
        include_isolated=include_isolated,
        max_cluster_size=max_cluster_size,
        frames=frames,
        ncore=ncore,
        backend=backend,
        show_progress=show_progress,
    )
    sharing_distribution = legacy.sharing_distribution.rename(
        columns={"shared_ligands": "shared_neighbors"}
    )
    metadata = {
        "method": "shared_neighbors",
        "centers": str(centers),
        "neighbors": str(neighbors),
        "cutoff": float(cutoff),
        "connections": tuple(str(connection) for connection in resolved_connections),
        "include_isolated": bool(include_isolated),
        "min_shared_neighbors": int(min_shared_neighbors),
        "execution": legacy.metadata["execution"],
    }
    for table in (
        sharing_distribution,
        legacy.cluster_distribution,
        legacy.frame_summary,
        legacy.percolation_cluster_distribution,
        legacy.percolation_summary,
    ):
        table.attrs["analysis"] = metadata
    return SharedNeighborClusterResult(
        cluster_distribution=legacy.cluster_distribution,
        metadata=metadata,
        sharing_distribution=sharing_distribution,
        frame_summary=legacy.frame_summary,
        percolation_cluster_distribution=legacy.percolation_cluster_distribution,
        percolation_summary=legacy.percolation_summary,
    )


__all__ = ["ALL_CONNECTIONS", "compute_by_shared_neighbors"]
