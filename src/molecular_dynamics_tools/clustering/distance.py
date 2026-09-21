"""Distance-connected clustering public API."""

from __future__ import annotations

from collections.abc import Sequence

from .._execution import Backend
from ..clusters.pairs import CutoffClusterDefinition, compute_cutoff_clusters
from ..trajectory import Trajectory
from .results import ClusterResult


def _species_pair(species: str | Sequence[str]) -> tuple[str, str]:
    if isinstance(species, str):
        value = str(species)
        return value, value
    values = tuple(str(value) for value in species)
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values
    raise ValueError("species must be one species or a pair of species")


def compute_by_distance(
    trajectory: Trajectory,
    species: str | Sequence[str],
    cutoff: float,
    *,
    r_min: float = 0.0,
    count_species: str | None = None,
    include_isolated: bool = False,
    max_cluster_size: int | None = None,
    frames: slice | Sequence[int] | None = None,
    ncore: int | None = 1,
    backend: Backend = "auto",
    show_progress: bool = False,
) -> ClusterResult:
    """Compute clusters from direct distance connections.

    ``species`` may name one species or a pair. For a pair, the graph contains
    only cross-species edges. By default cluster size is the total number of
    selected atoms and each component contributes once. Setting
    ``count_species`` reports the number of those atoms in each cluster and
    samples once per counted atom, matching a central-atom cluster
    distribution. Same-species searches always exclude self-pairs.

    Args:
        trajectory: Trajectory returned by :func:`load_trajectory`.
        species: One species or a pair of species.
        cutoff: Maximum connection distance.
        r_min: Minimum connection distance. It does not enable self-pairs.
        count_species: Species counted in cluster size. If omitted, count all
            selected atoms and sample once per component.
        include_isolated: Include size-one components. With ``count_species``,
            isolation refers to the graph projected onto that species.
        max_cluster_size: Largest size represented explicitly in the table.
        frames: Frame slice or explicit frame indices.
        ncore: Total logical-CPU budget.
        backend: ``"auto"``, ``"serial"``, or ``"multiprocessing"``.
        show_progress: Display calculation progress when available.

    Returns:
        Cluster-size distribution and calculation metadata.
    """

    first, second = _species_pair(species)
    counted = None if count_species is None else str(count_species)
    if counted is not None:
        if counted not in {first, second}:
            raise ValueError("count_species must be one of the selected species")
        if counted == second and first != second:
            first, second = second, first
        normalization = "atom1"
    else:
        normalization = "total"

    definition = CutoffClusterDefinition(
        first=first,
        second=second,
        r_min=float(r_min),
        r_max=float(cutoff),
        label="probability",
    )
    distribution = compute_cutoff_clusters(
        trajectory,
        [definition],
        normalize=normalization,
        include_unbonded=include_isolated,
        max_cluster_size=max_cluster_size,
        frames=frames,
        ncore=ncore,
        backend=backend,
        show_progress=show_progress,
    )
    # In a two-species graph, a counted atom bonded only to terminal neighbors
    # has size one after projection onto the counted species. The legacy
    # distance kernel regards that component as bonded, whereas the public
    # ``include_isolated`` option refers to the projected graph. Remove and
    # renormalize those samples so this distribution matches the connected
    # shared-neighbor result.
    if counted is not None and not include_isolated:
        size_one = distribution["cluster_size"] == 1
        isolated_probability = float(distribution.loc[size_one, "probability"].sum())
        remaining_probability = 1.0 - isolated_probability
        distribution.loc[size_one, "probability"] = 0.0
        if remaining_probability > 0:
            distribution["probability"] /= remaining_probability
            overflow = distribution.attrs["overflow_probability"]
            overflow["probability"] /= remaining_probability
        sample_counts = distribution.attrs["sample_counts"]
        sample_counts["probability"] = round(
            sample_counts["probability"] * max(0.0, remaining_probability)
        )
    metadata = {
        "method": "distance",
        "species": (first, second) if first != second else (first,),
        "r_min": float(r_min),
        "cutoff": float(cutoff),
        "count_species": counted,
        "include_isolated": bool(include_isolated),
        "sample_basis": "counted_atoms" if counted is not None else "clusters",
        "execution": distribution.attrs["execution"],
    }
    distribution.attrs["analysis"] = metadata
    return ClusterResult(cluster_distribution=distribution, metadata=metadata)


__all__ = ["compute_by_distance"]
