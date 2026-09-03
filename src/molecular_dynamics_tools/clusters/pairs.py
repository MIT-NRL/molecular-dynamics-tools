"""Cutoff-bonded atomic cluster distributions."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import freud
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .._execution import Backend, execute_frame_chunks, plan_execution
from .._geometry import (
    accumulate_integer_histogram,
    freud_box,
    merge_integer_histograms,
)
from ..trajectory import Trajectory
from .core import DisjointSet

ClusterNormalization = Literal["atom1", "total"]


@dataclass(frozen=True, slots=True)
class CutoffClusterDefinition:
    """One distance-based bond graph between two species."""

    first: str
    second: str
    r_max: float
    r_min: float = 0.0
    label: str | None = None

    @property
    def name(self) -> str:
        return self.label or f"{self.first}-{self.second}"


def _normalize_definitions(
    trajectory: Trajectory,
    definitions: Iterable[CutoffClusterDefinition | Sequence[object]],
) -> tuple[CutoffClusterDefinition, ...]:
    normalized: list[CutoffClusterDefinition] = []
    for raw in definitions:
        if isinstance(raw, CutoffClusterDefinition):
            definition = raw
        else:
            values = tuple(raw)
            if len(values) == 3:
                definition = CutoffClusterDefinition(
                    str(values[0]), str(values[1]), float(values[2])
                )
            elif len(values) == 4:
                definition = CutoffClusterDefinition(
                    str(values[0]), str(values[1]), float(values[3]), float(values[2])
                )
            elif len(values) == 5:
                definition = CutoffClusterDefinition(
                    str(values[0]),
                    str(values[1]),
                    float(values[3]),
                    float(values[2]),
                    str(values[4]),
                )
            else:
                raise ValueError("cluster definitions must contain 3, 4, or 5 values")
        if (
            not np.isfinite(definition.r_min)
            or not np.isfinite(definition.r_max)
            or definition.r_min < 0
            or definition.r_max <= definition.r_min
        ):
            raise ValueError("cluster cutoffs require 0 <= r_min < r_max")
        normalized.append(definition)
    if not normalized:
        raise ValueError("definitions must contain at least one cutoff cluster")
    names = [definition.name for definition in normalized]
    if len(set(names)) != len(names):
        raise ValueError("cluster labels must be unique; use CutoffClusterDefinition.label")
    missing = {
        species
        for definition in normalized
        for species in (definition.first, definition.second)
        if species not in trajectory.species
    }
    if missing:
        raise ValueError(f"unknown species in cluster definitions: {sorted(missing)}")
    return tuple(normalized)


def _cluster_sizes(
    box: freud.Box,
    first_positions: NDArray[np.float32],
    second_positions: NDArray[np.float32],
    definition: CutoffClusterDefinition,
    normalize: ClusterNormalization,
    include_unbonded: bool,
) -> NDArray[np.int64]:
    same_species = definition.first == definition.second
    first_count = len(first_positions)
    if same_species:
        disjoint = DisjointSet(first_count)
        neighbors = freud.locality.AABBQuery(box, first_positions).query(
            first_positions,
            {
                "mode": "ball",
                "r_min": definition.r_min,
                "r_max": definition.r_max,
                "exclude_ii": True,
            },
        ).toNeighborList()
        for left, right in zip(
            neighbors.query_point_indices, neighbors.point_indices, strict=True
        ):
            disjoint.union(int(left), int(right))
        roots = np.asarray([disjoint.find(index) for index in range(first_count)])
        _, inverse, component_sizes = np.unique(
            roots, return_inverse=True, return_counts=True
        )
        if normalize == "atom1":
            sizes = component_sizes[inverse]
            return sizes if include_unbonded else sizes[sizes > 1]
        return component_sizes if include_unbonded else component_sizes[component_sizes > 1]

    second_count = len(second_positions)
    disjoint = DisjointSet(first_count + second_count)
    neighbors = freud.locality.AABBQuery(box, second_positions).query(
        first_positions,
        {
            "mode": "ball",
            "r_min": definition.r_min,
            "r_max": definition.r_max,
        },
    ).toNeighborList()
    for first_index, second_index in zip(
        neighbors.query_point_indices, neighbors.point_indices, strict=True
    ):
        disjoint.union(int(first_index), first_count + int(second_index))
    roots = np.asarray(
        [disjoint.find(index) for index in range(first_count + second_count)]
    )
    unique, inverse, component_sizes = np.unique(
        roots, return_inverse=True, return_counts=True
    )
    first_sizes = np.bincount(inverse[:first_count], minlength=len(unique))
    second_sizes = np.bincount(inverse[first_count:], minlength=len(unique))
    bonded = (first_sizes > 0) & (second_sizes > 0)
    if normalize == "atom1":
        sizes = first_sizes[inverse[:first_count]]
        return sizes if include_unbonded else sizes[bonded[inverse[:first_count]]]
    return component_sizes if include_unbonded else component_sizes[bonded]


def _cutoff_cluster_chunk(
    trajectory: Trajectory,
    frame_indices: tuple[int, ...],
    definitions: tuple[CutoffClusterDefinition, ...],
    normalize: ClusterNormalization,
    include_unbonded: bool,
) -> dict[str, NDArray[np.int64]]:
    histograms = {definition.name: np.zeros(1, dtype=np.int64) for definition in definitions}
    for frame in trajectory.iter_frames(frame_indices):
        box = freud_box(frame.dimensions)
        for definition in definitions:
            sizes = _cluster_sizes(
                box,
                frame.positions_of(definition.first),
                frame.positions_of(definition.second),
                definition,
                normalize,
                include_unbonded,
            )
            histograms[definition.name] = accumulate_integer_histogram(
                histograms[definition.name], sizes
            )
    return histograms


def compute_cutoff_clusters(
    trajectory: Trajectory,
    definitions: Iterable[CutoffClusterDefinition | Sequence[object]],
    *,
    normalize: ClusterNormalization = "atom1",
    include_unbonded: bool = False,
    max_cluster_size: int | None = None,
    frames: slice | Sequence[int] | None = None,
    ncore: int | None = 1,
    backend: Backend = "auto",
    show_progress: bool = False,
) -> pd.DataFrame:
    """Analyze connected components of direct distance-cutoff bond graphs.

    Tuple definitions use ``(first, second, r_max)`` or
    ``(first, second, r_min, r_max)``. ``normalize='atom1'`` reports the number
    of first-species atoms in the cluster and samples once per first-species
    atom. ``normalize='total'`` reports total atoms and samples once per cluster.
    """

    if normalize not in {"atom1", "total"}:
        raise ValueError("normalize must be 'atom1' or 'total'")
    normalized = _normalize_definitions(trajectory, definitions)
    frame_indices = trajectory.resolve_frame_indices(frames)
    plan = plan_execution(
        len(frame_indices),
        ncore=ncore,
        backend=backend,
        workload=len(frame_indices) * trajectory.n_atoms * len(normalized),
        auto_multiprocessing_threshold=3_000_000,
    )
    partials, execution = execute_frame_chunks(
        trajectory,
        frame_indices,
        plan,
        _cutoff_cluster_chunk,
        (normalized, normalize, bool(include_unbonded)),
        show_progress=show_progress,
        description="Cutoff clusters",
    )
    histograms = {definition.name: np.zeros(1, dtype=np.int64) for definition in normalized}
    for partial in partials:
        for name, histogram in partial.items():
            histograms[name] = merge_integer_histograms(histograms[name], histogram)

    observed_max = max(len(histogram) - 1 for histogram in histograms.values())
    if max_cluster_size is None:
        resolved_max = max(1, observed_max)
    else:
        resolved_max = int(max_cluster_size)
        if resolved_max < 1:
            raise ValueError("max_cluster_size must be positive or None")
    axis = np.arange(1, resolved_max + 1)
    result = pd.DataFrame({"cluster_size": axis})
    overflow: dict[str, float] = {}
    sample_counts: dict[str, int] = {}
    for definition in normalized:
        histogram = histograms[definition.name]
        total = int(histogram.sum())
        values = np.zeros(len(axis), dtype=float)
        upper = min(len(histogram), resolved_max + 1)
        if total and upper > 1:
            values[: upper - 1] = histogram[1:upper] / total
        result[definition.name] = values
        overflow[definition.name] = (
            float(histogram[resolved_max + 1 :].sum() / total) if total else 0.0
        )
        sample_counts[definition.name] = total
    result.attrs["method"] = "cutoff"
    result.attrs["definitions"] = normalized
    result.attrs["normalization"] = normalize
    result.attrs["include_unbonded"] = bool(include_unbonded)
    result.attrs["sample_counts"] = sample_counts
    result.attrs["overflow_probability"] = overflow
    result.attrs["execution"] = execution
    return result


# The architecture's original planned name remains a descriptive convenience.
compute_pair_clusters = compute_cutoff_clusters


__all__ = [
    "ClusterNormalization",
    "CutoffClusterDefinition",
    "compute_cutoff_clusters",
    "compute_pair_clusters",
]
