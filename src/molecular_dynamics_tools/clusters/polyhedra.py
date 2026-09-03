"""Bridging-ligand cluster networks and periodic percolation analyses."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

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
from .core import component_members, periodic_components

SharingMode = Literal["corner", "edge", "face", "connected"]
_ALL_SHARING_MODES: tuple[SharingMode, ...] = ("corner", "edge", "face", "connected")


@dataclass(slots=True)
class BridgingClusterResult:
    """Related tables derived from one shared-ligand network calculation."""

    sharing_distribution: pd.DataFrame
    cluster_distribution: pd.DataFrame
    frame_summary: pd.DataFrame
    percolation_cluster_distribution: pd.DataFrame
    percolation_summary: pd.DataFrame
    metadata: dict[str, Any]


def _sharing_type(shared_ligands: int) -> str:
    if shared_ligands == 1:
        return "corner"
    if shared_ligands == 2:
        return "edge"
    if shared_ligands >= 3:
        return "face"
    return "none"


def _shared_ligand_network(
    box: freud.Box,
    center_positions: NDArray[np.float32],
    ligand_positions: NDArray[np.float32],
    cutoff: float,
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], list[NDArray[np.int32]]]]:
    """Return shared counts and periodic translations for center-center links."""

    neighbors = freud.locality.AABBQuery(box, ligand_positions).query(
        center_positions, {"mode": "ball", "r_max": cutoff}
    ).toNeighborList()
    if len(neighbors) == 0:
        return {}, {}
    centers = np.asarray(neighbors.query_point_indices, dtype=np.int32)
    ligands = np.asarray(neighbors.point_indices, dtype=np.int32)
    displacement = ligand_positions[ligands] - center_positions[centers]
    images = np.asarray(box.get_images(displacement), dtype=np.int32)

    ligand_centers: dict[int, dict[int, NDArray[np.int32]]] = defaultdict(dict)
    for center, ligand, image in zip(centers, ligands, images, strict=True):
        ligand_centers[int(ligand)].setdefault(int(center), image.copy())

    shared_counts: dict[tuple[int, int], int] = defaultdict(int)
    translations: dict[tuple[int, int], set[tuple[int, int, int]]] = defaultdict(set)
    for center_images in ligand_centers.values():
        linked = sorted(center_images)
        for left_offset, left in enumerate(linked[:-1]):
            for right in linked[left_offset + 1 :]:
                pair = (left, right)
                shared_counts[pair] += 1
                translation = center_images[right] - center_images[left]
                translations[pair].add(tuple(int(value) for value in translation))
    return dict(shared_counts), {
        pair: [np.asarray(value, dtype=np.int32) for value in sorted(values)]
        for pair, values in translations.items()
    }


def _sharing_cluster_sizes(
    center_count: int,
    shared_counts: dict[tuple[int, int], int],
    mode: SharingMode,
    include_unshared: bool,
) -> NDArray[np.int64]:
    edges: list[tuple[int, int]] = []
    for pair, count in shared_counts.items():
        keep = count >= 1 if mode == "connected" else _sharing_type(count) == mode
        if keep:
            edges.append(pair)
    components, inverse = component_members(center_count, edges)
    if not components:
        return np.empty(0, dtype=np.int64)
    sizes = np.asarray([len(component) for component in components], dtype=np.int64)
    center_sizes = sizes[inverse]
    return center_sizes if include_unshared else center_sizes[center_sizes > 1]


def _percolation_metrics(
    center_count: int,
    shared_counts: dict[tuple[int, int], int],
    translations: dict[tuple[int, int], list[NDArray[np.int32]]],
    min_shared_ligands: int,
    include_isolated: bool,
) -> tuple[dict[str, Any], NDArray[np.int64], NDArray[np.int64]]:
    translated_edges: list[tuple[int, int, NDArray[np.int32]]] = []
    for (left, right), count in shared_counts.items():
        if count < min_shared_ligands:
            continue
        shifts = translations.get((left, right)) or [np.zeros(3, dtype=np.int32)]
        translated_edges.extend((left, right, shift) for shift in shifts)
    components = periodic_components(center_count, translated_edges)
    eligible = [
        component
        for component in components
        if include_isolated or int(component["size"]) > 1
    ]
    finite = [component for component in eligible if not bool(component["wrap_any"])]
    total_sizes = np.asarray([component["size"] for component in eligible], dtype=np.int64)
    finite_sizes = np.asarray([component["size"] for component in finite], dtype=np.int64)
    largest_size = max((int(component["size"]) for component in components), default=0)
    wrapping_mass = sum(
        int(component["size"]) for component in components if bool(component["wrap_any"])
    )
    row = {
        "n_centers": center_count,
        "n_clusters_total": len(eligible),
        "n_clusters_finite": len(finite),
        "n_wrapping_clusters": sum(bool(component["wrap_any"]) for component in eligible),
        "largest_cluster_size": largest_size,
        "largest_cluster_fraction": largest_size / center_count if center_count else 0.0,
        "percolation_strength": wrapping_mass / center_count if center_count else 0.0,
        "wrap_x": any(bool(component["wrap_x"]) for component in components),
        "wrap_y": any(bool(component["wrap_y"]) for component in components),
        "wrap_z": any(bool(component["wrap_z"]) for component in components),
        "wrap_any": any(bool(component["wrap_any"]) for component in components),
        "M0_finite": float(len(finite_sizes)),
        "M1_finite": float(finite_sizes.sum()),
        "M2_finite": float(np.square(finite_sizes.astype(float)).sum()),
    }
    return row, total_sizes, finite_sizes


def _bridging_chunk(
    trajectory: Trajectory,
    frame_indices: tuple[int, ...],
    center: str,
    ligand: str,
    cutoff: float,
    sharing_modes: tuple[SharingMode, ...],
    include_unshared: bool,
    min_shared_ligands: int,
    include_isolated: bool,
) -> dict[str, Any]:
    shared_histogram = np.zeros(1, dtype=np.int64)
    cluster_histograms = {mode: np.zeros(1, dtype=np.int64) for mode in sharing_modes}
    total_histogram = np.zeros(1, dtype=np.int64)
    finite_histogram = np.zeros(1, dtype=np.int64)
    frame_rows: list[dict[str, Any]] = []
    for frame in trajectory.iter_frames(frame_indices):
        center_positions = frame.positions_of(center)
        shared_counts, translations = _shared_ligand_network(
            freud_box(frame.dimensions),
            center_positions,
            frame.positions_of(ligand),
            cutoff,
        )
        shared_values = np.fromiter(shared_counts.values(), dtype=np.int64)
        shared_histogram = accumulate_integer_histogram(shared_histogram, shared_values)
        for mode in sharing_modes:
            cluster_histograms[mode] = accumulate_integer_histogram(
                cluster_histograms[mode],
                _sharing_cluster_sizes(
                    len(center_positions), shared_counts, mode, include_unshared
                ),
            )
        percolation, total_sizes, finite_sizes = _percolation_metrics(
            len(center_positions),
            shared_counts,
            translations,
            min_shared_ligands,
            include_isolated,
        )
        total_histogram = accumulate_integer_histogram(total_histogram, total_sizes)
        finite_histogram = accumulate_integer_histogram(finite_histogram, finite_sizes)
        frame_rows.append(
            {
                "frame": frame.index,
                "source_frame": frame.source_index,
                "corner_links": int(np.count_nonzero(shared_values == 1)),
                "edge_links": int(np.count_nonzero(shared_values == 2)),
                "face_links": int(np.count_nonzero(shared_values >= 3)),
                "connected_links": len(shared_values),
                **percolation,
            }
        )
    return {
        "shared_histogram": shared_histogram,
        "cluster_histograms": cluster_histograms,
        "total_histogram": total_histogram,
        "finite_histogram": finite_histogram,
        "frame_rows": frame_rows,
    }


def _percolation_summary(frame_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "largest_cluster_size",
        "largest_cluster_fraction",
        "percolation_strength",
        "M0_finite",
        "M1_finite",
        "M2_finite",
    )
    row: dict[str, float] = {}
    for metric in metrics:
        values = frame_summary[metric].to_numpy(dtype=float)
        row[f"mean_{metric}"] = float(values.mean())
        row[f"std_{metric}"] = float(values.std(ddof=0))
    for axis in ("x", "y", "z", "any"):
        row[f"wrap_probability_{axis}"] = float(frame_summary[f"wrap_{axis}"].mean())
    return pd.DataFrame([row])


def analyze_bridging_clusters(
    trajectory: Trajectory,
    center: str,
    ligand: str,
    r_center_ligand: float,
    *,
    sharing_modes: Sequence[SharingMode] = _ALL_SHARING_MODES,
    include_unshared: bool = False,
    min_shared_ligands: int = 1,
    include_isolated: bool = True,
    max_cluster_size: int | None = None,
    frames: slice | Sequence[int] | None = None,
    ncore: int | None = 1,
    backend: Backend = "auto",
    show_progress: bool = False,
) -> BridgingClusterResult:
    """Analyze a center network induced by bridging/shared ligand atoms.

    A single center-ligand neighbor search per frame feeds link sharing,
    corner/edge/face/connected cluster distributions, and periodic wrapping
    analysis. Percolation links require at least ``min_shared_ligands``.
    """

    center = str(center)
    ligand = str(ligand)
    if center == ligand:
        raise ValueError("center and ligand species must be different")
    missing = {center, ligand}.difference(trajectory.species)
    if missing:
        raise ValueError(f"unknown species in bridging cluster definition: {sorted(missing)}")
    cutoff = float(r_center_ligand)
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("r_center_ligand must be finite and positive")
    modes = tuple(str(mode) for mode in sharing_modes)
    if not modes or len(set(modes)) != len(modes) or not set(modes).issubset(_ALL_SHARING_MODES):
        raise ValueError("sharing_modes must be unique values from corner, edge, face, connected")
    min_shared_ligands = int(min_shared_ligands)
    if min_shared_ligands < 1:
        raise ValueError("min_shared_ligands must be at least 1")

    frame_indices = trajectory.resolve_frame_indices(frames)
    plan = plan_execution(
        len(frame_indices),
        ncore=ncore,
        backend=backend,
        workload=len(frame_indices) * trajectory.n_atoms,
        auto_multiprocessing_threshold=1_500_000,
    )
    partials, execution = execute_frame_chunks(
        trajectory,
        frame_indices,
        plan,
        _bridging_chunk,
        (
            center,
            ligand,
            cutoff,
            modes,
            bool(include_unshared),
            min_shared_ligands,
            bool(include_isolated),
        ),
        show_progress=show_progress,
        description="Bridging-ligand clusters",
    )
    shared_histogram = np.zeros(1, dtype=np.int64)
    cluster_histograms = {mode: np.zeros(1, dtype=np.int64) for mode in modes}
    total_histogram = np.zeros(1, dtype=np.int64)
    finite_histogram = np.zeros(1, dtype=np.int64)
    frame_rows: list[dict[str, Any]] = []
    for partial in partials:
        shared_histogram = merge_integer_histograms(
            shared_histogram, partial["shared_histogram"]
        )
        for mode in modes:
            cluster_histograms[mode] = merge_integer_histograms(
                cluster_histograms[mode], partial["cluster_histograms"][mode]
            )
        total_histogram = merge_integer_histograms(
            total_histogram, partial["total_histogram"]
        )
        finite_histogram = merge_integer_histograms(
            finite_histogram, partial["finite_histogram"]
        )
        frame_rows.extend(partial["frame_rows"])

    shared_indices = np.flatnonzero(shared_histogram)
    shared_indices = shared_indices[shared_indices > 0]
    shared_total = int(shared_histogram.sum())
    sharing_distribution = pd.DataFrame(
        {
            "shared_ligands": shared_indices,
            "probability": (
                shared_histogram[shared_indices] / shared_total
                if shared_total
                else np.empty(0, dtype=float)
            ),
            "sharing_type": [_sharing_type(int(value)) for value in shared_indices],
        }
    )

    observed_max = max(
        [len(total_histogram) - 1, len(finite_histogram) - 1]
        + [len(histogram) - 1 for histogram in cluster_histograms.values()]
    )
    if max_cluster_size is None:
        resolved_max = max(1, observed_max)
    else:
        resolved_max = int(max_cluster_size)
        if resolved_max < 1:
            raise ValueError("max_cluster_size must be positive or None")
    sizes = np.arange(1, resolved_max + 1)
    cluster_distribution = pd.DataFrame({"cluster_size": sizes})
    sharing_overflow: dict[str, float] = {}
    for mode in modes:
        histogram = cluster_histograms[mode]
        total = int(histogram.sum())
        values = np.zeros(resolved_max, dtype=float)
        upper = min(len(histogram), resolved_max + 1)
        if total and upper > 1:
            values[: upper - 1] = histogram[1:upper] / total
        cluster_distribution[mode] = values
        sharing_overflow[mode] = (
            float(histogram[resolved_max + 1 :].sum() / total) if total else 0.0
        )
    cluster_distribution.attrs["overflow_probability"] = sharing_overflow

    percolation_distribution = pd.DataFrame({"cluster_size": sizes})
    for name, histogram in (
        ("mean_cluster_count_total", total_histogram),
        ("mean_cluster_count_finite", finite_histogram),
    ):
        values = np.zeros(resolved_max, dtype=float)
        upper = min(len(histogram), resolved_max + 1)
        if upper > 1:
            values[: upper - 1] = histogram[1:upper] / len(frame_indices)
        percolation_distribution[name] = values

    frame_summary = pd.DataFrame(frame_rows).sort_values("frame").reset_index(drop=True)
    metadata = {
        "method": "bridging_ligand",
        "center": center,
        "ligand": ligand,
        "r_center_ligand": cutoff,
        "sharing_modes": modes,
        "include_unshared": bool(include_unshared),
        "min_shared_ligands": min_shared_ligands,
        "include_isolated": bool(include_isolated),
        "execution": execution,
    }
    for table in (
        sharing_distribution,
        cluster_distribution,
        frame_summary,
        percolation_distribution,
    ):
        table.attrs["analysis"] = metadata
    return BridgingClusterResult(
        sharing_distribution=sharing_distribution,
        cluster_distribution=cluster_distribution,
        frame_summary=frame_summary,
        percolation_cluster_distribution=percolation_distribution,
        percolation_summary=_percolation_summary(frame_summary),
        metadata=metadata,
    )


analyze_polyhedra = analyze_bridging_clusters


__all__ = [
    "BridgingClusterResult",
    "SharingMode",
    "analyze_bridging_clusters",
    "analyze_polyhedra",
]
