"""Bond-angle distribution calculations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import islice

import freud
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from ._execution import Backend, execute_frame_chunks, plan_execution
from ._geometry import freud_box, neighbor_map
from .trajectory import Trajectory

_CENTER_BATCH_SIZE = 512
_MAX_BATCH_NEIGHBORS = 4096
_PAIR_BLOCK_SIZE = 64


def _center_batches(first_map, third_map):
    batch, neighbors = [], 0
    for center in sorted(first_map.keys() & third_map.keys()):
        size = len(first_map[center]) + len(third_map[center])
        if batch and (len(batch) >= _CENTER_BATCH_SIZE or neighbors + size > _MAX_BATCH_NEIGHBORS):
            yield batch
            batch, neighbors = [], 0
        batch.append(center)
        neighbors += size
    if batch:
        yield batch


def _prepare_center_batch(
    box, centers, first_positions, center_positions, third_positions, first_map, third_map
):
    """Perform read-only Freud geometry in the parent, before dispatching workers."""
    sizes, displacements = [], []
    for center in centers:
        first_indices, third_indices = first_map[center], third_map[center]
        sizes.append((len(first_indices), len(third_indices)))
        displacements.extend(
            (
                first_positions[first_indices] - center_positions[center],
                third_positions[third_indices] - center_positions[center],
            )
        )
    displacements = np.concatenate(displacements)
    vectors = np.empty(displacements.shape, dtype=np.float64)
    for start in range(0, len(vectors), _MAX_BATCH_NEIGHBORS):
        stop = start + _MAX_BATCH_NEIGHBORS
        vectors[start:stop] = box.wrap(displacements[start:stop])
    norms = np.linalg.norm(vectors, axis=1)
    prepared, offset = [], 0
    for center, (first_size, third_size) in zip(centers, sizes, strict=True):
        first_vectors = vectors[offset : offset + first_size]
        first_norms = norms[offset : offset + first_size]
        offset += first_size
        third_vectors = vectors[offset : offset + third_size]
        third_norms = norms[offset : offset + third_size]
        offset += third_size
        first_valid, third_valid = first_norms > 0, third_norms > 0
        if not np.any(first_valid) or not np.any(third_valid):
            continue
        prepared.append(
            (
                first_map[center][first_valid],
                third_map[center][third_valid],
                first_vectors[first_valid] / first_norms[first_valid, None],
                third_vectors[third_valid] / third_norms[third_valid, None],
            )
        )
    return prepared


def _counts_for_center_batch(prepared, bin_edges, same_outer, symmetric_shell):
    """Accumulate a bounded center batch without changing native thread settings."""
    histogram = np.zeros(len(bin_edges) - 1, dtype=np.int64)
    small = {}
    for first_indices, third_indices, first_unit, third_unit in prepared:
        first_size, third_size = len(first_indices), len(third_indices)
        if symmetric_shell and first_size < 2:
            continue
        if first_size <= 8 and third_size <= 8:
            small.setdefault((first_size, third_size), []).append(
                (first_indices, third_indices, first_unit, third_unit)
            )
            continue
        # Bound pair scratch for highly coordinated centers rather than making
        # a full local coordination-squared cosine matrix.
        column_size = first_size if symmetric_shell else third_size
        if symmetric_shell and third_size < first_size:
            raise IndexError("symmetric angle neighbor shells have incompatible sizes")
        for first in range(0, first_size, _PAIR_BLOCK_SIZE):
            left_stop = min(first_size, first + _PAIR_BLOCK_SIZE)
            for third in range(0, column_size, _PAIR_BLOCK_SIZE):
                right_stop = min(column_size, third + _PAIR_BLOCK_SIZE)
                if symmetric_shell and right_stop <= first:
                    continue
                cosine = np.clip(
                    first_unit[first:left_stop] @ third_unit[third:right_stop].T, -1, 1
                )
                if symmetric_shell:
                    mask = np.arange(first, left_stop)[:, None] < np.arange(third, right_stop)
                    values = cosine[mask]
                elif same_outer:
                    mask = (
                        first_indices[first:left_stop, None]
                        != third_indices[None, third:right_stop]
                    )
                    values = cosine[mask]
                else:
                    values = cosine.ravel()
                histogram += np.histogram(np.degrees(np.arccos(values)), bins=bin_edges)[0]
    for (first_size, _third_size), centers in small.items():
        first_unit = np.stack([center[2] for center in centers])
        third_unit = np.stack([center[3] for center in centers])
        cosine = np.clip(first_unit @ np.swapaxes(third_unit, -2, -1), -1, 1)
        if symmetric_shell:
            left, right = np.triu_indices(first_size, k=1)
            values = cosine[:, left, right]
        elif same_outer:
            first_indices = np.stack([center[0] for center in centers])
            third_indices = np.stack([center[1] for center in centers])
            values = cosine[first_indices[:, :, None] != third_indices[:, None, :]]
        else:
            values = cosine.ravel()
        histogram += np.histogram(np.degrees(np.arccos(values)), bins=bin_edges)[0]
    return histogram


@dataclass(frozen=True, slots=True)
class AngleDefinition:
    """An outer-center-outer bond angle and its two bond cutoffs."""

    first: str
    center: str
    third: str
    first_center_max: float
    center_third_max: float
    label: str | None = None

    @property
    def name(self) -> str:
        return self.label or f"{self.first}-{self.center}-{self.third}"


def _normalize_definitions(
    trajectory: Trajectory,
    definitions: Iterable[AngleDefinition | Sequence[object]],
) -> tuple[AngleDefinition, ...]:
    normalized: list[AngleDefinition] = []
    for raw in definitions:
        if isinstance(raw, AngleDefinition):
            definition = raw
        else:
            values = tuple(raw)
            if len(values) not in {5, 6}:
                raise ValueError("angle definitions must contain 5 values, plus an optional label")
            definition = AngleDefinition(
                str(values[0]),
                str(values[1]),
                str(values[2]),
                float(values[3]),
                float(values[4]),
                None if len(values) == 5 else str(values[5]),
            )
        if (
            not np.isfinite(definition.first_center_max)
            or not np.isfinite(definition.center_third_max)
            or definition.first_center_max <= 0
            or definition.center_third_max <= 0
        ):
            raise ValueError("bond cutoffs must be finite and positive")
        normalized.append(definition)
    if not normalized:
        raise ValueError("definitions must contain at least one bond angle")
    names = [definition.name for definition in normalized]
    if len(set(names)) != len(names):
        raise ValueError("angle labels must be unique; use AngleDefinition.label")
    missing = {
        species
        for definition in normalized
        for species in (definition.first, definition.center, definition.third)
        if species not in trajectory.species
    }
    if missing:
        raise ValueError(f"unknown species in angle definitions: {sorted(missing)}")
    return tuple(normalized)


def _angle_counts_for_frame(
    box: freud.Box,
    positions: NDArray[np.float32],
    species: NDArray[np.str_],
    definition: AngleDefinition,
    bin_edges: NDArray[np.float64],
    ncore: int = 1,
) -> NDArray[np.int64]:
    first_positions = positions[species == definition.first]
    center_positions = positions[species == definition.center]
    third_positions = positions[species == definition.third]

    first_neighbors = (
        freud.locality.AABBQuery(box, first_positions)
        .query(
            center_positions,
            {"mode": "ball", "r_max": definition.first_center_max},
        )
        .toNeighborList()
    )
    third_neighbors = (
        freud.locality.AABBQuery(box, third_positions)
        .query(
            center_positions,
            {"mode": "ball", "r_max": definition.center_third_max},
        )
        .toNeighborList()
    )
    first_map = neighbor_map(first_neighbors)
    third_map = neighbor_map(third_neighbors)
    histogram = np.zeros(len(bin_edges) - 1, dtype=np.int64)
    same_outer = definition.first == definition.third
    symmetric_shell = same_outer and np.isclose(
        definition.first_center_max, definition.center_third_max
    )

    batches = iter(_center_batches(first_map, third_map))
    pool = ThreadPoolExecutor(max_workers=ncore) if ncore > 1 else nullcontext()
    # Native neighbor queries finish before thread workers run. During the
    # thread phase BLAS stays single-threaded, honoring ncore as a total budget.
    limits = threadpool_limits(limits=1) if ncore > 1 else nullcontext()
    with limits, pool as executor:
        while selected := list(islice(batches, ncore)):
            prepared = [
                _prepare_center_batch(
                    box,
                    centers,
                    first_positions,
                    center_positions,
                    third_positions,
                    first_map,
                    third_map,
                )
                for centers in selected
            ]
            if executor:
                futures = [
                    executor.submit(
                        _counts_for_center_batch, centers, bin_edges, same_outer, symmetric_shell
                    )
                    for centers in prepared
                ]
                for future in futures:
                    histogram += future.result()
            else:
                for centers in prepared:
                    histogram += _counts_for_center_batch(
                        centers, bin_edges, same_outer, symmetric_shell
                    )
    return histogram


def _angle_chunk(
    trajectory: Trajectory,
    frame_indices: tuple[int, ...],
    definitions: tuple[AngleDefinition, ...],
    bin_edges: NDArray[np.float64],
    center_threads: int = 1,
) -> dict[str, NDArray[np.int64]]:
    histograms = {
        definition.name: np.zeros(len(bin_edges) - 1, dtype=np.int64) for definition in definitions
    }
    for frame in trajectory.iter_frames(frame_indices):
        box = freud_box(frame.dimensions)
        for definition in definitions:
            histograms[definition.name] += _angle_counts_for_frame(
                box, frame.positions, frame.species, definition, bin_edges, ncore=center_threads
            )
    return histograms


def compute_bond_angles(
    trajectory: Trajectory,
    definitions: Iterable[AngleDefinition | Sequence[object]],
    *,
    bins: int = 180,
    angle_range: tuple[float, float] = (0.0, 180.0),
    frames: slice | Sequence[int] | None = None,
    ncore: int | None = 1,
    backend: Backend = "auto",
    show_progress: bool = False,
) -> pd.DataFrame:
    """Compute one or more normalized bond-angle probability densities.

    Tuple definitions have the form ``(first, center, third,
    first_center_max, center_third_max)``. All definitions are accumulated in
    one pass over each frame chunk. ``ncore`` is the total CPU budget: the
    serial frame backend shares bounded center batches among threads, while
    multiprocessing frame workers each compute their centers single-threaded.
    Coordinates and neighbor maps are shared by center threads, with no
    all-atom pair matrix or full-trajectory coordinate cache.
    """

    normalized = _normalize_definitions(trajectory, definitions)
    resolved_bins = int(bins)
    lower, upper = (float(value) for value in angle_range)
    if resolved_bins < 1:
        raise ValueError("bins must be positive")
    if not np.isfinite(lower) or not np.isfinite(upper) or not 0 <= lower < upper <= 180:
        raise ValueError("angle_range must satisfy 0 <= lower < upper <= 180")
    bin_edges = np.linspace(lower, upper, resolved_bins + 1)
    frame_indices = trajectory.resolve_frame_indices(frames)
    plan = plan_execution(
        len(frame_indices),
        ncore=ncore,
        backend=backend,
        workload=len(frame_indices) * trajectory.n_atoms * len(normalized),
        auto_multiprocessing_threshold=400_000,
    )
    partials, execution = execute_frame_chunks(
        trajectory,
        frame_indices,
        plan,
        _angle_chunk,
        (normalized, bin_edges, plan.threads_per_worker),
        show_progress=show_progress,
        description="Bond angles",
    )
    histograms = {
        definition.name: np.zeros(resolved_bins, dtype=np.int64) for definition in normalized
    }
    for partial in partials:
        for name, histogram in partial.items():
            histograms[name] += histogram

    widths = np.diff(bin_edges)
    result = pd.DataFrame({"angle": 0.5 * (bin_edges[:-1] + bin_edges[1:])})
    sample_counts: dict[str, int] = {}
    for definition in normalized:
        histogram = histograms[definition.name]
        sample_count = int(histogram.sum())
        if sample_count == 0:
            raise RuntimeError(f"no angles were found for {definition.name}")
        result[definition.name] = histogram / (sample_count * widths)
        sample_counts[definition.name] = sample_count
    result.attrs["definitions"] = normalized
    result.attrs["sample_counts"] = sample_counts
    result.attrs["angle_range"] = (lower, upper)
    result.attrs["bins"] = resolved_bins
    result.attrs["execution"] = {
        **execution,
        "center_backend": "shared_threads" if plan.threads_per_worker > 1 else "serial",
        "center_threads_per_worker": plan.threads_per_worker,
        "center_batch_size": _CENTER_BATCH_SIZE,
        "center_batch_neighbor_limit": _MAX_BATCH_NEIGHBORS,
    }
    return result


__all__ = ["AngleDefinition", "compute_bond_angles"]
