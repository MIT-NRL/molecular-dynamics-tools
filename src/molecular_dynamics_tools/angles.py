"""Bond-angle distribution calculations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import freud
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._execution import Backend, execute_frame_chunks, plan_execution
from ._geometry import freud_box, neighbor_map
from .trajectory import Trajectory


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
) -> NDArray[np.int64]:
    first_positions = positions[species == definition.first]
    center_positions = positions[species == definition.center]
    third_positions = positions[species == definition.third]

    first_neighbors = freud.locality.AABBQuery(box, first_positions).query(
        center_positions,
        {"mode": "ball", "r_max": definition.first_center_max},
    ).toNeighborList()
    third_neighbors = freud.locality.AABBQuery(box, third_positions).query(
        center_positions,
        {"mode": "ball", "r_max": definition.center_third_max},
    ).toNeighborList()
    first_map = neighbor_map(first_neighbors)
    third_map = neighbor_map(third_neighbors)
    histogram = np.zeros(len(bin_edges) - 1, dtype=np.int64)
    same_outer = definition.first == definition.third
    symmetric_shell = same_outer and np.isclose(
        definition.first_center_max, definition.center_third_max
    )

    for center_index in first_map.keys() & third_map.keys():
        first_indices = first_map[center_index]
        third_indices = third_map[center_index]
        center_position = center_positions[center_index]
        first_vectors = np.asarray(
            box.wrap(first_positions[first_indices] - center_position), dtype=np.float64
        )
        third_vectors = np.asarray(
            box.wrap(third_positions[third_indices] - center_position), dtype=np.float64
        )
        first_norms = np.linalg.norm(first_vectors, axis=1)
        third_norms = np.linalg.norm(third_vectors, axis=1)
        first_valid = first_norms > 0
        third_valid = third_norms > 0
        if not np.any(first_valid) or not np.any(third_valid):
            continue
        first_indices = first_indices[first_valid]
        third_indices = third_indices[third_valid]
        first_unit = first_vectors[first_valid] / first_norms[first_valid, None]
        third_unit = third_vectors[third_valid] / third_norms[third_valid, None]
        cosine = np.clip(first_unit @ third_unit.T, -1.0, 1.0)

        if symmetric_shell:
            if len(first_indices) < 2:
                continue
            left, right = np.triu_indices(len(first_indices), k=1)
            cosine_values = cosine[left, right]
        elif same_outer:
            cosine_values = cosine[first_indices[:, None] != third_indices[None, :]]
        else:
            cosine_values = cosine.ravel()
        if cosine_values.size:
            angles = np.degrees(np.arccos(cosine_values))
            histogram += np.histogram(angles, bins=bin_edges)[0]
    return histogram


def _angle_chunk(
    trajectory: Trajectory,
    frame_indices: tuple[int, ...],
    definitions: tuple[AngleDefinition, ...],
    bin_edges: NDArray[np.float64],
) -> dict[str, NDArray[np.int64]]:
    histograms = {
        definition.name: np.zeros(len(bin_edges) - 1, dtype=np.int64)
        for definition in definitions
    }
    for frame in trajectory.iter_frames(frame_indices):
        box = freud_box(frame.dimensions)
        for definition in definitions:
            histograms[definition.name] += _angle_counts_for_frame(
                box, frame.positions, frame.species, definition, bin_edges
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
    one pass over each frame chunk.
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
        (normalized, bin_edges),
        show_progress=show_progress,
        description="Bond angles",
    )
    histograms = {
        definition.name: np.zeros(resolved_bins, dtype=np.int64)
        for definition in normalized
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
    result.attrs["execution"] = execution
    return result


__all__ = ["AngleDefinition", "compute_bond_angles"]
