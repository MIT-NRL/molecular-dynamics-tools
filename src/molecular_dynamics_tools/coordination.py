"""Cutoff and relative-angular-distance coordination analyses."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import freud
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._execution import Backend, execute_frame_chunks, plan_execution
from ._geometry import (
    accumulate_integer_histogram,
    freud_box,
    merge_integer_histograms,
)
from ._rad import (
    RADBondMode,
    normalize_rad_bond_mode,
    normalize_rad_variant,
    rad_neighbors,
)
from .trajectory import Trajectory


@dataclass(frozen=True, slots=True)
class CoordinationDefinition:
    """One center-neighbor coordination definition."""

    center: str
    neighbor: str
    r_min: float = 0.0
    r_max: float = 3.0
    label: str | None = None

    @property
    def name(self) -> str:
        return self.label or f"{self.center}-{self.neighbor}"


def _normalize_definitions(
    trajectory: Trajectory,
    definitions: Iterable[CoordinationDefinition | Sequence[object]],
    *,
    require_cutoff: bool,
) -> tuple[CoordinationDefinition, ...]:
    normalized: list[CoordinationDefinition] = []
    for raw in definitions:
        if isinstance(raw, CoordinationDefinition):
            definition = raw
        else:
            values = tuple(raw)
            if len(values) == 2 and not require_cutoff:
                definition = CoordinationDefinition(str(values[0]), str(values[1]))
            elif len(values) == 3:
                definition = CoordinationDefinition(
                    str(values[0]), str(values[1]), 0.0, float(values[2])
                )
            elif len(values) == 4:
                definition = CoordinationDefinition(
                    str(values[0]), str(values[1]), float(values[2]), float(values[3])
                )
            elif len(values) == 5:
                definition = CoordinationDefinition(
                    str(values[0]),
                    str(values[1]),
                    float(values[2]),
                    float(values[3]),
                    str(values[4]),
                )
            else:
                expected = "2, 3, 4, or 5" if not require_cutoff else "3, 4, or 5"
                raise ValueError(f"coordination definitions must contain {expected} values")
        if not definition.center or not definition.neighbor:
            raise ValueError("center and neighbor species must be nonempty")
        if require_cutoff and (
            not np.isfinite(definition.r_min)
            or not np.isfinite(definition.r_max)
            or definition.r_min < 0
            or definition.r_max <= definition.r_min
        ):
            raise ValueError("coordination cutoffs require 0 <= r_min < r_max")
        normalized.append(definition)

    if not normalized:
        raise ValueError("definitions must contain at least one coordination pair")
    names = [definition.name for definition in normalized]
    if len(set(names)) != len(names):
        raise ValueError("coordination labels must be unique; use CoordinationDefinition.label")
    missing = {
        species
        for definition in normalized
        for species in (definition.center, definition.neighbor)
        if species not in trajectory.species
    }
    if missing:
        raise ValueError(f"unknown species in coordination definitions: {sorted(missing)}")
    return tuple(normalized)


def _empty_histograms(
    definitions: Sequence[CoordinationDefinition],
) -> dict[str, NDArray[np.int64]]:
    return {definition.name: np.zeros(1, dtype=np.int64) for definition in definitions}


def _cutoff_chunk(
    trajectory: Trajectory,
    frame_indices: tuple[int, ...],
    definitions: tuple[CoordinationDefinition, ...],
) -> dict[str, NDArray[np.int64]]:
    histograms = _empty_histograms(definitions)
    for frame in trajectory.iter_frames(frame_indices):
        box = freud_box(frame.dimensions)
        for definition in definitions:
            center_positions = frame.positions_of(definition.center)
            neighbor_positions = frame.positions_of(definition.neighbor)
            query = freud.locality.AABBQuery(box, neighbor_positions)
            neighbors = query.query(
                center_positions,
                {
                    "mode": "ball",
                    "r_min": definition.r_min,
                    "r_max": definition.r_max,
                    "exclude_ii": definition.center == definition.neighbor,
                },
            ).toNeighborList()
            counts = np.bincount(
                np.asarray(neighbors.query_point_indices, dtype=np.int64),
                minlength=len(center_positions),
            )
            histograms[definition.name] = accumulate_integer_histogram(
                histograms[definition.name], counts
            )
    return histograms


def _rad_chunk(
    trajectory: Trajectory,
    frame_indices: tuple[int, ...],
    definitions: tuple[CoordinationDefinition, ...],
    bond_mode: RADBondMode,
    variant: str,
) -> dict[str, NDArray[np.int64]]:
    histograms = _empty_histograms(definitions)
    requested: dict[str, list[CoordinationDefinition]] = {}
    for definition in definitions:
        requested.setdefault(definition.center, []).append(definition)

    for frame in trajectory.iter_frames(frame_indices):
        box = freud_box(frame.dimensions)
        shells: dict[int, NDArray[np.int64]] = {}
        shell_sets: dict[int, set[int]] = {}

        def shell(index: int) -> NDArray[np.int64]:
            if index not in shells:
                shells[index] = rad_neighbors(
                    box, frame.positions, index, variant=variant
                )
            return shells[index]

        def shell_set(index: int) -> set[int]:
            if index not in shell_sets:
                shell_sets[index] = set(shell(index).tolist())
            return shell_sets[index]

        for center_species, center_definitions in requested.items():
            center_indices = np.flatnonzero(frame.species == center_species)
            counts = {
                definition.name: np.zeros(len(center_indices), dtype=np.int64)
                for definition in center_definitions
            }
            for local_index, center_index_raw in enumerate(center_indices):
                center_index = int(center_index_raw)
                neighbors = shell(center_index)
                if bond_mode == "mutual":
                    neighbors = np.asarray(
                        [index for index in neighbors if center_index in shell_set(int(index))],
                        dtype=np.int64,
                    )
                neighbor_species = frame.species[neighbors]
                for definition in center_definitions:
                    counts[definition.name][local_index] = np.count_nonzero(
                        neighbor_species == definition.neighbor
                    )
            for name, values in counts.items():
                histograms[name] = accumulate_integer_histogram(histograms[name], values)
    return histograms


def _merge_histogram_chunks(
    partials: Sequence[dict[str, NDArray[np.int64]]],
    definitions: Sequence[CoordinationDefinition],
) -> dict[str, NDArray[np.int64]]:
    merged = _empty_histograms(definitions)
    for partial in partials:
        for name, histogram in partial.items():
            merged[name] = merge_integer_histograms(merged[name], histogram)
    return merged


def _histograms_to_table(
    histograms: dict[str, NDArray[np.int64]],
    *,
    max_coordination: int | None,
) -> pd.DataFrame:
    observed_max = max(len(histogram) - 1 for histogram in histograms.values())
    if max_coordination is None:
        resolved_max = observed_max
    else:
        resolved_max = int(max_coordination)
        if resolved_max < 0:
            raise ValueError("max_coordination must be nonnegative or None")
    axis = np.arange(resolved_max + 1)
    result = pd.DataFrame({"coordination": axis})
    overflow: dict[str, float] = {}
    for name, histogram in histograms.items():
        total = int(histogram.sum())
        if total == 0:
            raise RuntimeError(f"no coordination samples were found for {name}")
        values = np.zeros(len(axis), dtype=float)
        upper = min(len(histogram), len(axis))
        values[:upper] = histogram[:upper] / total
        result[name] = values
        overflow[name] = float(histogram[len(axis) :].sum() / total)
    result.attrs["overflow_probability"] = overflow
    return result


def compute_coordination(
    trajectory: Trajectory,
    definitions: Iterable[CoordinationDefinition | Sequence[object]],
    *,
    frames: slice | Sequence[int] | None = None,
    max_coordination: int | None = None,
    ncore: int | None = 1,
    backend: Backend = "auto",
    show_progress: bool = False,
) -> pd.DataFrame:
    """Compute cutoff coordination-number distributions in one trajectory pass.

    Tuple definitions may be ``(center, neighbor, r_max)`` or
    ``(center, neighbor, r_min, r_max)``. Use :class:`CoordinationDefinition`
    to provide a custom output label.
    """

    normalized = _normalize_definitions(trajectory, definitions, require_cutoff=True)
    frame_indices = trajectory.resolve_frame_indices(frames)
    plan = plan_execution(
        len(frame_indices),
        ncore=ncore,
        backend=backend,
        workload=len(frame_indices) * trajectory.n_atoms * len(normalized),
        auto_multiprocessing_threshold=800_000,
    )
    partials, execution = execute_frame_chunks(
        trajectory,
        frame_indices,
        plan,
        _cutoff_chunk,
        (normalized,),
        show_progress=show_progress,
        description="Cutoff coordination",
    )
    result = _histograms_to_table(
        _merge_histogram_chunks(partials, normalized),
        max_coordination=max_coordination,
    )
    result.attrs["method"] = "cutoff"
    result.attrs["definitions"] = normalized
    result.attrs["execution"] = execution
    return result


def compute_rad_coordination(
    trajectory: Trajectory,
    definitions: Iterable[CoordinationDefinition | Sequence[object]],
    *,
    bond_mode: str = "directed",
    variant: str = "closed",
    frames: slice | Sequence[int] | None = None,
    max_coordination: int | None = None,
    ncore: int | None = 1,
    backend: Backend = "auto",
    show_progress: bool = False,
) -> pd.DataFrame:
    """Compute relative-angular-distance (RAD) coordination distributions.

    ``bond_mode='directed'`` uses each center's shell. ``'mutual'`` keeps a
    neighbor only when both atoms include one another in their RAD shells.
    The implemented ``variant='closed'`` closes a shell at its first blocked
    candidate. Cutoffs in legacy four-value definitions are accepted but ignored.
    """

    normalized = _normalize_definitions(trajectory, definitions, require_cutoff=False)
    resolved_mode = normalize_rad_bond_mode(bond_mode)
    resolved_variant = normalize_rad_variant(variant)
    frame_indices = trajectory.resolve_frame_indices(frames)
    center_count = sum(
        trajectory.atom_counts[species]
        for species in {definition.center for definition in normalized}
    )
    plan = plan_execution(
        len(frame_indices),
        ncore=ncore,
        backend=backend,
        workload=len(frame_indices) * trajectory.n_atoms * center_count,
        auto_multiprocessing_threshold=15_000_000,
    )
    partials, execution = execute_frame_chunks(
        trajectory,
        frame_indices,
        plan,
        _rad_chunk,
        (normalized, resolved_mode, resolved_variant),
        show_progress=show_progress,
        description=f"RAD coordination ({resolved_mode})",
    )
    result = _histograms_to_table(
        _merge_histogram_chunks(partials, normalized),
        max_coordination=max_coordination,
    )
    result.attrs["method"] = "relative_angular_distance"
    result.attrs["bond_mode"] = resolved_mode
    result.attrs["variant"] = resolved_variant
    result.attrs["definitions"] = normalized
    result.attrs["execution"] = execution
    return result


def summarize_coordination(distribution: pd.DataFrame) -> pd.DataFrame:
    """Return the mean, variance, and standard deviation of each distribution."""

    if "coordination" not in distribution:
        raise KeyError("distribution must contain a 'coordination' column")
    axis = distribution["coordination"].to_numpy(dtype=float)
    rows: list[dict[str, float | str]] = []
    overflow = distribution.attrs.get("overflow_probability", {})
    for column in distribution.columns:
        if column == "coordination":
            continue
        probability = distribution[column].to_numpy(dtype=float)
        represented = float(np.nansum(probability))
        if represented <= 0:
            mean = variance = float("nan")
        else:
            normalized = probability / represented
            mean = float(np.nansum(axis * normalized))
            variance = float(np.nansum((axis - mean) ** 2 * normalized))
        rows.append(
            {
                "pair": column,
                "mean": mean,
                "variance": variance,
                "sd": float(np.sqrt(variance)),
                "represented_probability": represented,
                "overflow_probability": float(overflow.get(column, 0.0)),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "CoordinationDefinition",
    "compute_coordination",
    "compute_rad_coordination",
    "summarize_coordination",
]
