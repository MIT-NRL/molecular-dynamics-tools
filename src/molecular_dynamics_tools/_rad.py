"""Internal relative-angular-distance (RAD) geometry helpers."""

from __future__ import annotations

from typing import Literal

import freud
import numpy as np
from numpy.typing import NDArray

RADBondMode = Literal["directed", "mutual"]
RADVariant = Literal["closed"]


def normalize_rad_bond_mode(mode: str) -> RADBondMode:
    """Normalize directed and reciprocal RAD bond-mode aliases."""

    normalized = str(mode).strip().lower().replace("-", "_")
    aliases: dict[str, RADBondMode] = {
        "directed": "directed",
        "center": "directed",
        "one_way": "directed",
        "rad": "directed",
        "mutual": "mutual",
        "and": "mutual",
        "symmetric": "mutual",
        "intersection": "mutual",
        "rad_and": "mutual",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError("bond_mode must be 'directed' or 'mutual'") from exc


def normalize_rad_variant(variant: str) -> RADVariant:
    """Validate the supported Higham--Henchman RAD variant."""

    normalized = str(variant).strip().lower().replace("-", "_")
    if normalized in {"closed", "rad_closed"}:
        return "closed"
    raise ValueError("variant must be 'closed'")


def rad_neighbors_with_distances(
    box: freud.Box,
    positions: NDArray[np.floating],
    center_index: int,
    *,
    variant: str = "closed",
    distance_tolerance: float = 1e-8,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Return a center's RAD neighbor indices and minimum-image distances.

    ``closed`` is the RAD-closed construction: candidates are considered in
    increasing distance order and the shell closes at the first blocked
    candidate. Coincident coordinates within ``distance_tolerance`` are
    excluded, including the center itself.
    """

    normalize_rad_variant(variant)
    tolerance = float(distance_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("distance_tolerance must be finite and nonnegative")

    coordinates = np.asarray(positions, dtype=np.float64)
    center = int(center_index)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3)")
    if center < 0 or center >= len(coordinates):
        raise IndexError("center_index is outside positions")

    displacements = np.asarray(
        box.wrap(coordinates - coordinates[center]), dtype=np.float64
    )
    squared_distances = np.einsum("ij,ij->i", displacements, displacements)
    candidates = np.flatnonzero(squared_distances > tolerance**2)
    if candidates.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    order = np.argsort(squared_distances[candidates], kind="stable")
    candidates = candidates[order]

    accepted: list[int] = []
    closer_vectors: list[NDArray[np.float64]] = []
    closer_distances: list[float] = []
    accepted_distances: list[float] = []
    for candidate_raw in candidates:
        candidate = int(candidate_raw)
        distance_squared = float(squared_distances[candidate])
        distance = float(np.sqrt(distance_squared))
        if closer_vectors:
            blockers = np.asarray(closer_vectors)
            blocker_distances = np.asarray(closer_distances)
            cosine = (blockers @ displacements[candidate]) / (
                blocker_distances * distance
            )
            if np.any((1.0 / distance_squared) <= cosine / blocker_distances**2):
                break
        accepted.append(candidate)
        accepted_distances.append(distance)
        closer_vectors.append(displacements[candidate])
        closer_distances.append(distance)
    return (
        np.asarray(accepted, dtype=np.int64),
        np.asarray(accepted_distances, dtype=np.float64),
    )


def rad_neighbors(
    box: freud.Box,
    positions: NDArray[np.floating],
    center_index: int,
    *,
    variant: str = "closed",
    distance_tolerance: float = 1e-8,
) -> NDArray[np.int64]:
    """Return a center's RAD neighbor indices."""

    return rad_neighbors_with_distances(
        box,
        positions,
        center_index,
        variant=variant,
        distance_tolerance=distance_tolerance,
    )[0]


__all__ = [
    "RADBondMode",
    "RADVariant",
    "normalize_rad_bond_mode",
    "normalize_rad_variant",
    "rad_neighbors",
    "rad_neighbors_with_distances",
]
