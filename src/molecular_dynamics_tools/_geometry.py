"""Small geometry and histogram kernels shared by structural analyses."""

from __future__ import annotations

from collections.abc import Sequence

import freud
import numpy as np
from numpy.typing import NDArray


def freud_box(dimensions: Sequence[float]) -> freud.Box:
    """Convert MDAnalysis ``[a, b, c, alpha, beta, gamma]`` dimensions."""

    lengths = np.asarray(dimensions[:3], dtype=float)
    angles = np.asarray(dimensions[3:6], dtype=float)
    if np.any(lengths <= 0) or not np.all(np.isfinite(lengths)):
        raise ValueError("periodic box lengths must be finite and positive")
    if np.allclose(angles, 90.0, atol=1e-7):
        return freud.Box(Lx=lengths[0], Ly=lengths[1], Lz=lengths[2])
    return freud.Box.from_box_lengths_and_angles(
        lengths[0],
        lengths[1],
        lengths[2],
        np.radians(angles[0]),
        np.radians(angles[1]),
        np.radians(angles[2]),
    )


def safe_periodic_radius(dimensions: Sequence[float]) -> float:
    """Return Freud's maximum unique-image query radius for a periodic cell.

    For a triclinic cell this is half the smallest perpendicular distance
    between opposite faces, not half the shortest lattice-vector length.
    The result is moved down by one float32 step because Freud stores box and
    query geometry at single precision.
    """

    box = freud_box(dimensions)
    vectors = (np.asarray(box.v1), np.asarray(box.v2), np.asarray(box.v3))
    volume = abs(float(np.linalg.det(np.column_stack(vectors))))
    plane_distances = (
        volume / np.linalg.norm(np.cross(vectors[1], vectors[2])),
        volume / np.linalg.norm(np.cross(vectors[2], vectors[0])),
        volume / np.linalg.norm(np.cross(vectors[0], vectors[1])),
    )
    half_distance = np.float32(0.5 * min(plane_distances))
    return float(np.nextafter(half_distance, np.float32(0.0)))


def accumulate_integer_histogram(
    histogram: NDArray[np.int64], values: Sequence[int] | NDArray[np.integer]
) -> NDArray[np.int64]:
    """Accumulate nonnegative integer samples into a dynamically sized array."""

    samples = np.asarray(values, dtype=np.int64)
    if samples.size == 0:
        return histogram
    if np.any(samples < 0):
        raise ValueError("histogram samples must be nonnegative")
    partial = np.bincount(samples)
    if len(histogram) < len(partial):
        histogram = np.pad(histogram, (0, len(partial) - len(histogram)))
    histogram[: len(partial)] += partial
    return histogram


def merge_integer_histograms(
    base: NDArray[np.int64], incoming: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Merge two dynamic integer histograms."""

    if len(base) < len(incoming):
        base = np.pad(base, (0, len(incoming) - len(base)))
    base[: len(incoming)] += incoming
    return base


def neighbor_map(neighbor_list: freud.locality.NeighborList) -> dict[int, NDArray[np.int64]]:
    """Group point indices by query-point index."""

    query = np.asarray(neighbor_list.query_point_indices, dtype=np.int64)
    points = np.asarray(neighbor_list.point_indices, dtype=np.int64)
    if query.size == 0:
        return {}
    order = np.argsort(query, kind="stable")
    query = query[order]
    points = points[order]
    unique, starts, counts = np.unique(query, return_index=True, return_counts=True)
    return {
        int(index): points[start : start + count]
        for index, start, count in zip(unique, starts, counts, strict=True)
    }


__all__ = [
    "accumulate_integer_histogram",
    "freud_box",
    "merge_integer_histograms",
    "neighbor_map",
    "safe_periodic_radius",
]
