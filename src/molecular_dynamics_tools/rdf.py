"""Partial radial-distribution-function calculations."""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations_with_replacement
from numbers import Integral
from typing import Literal

import freud
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._execution import (
    Backend,
    apply_worker_limits,
    contiguous_chunks,
    freud_thread_limit,
    plan_execution,
    restricted_cpu_affinity,
)
from ._geometry import freud_box as _freud_box
from ._geometry import safe_periodic_radius
from .trajectory import Trajectory

Pair = tuple[str, str]
ModeSpec = int | Literal["auto"] | Mapping[Pair, int]
_WORKER_TRAJECTORY: Trajectory | None = None


def _normalize_pairs(
    trajectory: Trajectory, pairs: Iterable[Pair] | None
) -> tuple[Pair, ...]:
    if pairs is None:
        atom_counts = trajectory.atom_counts
        normalized = tuple(
            pair
            for pair in combinations_with_replacement(trajectory.species, 2)
            if pair[0] != pair[1] or atom_counts[pair[0]] > 1
        )
    else:
        normalized = tuple((str(left), str(right)) for left, right in pairs)
    if not normalized:
        raise ValueError("pairs must contain at least one atom pair")
    if len(set(normalized)) != len(normalized):
        raise ValueError("pairs must not contain duplicates")
    known_species = set(trajectory.species)
    for left, right in normalized:
        missing = {left, right}.difference(known_species)
        if missing:
            raise ValueError(
                f"pair {left}-{right} contains unknown species: {sorted(missing)}"
            )
        if "-" in left or "-" in right:
            raise ValueError("species names containing '-' are not supported")
    return normalized


def _resolve_radius_range(
    trajectory: Trajectory,
    frame_indices: Sequence[int],
    r_min: float,
    r_max: float | None,
) -> tuple[float, float]:
    r_min = float(r_min)
    if not np.isfinite(r_min) or r_min < 0:
        raise ValueError("r_min must be finite and nonnegative")
    known = [trajectory.source.frames[index].dimensions for index in frame_indices]
    if all(dimensions is not None for dimensions in known):
        safe_by_frame = [
            (trajectory.source.frames[index].source_index, safe_periodic_radius(dimensions))
            for index, dimensions in zip(frame_indices, known, strict=True)
            if dimensions is not None
        ]
    else:
        safe_by_frame = [
            (frame.source_index, safe_periodic_radius(frame.dimensions))
            for frame in trajectory.iter_frames(frame_indices)
        ]
    limiting_frame, safe_max = min(safe_by_frame, key=lambda item: item[1])
    if r_max is None:
        if safe_max <= r_min:
            raise ValueError(
                f"r_min={r_min:g} leaves no usable range below the safe periodic "
                f"radius {safe_max:g} at source frame {limiting_frame}"
            )
        return r_min, safe_max

    resolved = float(r_max)
    if not np.isfinite(resolved) or resolved <= r_min:
        raise ValueError("r_max must be finite and greater than r_min")
    if resolved > safe_max:
        raise ValueError(
            f"r_max={resolved:g} exceeds the safe periodic radius {safe_max:g} "
            f"at source frame {limiting_frame}; reduce r_range[1] or use None"
        )
    return r_min, resolved


def _resolve_bin_edges(
    r_min: float,
    r_max: float,
    bins: int | None,
    step: float | None,
) -> NDArray[np.float64]:
    """Construct uniform edges from either a bin count or exact step size."""

    if bins is not None and step is not None:
        raise ValueError("pass either bins or step, not both")
    if bins is None and step is None:
        bins = 800

    if step is not None:
        resolved_step = float(step)
        if not np.isfinite(resolved_step) or resolved_step <= 0:
            raise ValueError("step must be finite and positive")
        bin_count = int(np.floor((r_max - r_min) / resolved_step + 1e-12))
        if bin_count < 1:
            raise ValueError("step is larger than the selected radial range")
        return r_min + resolved_step * np.arange(
            bin_count + 1, dtype=np.float64
        )

    assert bins is not None
    resolved_bins = int(bins)
    if resolved_bins < 1:
        raise ValueError("bins must be a positive integer")
    return np.linspace(r_min, r_max, resolved_bins + 1, dtype=np.float64)


def _resolve_evaluation_grid(
    r_min: float, r_max: float, step: float
) -> NDArray[np.float64]:
    """Construct an exact-step sampling grid for a continuous RDF."""

    resolved_step = float(step)
    if not np.isfinite(resolved_step) or resolved_step <= 0:
        raise ValueError("step must be finite and positive")
    interval_count = int(np.floor((r_max - r_min) / resolved_step + 1e-12))
    if interval_count < 1:
        raise ValueError("step is larger than the selected radial range")
    return r_min + resolved_step * np.arange(interval_count + 1, dtype=np.float64)


def _canonical_pair(pair: Pair) -> Pair:
    left, right = sorted(pair)
    return left, right


def _validate_mode_count(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{context} must be a positive integer")
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{context} must be a positive integer")
    return resolved


def _resolve_pair_modes(pairs: Sequence[Pair], modes: ModeSpec) -> tuple[int, ...]:
    """Resolve one spectral cutoff for each requested pair."""

    if not isinstance(modes, Mapping):
        resolved = _validate_mode_count(modes, context="modes")
        return (resolved,) * len(pairs)

    requested = {_canonical_pair(pair): pair for pair in pairs}
    supplied: dict[Pair, int] = {}
    for key, value in modes.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("mode mapping keys must be two-species tuples")
        pair = (str(key[0]), str(key[1]))
        canonical = _canonical_pair(pair)
        if canonical in supplied:
            raise ValueError("duplicate mode mapping for " + "-".join(canonical))
        supplied[canonical] = _validate_mode_count(
            value, context="modes for " + "-".join(pair)
        )

    missing = set(requested).difference(supplied)
    extra = set(supplied).difference(requested)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(
                "missing " + ", ".join("-".join(pair) for pair in sorted(missing))
            )
        if extra:
            details.append(
                "unexpected " + ", ".join("-".join(pair) for pair in sorted(extra))
            )
        raise ValueError(
            "mode mapping does not match requested pairs: " + "; ".join(details)
        )
    return tuple(supplied[_canonical_pair(pair)] for pair in pairs)


def _resolve_auto_mode_parameters(
    frame_count: int,
    atom_count: int,
    auto_pilot_frames: int | None,
    auto_min_modes: int,
    auto_max_modes: int,
) -> tuple[int, int, int]:
    """Validate automatic-mode controls and choose a bounded pilot size."""

    minimum = _validate_mode_count(auto_min_modes, context="auto_min_modes")
    maximum = _validate_mode_count(auto_max_modes, context="auto_max_modes")
    if maximum < minimum + 5:
        raise ValueError(
            "auto_max_modes must be at least five greater than auto_min_modes"
        )
    if auto_pilot_frames is None:
        estimated = int(np.ceil(100_000 / max(1, atom_count)))
        pilot_count = min(frame_count, max(128, min(1024, estimated)))
    else:
        pilot_count = _validate_mode_count(
            auto_pilot_frames, context="auto_pilot_frames"
        )
        if pilot_count > frame_count:
            raise ValueError(
                f"auto_pilot_frames={pilot_count} exceeds the {frame_count} "
                "selected frames"
            )
    return pilot_count, minimum, maximum


def _fit_mode_elbow(
    coefficients: NDArray[np.float64],
    minimum: int,
    maximum: int,
) -> dict[str, float | int | bool]:
    """Fit exponential coefficient decay followed by a constant noise floor."""

    amplitudes = np.abs(np.asarray(coefficients[1 : maximum + 1], dtype=float))
    positive = amplitudes[amplitudes > 0]
    numerical_floor = (
        np.finfo(float).tiny if not len(positive) else float(positive.min()) * 1e-6
    )
    log_amplitudes = np.log(np.maximum(amplitudes, numerical_floor))
    mode_numbers = np.arange(1, maximum + 1, dtype=float)
    best: tuple[float, int, float, float] | None = None
    for cutoff in range(minimum, maximum - 4):
        predictor = np.minimum(mode_numbers, float(cutoff))
        design = np.column_stack((np.ones_like(predictor), predictor))
        intercept, slope = np.linalg.lstsq(
            design, log_amplitudes, rcond=None
        )[0]
        if slope >= 0:
            continue
        residual = log_amplitudes - (intercept + slope * predictor)
        mean_square = float(np.mean(residual**2))
        candidate = (mean_square, cutoff, float(intercept), float(slope))
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:
        return {
            "selected_mode": minimum,
            "decay_rate": 0.0,
            "log_noise_floor": float(np.median(log_amplitudes)),
            "fit_rmse": float("inf"),
            "fallback": True,
        }
    mean_square, cutoff, intercept, slope = best
    return {
        "selected_mode": cutoff,
        "decay_rate": -slope,
        "log_noise_floor": intercept + slope * cutoff,
        "fit_rmse": float(np.sqrt(mean_square)),
        "fallback": False,
    }


def _accumulate_rdf_chunk(
    trajectory: Trajectory,
    frame_indices: Sequence[int],
    pairs: Sequence[Pair],
    bin_edges: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    counts = np.zeros((len(pairs), len(bin_edges) - 1), dtype=np.int64)
    ideal_pair_scales = np.zeros(len(pairs), dtype=np.float64)
    required_species = tuple(sorted({name for pair in pairs for name in pair}))
    r_min, r_max = float(bin_edges[0]), float(bin_edges[-1])

    for frame in trajectory.iter_frames(frame_indices):
        safe_max = safe_periodic_radius(frame.dimensions)
        if r_max > safe_max:
            raise ValueError(
                f"r_max={r_max:g} exceeds the safe periodic cutoff "
                f"{safe_max:g} at source frame {frame.source_index}"
            )
        box = _freud_box(frame.dimensions)
        positions_by_species = {
            name: np.asarray(box.wrap(frame.positions_of(name)), dtype=np.float32)
            for name in required_species
        }
        for pair_index, (left, right) in enumerate(pairs):
            left_positions = positions_by_species[left]
            right_positions = positions_by_species[right]
            same_species = left == right
            if not len(left_positions) or not len(right_positions):
                continue
            neighbor_list = freud.locality.AABBQuery(box, right_positions).query(
                left_positions,
                {
                    "mode": "ball",
                    "r_min": r_min,
                    "r_max": r_max,
                    "exclude_ii": same_species,
                },
            ).toNeighborList()
            if len(neighbor_list):
                counts[pair_index] += np.histogram(
                    np.asarray(neighbor_list.distances), bins=bin_edges
                )[0]
            possible_neighbors = len(right_positions) - int(same_species)
            if possible_neighbors > 0:
                ideal_pair_scales[pair_index] += (
                    len(left_positions) * possible_neighbors / float(box.volume)
                )
    return counts, ideal_pair_scales


def _initialize_rdf_worker(trajectory: Trajectory, cpu_ids: Sequence[int]) -> None:
    global _WORKER_TRAJECTORY
    apply_worker_limits(cpu_ids)
    _WORKER_TRAJECTORY = trajectory


def _rdf_process_worker(
    frame_indices: tuple[int, ...],
    pairs: tuple[Pair, ...],
    bin_edges: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64], int, int, int]:
    if _WORKER_TRAJECTORY is None:
        raise RuntimeError("RDF worker was not initialized with a trajectory")
    counts, scales = _accumulate_rdf_chunk(
        _WORKER_TRAJECTORY, frame_indices, pairs, bin_edges
    )
    affinity_count = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    return counts, scales, len(frame_indices), os.getpid(), affinity_count


def _accumulate_spectral_chunk(
    trajectory: Trajectory,
    frame_indices: Sequence[int],
    pairs: Sequence[Pair],
    pair_modes: Sequence[int],
    r_min: float,
    r_max: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Stream pair distances into cosine-basis coefficient sums."""

    interval = r_max - r_min
    max_modes = max(pair_modes)
    coefficient_sums = np.zeros((len(pairs), max_modes + 1), dtype=np.float64)
    ideal_pair_scales = np.zeros(len(pairs), dtype=np.float64)
    required_species = tuple(sorted({name for pair in pairs for name in pair}))
    constant_normalization = 1.0 / np.sqrt(interval)
    cosine_normalization = np.sqrt(2.0 / interval)

    for frame in trajectory.iter_frames(frame_indices):
        safe_max = safe_periodic_radius(frame.dimensions)
        if r_max > safe_max:
            raise ValueError(
                f"r_max={r_max:g} exceeds the safe periodic cutoff "
                f"{safe_max:g} at source frame {frame.source_index}"
            )
        box = _freud_box(frame.dimensions)
        positions_by_species = {
            name: np.asarray(box.wrap(frame.positions_of(name)), dtype=np.float32)
            for name in required_species
        }
        for pair_index, ((left, right), mode_count) in enumerate(
            zip(pairs, pair_modes, strict=True)
        ):
            left_positions = positions_by_species[left]
            right_positions = positions_by_species[right]
            same_species = left == right
            if not len(left_positions) or not len(right_positions):
                continue
            neighbor_list = freud.locality.AABBQuery(box, right_positions).query(
                left_positions,
                {
                    "mode": "ball",
                    "r_min": r_min,
                    "r_max": r_max,
                    "exclude_ii": same_species,
                },
            ).toNeighborList()
            if len(neighbor_list):
                distances = np.asarray(neighbor_list.distances, dtype=np.float64)
                if np.any(distances <= 0.0):
                    raise ValueError(
                        "spectral RDFs are undefined for zero-distance atom pairs"
                    )
                weights = 1.0 / (4.0 * np.pi * distances**2)
                coefficient_sums[pair_index, 0] += (
                    constant_normalization * weights.sum()
                )
                theta = np.pi * (distances - r_min) / interval
                cosine_theta = np.cos(theta)
                cosine_previous = np.ones_like(theta)
                cosine_current = cosine_theta
                coefficient_sums[pair_index, 1] += cosine_normalization * np.dot(
                    weights, cosine_current
                )
                for mode in range(2, mode_count + 1):
                    cosine_next = (
                        2.0 * cosine_theta * cosine_current - cosine_previous
                    )
                    coefficient_sums[pair_index, mode] += (
                        cosine_normalization * np.dot(weights, cosine_next)
                    )
                    cosine_previous, cosine_current = cosine_current, cosine_next
            possible_neighbors = len(right_positions) - int(same_species)
            if possible_neighbors > 0:
                ideal_pair_scales[pair_index] += (
                    len(left_positions) * possible_neighbors / float(box.volume)
                )
    return coefficient_sums, ideal_pair_scales


def _spectral_process_worker(
    frame_indices: tuple[int, ...],
    pairs: tuple[Pair, ...],
    pair_modes: tuple[int, ...],
    r_min: float,
    r_max: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int, int, int]:
    if _WORKER_TRAJECTORY is None:
        raise RuntimeError("RDF worker was not initialized with a trajectory")
    coefficients, scales = _accumulate_spectral_chunk(
        _WORKER_TRAJECTORY,
        frame_indices,
        pairs,
        pair_modes,
        r_min,
        r_max,
    )
    affinity_count = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    return coefficients, scales, len(frame_indices), os.getpid(), affinity_count


def _progress(futures: Sequence, *, enabled: bool, description: str):
    if not enabled:
        return as_completed(futures)
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return as_completed(futures)
    return tqdm(as_completed(futures), total=len(futures), desc=description)


def _execute_frame_chunks(
    trajectory: Trajectory,
    frame_indices: tuple[int, ...],
    plan,
    total_values: NDArray,
    total_scales: NDArray[np.float64],
    serial_accumulator: Callable,
    process_worker: Callable,
    accumulator_args: tuple,
    *,
    show_progress: bool,
    description: str,
) -> tuple[NDArray, NDArray[np.float64], set[int], set[int]]:
    """Run any streaming RDF accumulator under the shared core contract."""

    worker_pids: set[int] = set()
    worker_affinity_counts: set[int] = set()
    if not plan.parallel:
        with restricted_cpu_affinity(plan.cpu_ids), freud_thread_limit(
            plan.threads_per_worker
        ):
            total_values, total_scales = serial_accumulator(
                trajectory, frame_indices, *accumulator_args
            )
            worker_pids.add(os.getpid())
            worker_affinity_counts.add(len(plan.cpu_ids))
        return total_values, total_scales, worker_pids, worker_affinity_counts

    trajectory.prepare_for_multiprocessing()
    chunks = contiguous_chunks(frame_indices, plan.worker_count)
    context = multiprocessing.get_context("spawn")
    with restricted_cpu_affinity(plan.cpu_ids):
        with ProcessPoolExecutor(
            max_workers=plan.worker_count,
            mp_context=context,
            initializer=_initialize_rdf_worker,
            initargs=(trajectory, plan.cpu_ids),
        ) as executor:
            futures = [
                executor.submit(process_worker, chunk, *accumulator_args)
                for chunk in chunks
            ]
            processed_frames = 0
            for future in _progress(
                futures,
                enabled=show_progress,
                description=description,
            ):
                partial_values, partial_scales, chunk_frames, pid, affinity_count = (
                    future.result()
                )
                total_values += partial_values
                total_scales += partial_scales
                processed_frames += chunk_frames
                worker_pids.add(pid)
                worker_affinity_counts.add(affinity_count)
    if processed_frames != len(frame_indices):
        raise RuntimeError(
            f"processed {processed_frames} frames, expected {len(frame_indices)}"
        )
    return total_values, total_scales, worker_pids, worker_affinity_counts


def _execution_metadata(
    plan, worker_pids: set[int], affinity_counts: set[int], frame_count: int
) -> dict:
    return {
        "backend": plan.backend,
        "ncore": plan.ncore,
        "worker_count": plan.worker_count,
        "threads_per_worker": plan.threads_per_worker,
        "cpu_ids": plan.cpu_ids,
        "worker_pids": tuple(sorted(worker_pids)),
        "worker_affinity_counts": tuple(sorted(affinity_counts)),
        "frame_count": frame_count,
        "trajectory_backend": "MDAnalysis.Universe",
        "universe_transfer": "worker initializer",
        "coordinate_precache": False,
    }


def _system_metadata(
    trajectory: Trajectory,
    pairs: Sequence[Pair],
    ideal_pair_scales: NDArray[np.float64],
    frame_count: int,
) -> dict:
    """Describe the fixed composition and mean instantaneous number density."""

    atom_counts = trajectory.atom_counts
    total_atoms = sum(atom_counts.values())
    mean_inverse_volume = None
    for (left, right), pair_scale in zip(
        pairs, ideal_pair_scales, strict=True
    ):
        possible_pairs = atom_counts[left] * (
            atom_counts[right] - int(left == right)
        )
        if possible_pairs > 0 and pair_scale > 0:
            mean_inverse_volume = pair_scale / (frame_count * possible_pairs)
            break
    if mean_inverse_volume is None:
        raise RuntimeError("unable to determine number density from pair normalization")
    return {
        "atom_counts": atom_counts,
        "atomic_fractions": {
            species: count / total_atoms for species, count in atom_counts.items()
        },
        "number_density": float(total_atoms * mean_inverse_volume),
        "number_density_units": "atoms/angstrom^3",
        "pairs": tuple(pairs),
    }


def compute_rdfs(
    trajectory: Trajectory,
    pairs: Iterable[Pair] | None = None,
    *,
    bins: int | None = None,
    step: float | None = None,
    r_range: tuple[float, float | None] = (0.0, None),
    frames: slice | Sequence[int] | None = None,
    ncore: int | None = 1,
    backend: Backend = "auto",
    show_progress: bool = False,
) -> pd.DataFrame:
    """Compute partial RDFs from an MDAnalysis-backed trajectory.

    Provide either ``bins`` for a fixed bin count or ``step`` for an exact
    radial bin width. When neither is supplied, 800 bins are used. If a range
    is not divisible by ``step``, its final partial bin is omitted.

    Spawned workers receive the indexed Universe once in their initializer and
    then receive only contiguous frame-index chunks. Coordinates are not cached
    in the parent or transferred between processes.
    """

    if not isinstance(trajectory, Trajectory):
        raise TypeError("trajectory must be returned by load_trajectory")
    if len(r_range) != 2:
        raise ValueError("r_range must contain (r_min, r_max)")
    frame_indices = trajectory.resolve_frame_indices(frames)
    if not frame_indices:
        raise ValueError("frames must select at least one frame")

    normalized_pairs = _normalize_pairs(trajectory, pairs)
    r_min, r_max = _resolve_radius_range(
        trajectory, frame_indices, r_range[0], r_range[1]
    )
    bin_edges = _resolve_bin_edges(r_min, r_max, bins, step)
    bin_count = len(bin_edges) - 1
    shell_volumes = (4.0 * np.pi / 3.0) * (
        bin_edges[1:] ** 3 - bin_edges[:-1] ** 3
    )
    plan = plan_execution(len(frame_indices), ncore=ncore, backend=backend)
    counts = np.zeros((len(normalized_pairs), bin_count), dtype=np.int64)
    ideal_pair_scales = np.zeros(len(normalized_pairs), dtype=np.float64)
    counts, ideal_pair_scales, worker_pids, worker_affinity_counts = (
        _execute_frame_chunks(
            trajectory,
            frame_indices,
            plan,
            counts,
            ideal_pair_scales,
            _accumulate_rdf_chunk,
            _rdf_process_worker,
            (normalized_pairs, bin_edges),
            show_progress=show_progress,
            description=f"RDF frame chunks ({plan.worker_count} workers)",
        )
    )

    rdf_values = np.empty_like(counts, dtype=np.float64)
    for pair_index, pair_scale in enumerate(ideal_pair_scales):
        if pair_scale <= 0:
            pair_name = "-".join(normalized_pairs[pair_index])
            raise RuntimeError(f"no valid pair normalization for {pair_name}")
        rdf_values[pair_index] = counts[pair_index] / (pair_scale * shell_volumes)
    result = pd.DataFrame({"r": 0.5 * (bin_edges[:-1] + bin_edges[1:])})
    for pair_index, (left, right) in enumerate(normalized_pairs):
        result[f"{left}-{right}"] = rdf_values[pair_index]
    result.attrs["method"] = "histogram"
    result.attrs["execution"] = _execution_metadata(
        plan, worker_pids, worker_affinity_counts, len(frame_indices)
    )
    result.attrs["system"] = _system_metadata(
        trajectory, normalized_pairs, ideal_pair_scales, len(frame_indices)
    )
    result.attrs["binning"] = {
        "bins": bin_count,
        "step": float(bin_edges[1] - bin_edges[0]),
        "r_min": float(bin_edges[0]),
        "r_max": float(bin_edges[-1]),
    }
    return result


def compute_spectral_rdfs(
    trajectory: Trajectory,
    pairs: Iterable[Pair] | None = None,
    *,
    modes: ModeSpec = 60,
    step: float = 0.025,
    r_range: tuple[float, float | None] = (0.0, None),
    frames: slice | Sequence[int] | None = None,
    ncore: int | None = 1,
    backend: Backend = "auto",
    show_progress: bool = False,
    auto_pilot_frames: int | None = None,
    auto_min_modes: int = 10,
    auto_max_modes: int = 120,
) -> pd.DataFrame:
    """Compute smooth partial RDFs using a truncated cosine expansion.

    Modes may be one positive integer, a complete pair-to-integer mapping, or
    "auto". Automatic mode selection computes a bounded pilot spectrum, fits
    the decay-to-noise-floor elbow independently for each pair, reuses those
    pilot coefficient sums, and processes remaining frames only through the
    selected pair cutoffs. Step controls only the returned sampling grid.

    Distances are accumulated into coefficients one frame at a time and are
    never cached or transferred between processes.
    """

    if not isinstance(trajectory, Trajectory):
        raise TypeError("trajectory must be returned by load_trajectory")
    if len(r_range) != 2:
        raise ValueError("r_range must contain (r_min, r_max)")
    frame_indices = trajectory.resolve_frame_indices(frames)
    if not frame_indices:
        raise ValueError("frames must select at least one frame")

    normalized_pairs = _normalize_pairs(trajectory, pairs)
    if isinstance(modes, str) and modes != "auto":
        raise ValueError("modes string must be auto")
    auto_requested = modes == "auto"
    r_min, r_max = _resolve_radius_range(
        trajectory, frame_indices, r_range[0], r_range[1]
    )
    radii = _resolve_evaluation_grid(r_min, r_max, step)
    overall_plan = plan_execution(len(frame_indices), ncore=ncore, backend=backend)
    auto_metadata = None

    if auto_requested:
        pilot_count, minimum, maximum = _resolve_auto_mode_parameters(
            len(frame_indices),
            trajectory.n_atoms,
            auto_pilot_frames,
            auto_min_modes,
            auto_max_modes,
        )
        pilot_indices = frame_indices[:pilot_count]
        pilot_modes = (maximum,) * len(normalized_pairs)
        pilot_plan = plan_execution(
            len(pilot_indices), ncore=ncore, backend=backend
        )
        pilot_sums = np.zeros(
            (len(normalized_pairs), maximum + 1), dtype=np.float64
        )
        pilot_scales = np.zeros(len(normalized_pairs), dtype=np.float64)
        pilot_sums, pilot_scales, worker_pids, affinity_counts = (
            _execute_frame_chunks(
                trajectory,
                pilot_indices,
                pilot_plan,
                pilot_sums,
                pilot_scales,
                _accumulate_spectral_chunk,
                _spectral_process_worker,
                (normalized_pairs, pilot_modes, r_min, r_max),
                show_progress=show_progress,
                description=(
                    f"Spectral RDF auto pilot ({pilot_plan.worker_count} workers)"
                ),
            )
        )

        diagnostics: dict[str, dict[str, float | int | bool]] = {}
        resolved_modes: list[int] = []
        for pair_index, pair in enumerate(normalized_pairs):
            pair_name = "-".join(pair)
            pair_scale = pilot_scales[pair_index]
            if pair_scale <= 0:
                raise RuntimeError(f"no valid pair normalization for {pair_name}")
            pilot_coefficients = pilot_sums[pair_index] / pair_scale
            diagnostic = _fit_mode_elbow(
                pilot_coefficients, minimum, maximum
            )
            diagnostics[pair_name] = diagnostic
            resolved_modes.append(int(diagnostic["selected_mode"]))
        pair_modes = tuple(resolved_modes)
        retained_width = max(pair_modes) + 1
        coefficient_sums = pilot_sums[:, :retained_width].copy()
        ideal_pair_scales = pilot_scales.copy()
        phases = [
            {
                "name": "auto pilot",
                "frame_count": pilot_count,
                "backend": pilot_plan.backend,
                "worker_count": pilot_plan.worker_count,
                "threads_per_worker": pilot_plan.threads_per_worker,
                "modes": maximum,
            }
        ]

        remaining_indices = frame_indices[pilot_count:]
        if remaining_indices:
            remaining_plan = plan_execution(
                len(remaining_indices), ncore=ncore, backend=backend
            )
            remaining_sums = np.zeros_like(coefficient_sums)
            remaining_scales = np.zeros_like(ideal_pair_scales)
            remaining_sums, remaining_scales, remaining_pids, remaining_affinity = (
                _execute_frame_chunks(
                    trajectory,
                    remaining_indices,
                    remaining_plan,
                    remaining_sums,
                    remaining_scales,
                    _accumulate_spectral_chunk,
                    _spectral_process_worker,
                    (normalized_pairs, pair_modes, r_min, r_max),
                    show_progress=show_progress,
                    description=(
                        "Spectral RDF selected modes "
                        f"({remaining_plan.worker_count} workers)"
                    ),
                )
            )
            coefficient_sums += remaining_sums
            ideal_pair_scales += remaining_scales
            worker_pids.update(remaining_pids)
            affinity_counts.update(remaining_affinity)
            phases.append(
                {
                    "name": "selected modes",
                    "frame_count": len(remaining_indices),
                    "backend": remaining_plan.backend,
                    "worker_count": remaining_plan.worker_count,
                    "threads_per_worker": remaining_plan.threads_per_worker,
                    "modes": {
                        "-".join(pair): mode
                        for pair, mode in zip(
                            normalized_pairs, pair_modes, strict=True
                        )
                    },
                }
            )
        auto_metadata = {
            "selection_method": "piecewise fit to log(abs(a_j))",
            "pilot_frame_count": pilot_count,
            "pilot_frame_range": (pilot_indices[0], pilot_indices[-1]),
            "pilot_mode_count": maximum,
            "minimum_mode": minimum,
            "target_atom_frames": 100_000,
            "pilot_reused": True,
            "diagnostics": diagnostics,
        }
    else:
        pair_modes = _resolve_pair_modes(normalized_pairs, modes)
        coefficient_sums = np.zeros(
            (len(normalized_pairs), max(pair_modes) + 1), dtype=np.float64
        )
        ideal_pair_scales = np.zeros(len(normalized_pairs), dtype=np.float64)
        coefficient_sums, ideal_pair_scales, worker_pids, affinity_counts = (
            _execute_frame_chunks(
                trajectory,
                frame_indices,
                overall_plan,
                coefficient_sums,
                ideal_pair_scales,
                _accumulate_spectral_chunk,
                _spectral_process_worker,
                (normalized_pairs, pair_modes, r_min, r_max),
                show_progress=show_progress,
                description=(
                    f"Spectral RDF frame chunks ({overall_plan.worker_count} workers)"
                ),
            )
        )
        phases = None

    interval = r_max - r_min
    constant_normalization = 1.0 / np.sqrt(interval)
    cosine_normalization = np.sqrt(2.0 / interval)
    shifted_radii = radii - r_min
    result = pd.DataFrame({"r": radii})
    coefficients_by_pair: dict[str, tuple[float, ...]] = {}
    selected_modes: dict[str, int] = {}
    normalization_scales: dict[str, float] = {}
    for pair_index, ((left, right), mode_count) in enumerate(
        zip(normalized_pairs, pair_modes, strict=True)
    ):
        pair_name = f"{left}-{right}"
        pair_scale = ideal_pair_scales[pair_index]
        if pair_scale <= 0:
            raise RuntimeError(f"no valid pair normalization for {pair_name}")
        coefficients = coefficient_sums[pair_index, : mode_count + 1] / pair_scale
        values = np.full_like(radii, coefficients[0] * constant_normalization)
        mode_numbers = np.arange(1, mode_count + 1, dtype=np.float64)
        values += cosine_normalization * (
            np.cos(np.pi * np.outer(shifted_radii, mode_numbers) / interval)
            @ coefficients[1:]
        )
        result[pair_name] = values
        coefficients_by_pair[pair_name] = tuple(float(value) for value in coefficients)
        selected_modes[pair_name] = mode_count
        normalization_scales[pair_name] = float(pair_scale)

    result.attrs["method"] = "spectral"
    result.attrs["execution"] = _execution_metadata(
        overall_plan, worker_pids, affinity_counts, len(frame_indices)
    )
    if phases is not None:
        result.attrs["execution"]["phases"] = tuple(phases)
    result.attrs["system"] = _system_metadata(
        trajectory, normalized_pairs, ideal_pair_scales, len(frame_indices)
    )
    result.attrs["spectral"] = {
        "basis": "orthonormal cosine",
        "citation_doi": "10.1063/1.4977516",
        "selected_modes": selected_modes,
        "coefficients": coefficients_by_pair,
        "normalization_scales": normalization_scales,
        "step": float(radii[1] - radii[0]),
        "r_min": float(r_min),
        "r_max": float(r_max),
        "evaluation_r_max": float(radii[-1]),
        "auto": auto_metadata,
    }
    return result


__all__ = ["ModeSpec", "Pair", "compute_rdfs", "compute_spectral_rdfs"]
