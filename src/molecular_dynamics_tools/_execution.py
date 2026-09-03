"""Shared execution planning and CPU-affinity controls.

Every trajectory calculator uses the same rule: ``ncore`` is the total logical
CPU budget for the call. A serial calculation gives that budget to native
scientific libraries, while a multiprocessing calculation uses at most
``ncore`` single-threaded workers. Freud and BLAS thread pools are limited
together, so nested process and native-thread parallelism is never enabled.
"""

from __future__ import annotations

import math
import multiprocessing
import os
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from threadpoolctl import threadpool_limits

Backend = Literal["auto", "serial", "multiprocessing"]
_WORKER_THREADPOOL_LIMIT: Any | None = None
_WORKER_TRAJECTORY: Any | None = None
ResultT = TypeVar("ResultT")


def available_cpu_ids() -> tuple[int, ...]:
    """Return the logical CPUs currently available to this process."""

    if hasattr(os, "sched_getaffinity"):
        return tuple(sorted(os.sched_getaffinity(0)))
    return tuple(range(os.cpu_count() or 1))


def resolve_ncore(ncore: int | None) -> tuple[int, tuple[int, ...]]:
    """Validate a requested total core budget and select its logical CPUs."""

    cpu_ids = available_cpu_ids()
    resolved = len(cpu_ids) if ncore is None else int(ncore)
    if resolved < 1:
        raise ValueError("ncore must be a positive integer or None")
    if resolved > len(cpu_ids):
        raise ValueError(
            f"ncore={resolved} exceeds the {len(cpu_ids)} logical CPUs "
            "available to this process"
        )
    return resolved, cpu_ids[:resolved]


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Resolved process/thread layout for one calculation."""

    backend: Literal["serial", "multiprocessing"]
    ncore: int
    worker_count: int
    threads_per_worker: int
    cpu_ids: tuple[int, ...]
    workload: float
    auto_multiprocessing_threshold: float

    @property
    def parallel(self) -> bool:
        return self.backend == "multiprocessing"


def plan_execution(
    item_count: int,
    *,
    ncore: int | None = 1,
    backend: Backend = "auto",
    workload: float | None = None,
    auto_multiprocessing_threshold: float = 2.0,
) -> ExecutionPlan:
    """Resolve an execution plan under one consistent total-core contract."""

    item_count = int(item_count)
    if item_count < 1:
        raise ValueError("item_count must be positive")
    if backend not in {"auto", "serial", "multiprocessing"}:
        raise ValueError("backend must be 'auto', 'serial', or 'multiprocessing'")

    resolved_ncore, cpu_ids = resolve_ncore(ncore)
    resolved_workload = float(item_count if workload is None else workload)
    threshold = float(auto_multiprocessing_threshold)
    if not math.isfinite(resolved_workload) or resolved_workload < 0:
        raise ValueError("workload must be finite and nonnegative")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("auto_multiprocessing_threshold must be finite and nonnegative")
    resolved_backend: Literal["serial", "multiprocessing"]
    if backend == "auto":
        resolved_backend = (
            "multiprocessing"
            if resolved_ncore > 1
            and item_count > 1
            and resolved_workload >= threshold
            else "serial"
        )
    else:
        resolved_backend = backend

    if resolved_backend == "serial":
        worker_count = 1
        threads_per_worker = resolved_ncore
    else:
        worker_count = min(resolved_ncore, item_count)
        threads_per_worker = 1

    return ExecutionPlan(
        backend=resolved_backend,
        ncore=resolved_ncore,
        worker_count=worker_count,
        threads_per_worker=threads_per_worker,
        cpu_ids=cpu_ids,
        workload=resolved_workload,
        auto_multiprocessing_threshold=threshold,
    )


def contiguous_chunks(
    items: Sequence[int],
    chunk_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Split ordered items into balanced, contiguous, nonempty chunks."""

    if len(items) == 0:
        return ()
    chunk_count = max(1, min(int(chunk_count), len(items)))
    base_size, remainder = divmod(len(items), chunk_count)
    chunks: list[tuple[int, ...]] = []
    start = 0
    for chunk_index in range(chunk_count):
        stop = start + base_size + (1 if chunk_index < remainder else 0)
        chunks.append(tuple(items[start:stop]))
        start = stop
    return tuple(chunks)


@contextmanager
def restricted_cpu_affinity(cpu_ids: Sequence[int]) -> Iterator[None]:
    """Temporarily restrict this process and newly created children."""

    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        yield
        return

    previous = set(os.sched_getaffinity(0))
    selected = set(int(cpu_id) for cpu_id in cpu_ids)
    if not selected.issubset(previous):
        raise ValueError("requested CPU affinity is outside the current allocation")

    os.sched_setaffinity(0, selected)
    try:
        yield
    finally:
        os.sched_setaffinity(0, previous)


def apply_worker_limits(cpu_ids: Sequence[int], *, freud_threads: int = 1) -> None:
    """Apply inherited worker limits before scientific work begins."""

    global _WORKER_THREADPOOL_LIMIT
    _WORKER_THREADPOOL_LIMIT = threadpool_limits(limits=int(freud_threads))

    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(int(cpu_id) for cpu_id in cpu_ids))

    try:
        import freud
    except ImportError:
        return
    freud.set_num_threads(int(freud_threads))


def _initialize_trajectory_worker(trajectory: Any, cpu_ids: Sequence[int]) -> None:
    """Install one read-only trajectory handle and resource limits per worker."""

    global _WORKER_TRAJECTORY
    _WORKER_TRAJECTORY = trajectory
    apply_worker_limits(cpu_ids, freud_threads=1)


def _run_trajectory_chunk(
    accumulator: Callable[..., ResultT],
    frame_indices: tuple[int, ...],
    accumulator_args: tuple[Any, ...],
) -> tuple[ResultT, int, int]:
    if _WORKER_TRAJECTORY is None:
        raise RuntimeError("trajectory worker was not initialized")
    result = accumulator(_WORKER_TRAJECTORY, frame_indices, *accumulator_args)
    affinity_count = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    return result, os.getpid(), affinity_count


def _completed_futures(futures: Sequence[Any], *, enabled: bool, description: str):
    completed = as_completed(futures)
    if not enabled:
        return completed
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return completed
    return tqdm(completed, total=len(futures), desc=description)


def execution_metadata(
    plan: ExecutionPlan,
    *,
    worker_pids: Sequence[int],
    affinity_counts: Sequence[int],
    frame_count: int,
) -> dict[str, Any]:
    """Return the common, user-visible execution record."""

    return {
        "backend": plan.backend,
        "ncore": plan.ncore,
        "worker_count": plan.worker_count,
        "threads_per_worker": plan.threads_per_worker,
        "cpu_ids": plan.cpu_ids,
        "worker_pids": tuple(sorted(set(worker_pids))),
        "worker_affinity_counts": tuple(sorted(set(affinity_counts))),
        "frame_count": int(frame_count),
        "workload": plan.workload,
        "auto_multiprocessing_threshold": plan.auto_multiprocessing_threshold,
        "trajectory_backend": "MDAnalysis.Universe",
        "universe_transfer": "worker initializer",
        "coordinate_precache": False,
    }


def execute_frame_chunks(
    trajectory: Any,
    frame_indices: Sequence[int],
    plan: ExecutionPlan,
    accumulator: Callable[..., ResultT],
    accumulator_args: tuple[Any, ...] = (),
    *,
    show_progress: bool = False,
    description: str = "Trajectory analysis",
) -> tuple[list[ResultT], dict[str, Any]]:
    """Execute a streaming frame accumulator under the shared core contract.

    ``accumulator`` must be a module-level callable accepting ``trajectory``, a
    tuple of logical frame indices, and ``accumulator_args``. Results are
    returned in frame-chunk order even though parallel chunks finish out of
    order. Coordinates are never materialized in the parent process.
    """

    selected = tuple(int(index) for index in frame_indices)
    if not selected:
        raise ValueError("frame_indices must contain at least one frame")

    if not plan.parallel:
        with restricted_cpu_affinity(plan.cpu_ids), freud_thread_limit(
            plan.threads_per_worker
        ):
            result = accumulator(trajectory, selected, *accumulator_args)
        metadata = execution_metadata(
            plan,
            worker_pids=(os.getpid(),),
            affinity_counts=(len(plan.cpu_ids),),
            frame_count=len(selected),
        )
        return [result], metadata

    trajectory.prepare_for_multiprocessing()
    chunks = contiguous_chunks(selected, plan.worker_count)
    ordered_results: list[ResultT | None] = [None] * len(chunks)
    worker_pids: set[int] = set()
    affinity_counts: set[int] = set()
    context = multiprocessing.get_context("spawn")

    with restricted_cpu_affinity(plan.cpu_ids):
        with ProcessPoolExecutor(
            max_workers=plan.worker_count,
            mp_context=context,
            initializer=_initialize_trajectory_worker,
            initargs=(trajectory, plan.cpu_ids),
        ) as executor:
            futures = {
                executor.submit(
                    _run_trajectory_chunk,
                    accumulator,
                    chunk,
                    accumulator_args,
                ): chunk_index
                for chunk_index, chunk in enumerate(chunks)
            }
            for future in _completed_futures(
                tuple(futures), enabled=show_progress, description=description
            ):
                result, pid, affinity_count = future.result()
                ordered_results[futures[future]] = result
                worker_pids.add(pid)
                affinity_counts.add(affinity_count)

    if any(result is None for result in ordered_results):
        raise RuntimeError("one or more trajectory chunks produced no result")
    metadata = execution_metadata(
        plan,
        worker_pids=tuple(worker_pids),
        affinity_counts=tuple(affinity_counts),
        frame_count=len(selected),
    )
    return [result for result in ordered_results if result is not None], metadata


@contextmanager
def freud_thread_limit(thread_count: int) -> Iterator[None]:
    """Temporarily set freud's native thread count."""

    import freud

    previous = freud.get_num_threads()
    freud.set_num_threads(int(thread_count))
    try:
        with threadpool_limits(limits=int(thread_count)):
            yield
    finally:
        freud.set_num_threads(previous)
