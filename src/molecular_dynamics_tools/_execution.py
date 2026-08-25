"""Shared execution planning and CPU-affinity controls.

Every trajectory calculator uses the same rule: ``ncore`` is the total logical
CPU budget for the call. A serial calculation gives that budget to native scientific libraries, while a multiprocessing calculation uses at most ``ncore``
single-threaded workers. Freud and BLAS thread pools are limited together, so
nested process and native-thread parallelism is never enabled.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from typing import Any, Iterator, Literal, Sequence

from threadpoolctl import threadpool_limits


Backend = Literal["auto", "serial", "multiprocessing"]
_WORKER_THREADPOOL_LIMIT: Any | None = None


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

    @property
    def parallel(self) -> bool:
        return self.backend == "multiprocessing"


def plan_execution(
    item_count: int,
    *,
    ncore: int | None = 1,
    backend: Backend = "auto",
) -> ExecutionPlan:
    """Resolve an execution plan under one consistent total-core contract."""

    item_count = int(item_count)
    if item_count < 1:
        raise ValueError("item_count must be positive")
    if backend not in {"auto", "serial", "multiprocessing"}:
        raise ValueError("backend must be 'auto', 'serial', or 'multiprocessing'")

    resolved_ncore, cpu_ids = resolve_ncore(ncore)
    resolved_backend: Literal["serial", "multiprocessing"]
    if backend == "auto":
        resolved_backend = (
            "multiprocessing" if resolved_ncore > 1 and item_count > 1 else "serial"
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
