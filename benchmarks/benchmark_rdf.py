"""Benchmark serial and multiprocessing RDF execution on a bounded XYZ slice."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

import molecular_dynamics_tools as mdt


def timed_rdf(trajectory, *, backend: str, ncore: int, bins: int, r_max: float):
    started = time.perf_counter()
    result = mdt.compute_rdfs(
        trajectory,
        bins=bins,
        r_range=(0.0, r_max),
        ncore=ncore,
        backend=backend,
        show_progress=False,
    )
    return result, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--ncore", type=int, default=8)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--r-max", type=float, default=5.0)
    args = parser.parse_args()

    if args.frames < 1:
        parser.error("--frames must be positive")
    if args.ncore < 1 or args.ncore > 128:
        parser.error("--ncore must be between 1 and 128 for benchmarks")

    trajectory = mdt.load_trajectory(
        args.trajectory,
        frames=slice(0, args.frames),
    )
    serial, serial_seconds = timed_rdf(
        trajectory,
        backend="serial",
        ncore=args.ncore,
        bins=args.bins,
        r_max=args.r_max,
    )
    index_started = time.perf_counter()
    trajectory.prepare_for_multiprocessing()
    index_seconds = time.perf_counter() - index_started
    parallel, parallel_seconds = timed_rdf(
        trajectory,
        backend="multiprocessing",
        ncore=args.ncore,
        bins=args.bins,
        r_max=args.r_max,
    )
    np.testing.assert_allclose(
        parallel.to_numpy(),
        serial.to_numpy(),
        rtol=1e-12,
        atol=1e-14,
    )

    print(f"frames: {len(trajectory)}")
    print(f"atoms/frame: {trajectory.n_atoms}")
    print(f"pairs: {len(serial.columns) - 1}")
    print(f"ncore budget: {args.ncore}")
    print(f"serial: {serial_seconds:.3f} s")
    print(f"MDAnalysis random-access indexing: {index_seconds:.3f} s")
    print(f"multiprocessing: {parallel_seconds:.3f} s")
    print(f"serial/parallel ratio: {serial_seconds / parallel_seconds:.3f}")
    print(f"execution: {parallel.attrs['execution']}")


if __name__ == "__main__":
    main()
