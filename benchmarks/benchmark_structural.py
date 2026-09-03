"""Benchmark migrated structural analyses and verify backend parity."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import molecular_dynamics_tools as mdt

Method = Callable[[str, int], pd.DataFrame | mdt.BridgingClusterResult]


def _time(function: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    result = function()
    return result, time.perf_counter() - started


def _assert_equivalent(
    candidate: pd.DataFrame | mdt.BridgingClusterResult,
    reference: pd.DataFrame | mdt.BridgingClusterResult,
) -> None:
    if isinstance(reference, mdt.BridgingClusterResult):
        assert isinstance(candidate, mdt.BridgingClusterResult)
        for attribute in (
            "sharing_distribution",
            "cluster_distribution",
            "frame_summary",
            "percolation_cluster_distribution",
            "percolation_summary",
        ):
            left = getattr(candidate, attribute)
            right = getattr(reference, attribute)
            pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-13)
        return
    assert isinstance(candidate, pd.DataFrame)
    np.testing.assert_allclose(candidate.to_numpy(), reference.to_numpy(), rtol=1e-13)


def _methods(
    trajectory: mdt.Trajectory,
    center: str,
    ligand: str,
    cutoff: float,
) -> dict[str, Method]:
    return {
        "coordination": lambda backend, ncore: mdt.compute_coordination(
            trajectory, [(center, ligand, cutoff)], backend=backend, ncore=ncore
        ),
        "rad": lambda backend, ncore: mdt.compute_rad_coordination(
            trajectory, [(center, ligand)], backend=backend, ncore=ncore
        ),
        "angles": lambda backend, ncore: mdt.compute_bond_angles(
            trajectory,
            [(ligand, center, ligand, cutoff, cutoff)],
            bins=180,
            backend=backend,
            ncore=ncore,
        ),
        "cutoff_clusters": lambda backend, ncore: mdt.compute_cutoff_clusters(
            trajectory, [(center, ligand, cutoff)], backend=backend, ncore=ncore
        ),
        "bridging_clusters": lambda backend, ncore: mdt.analyze_bridging_clusters(
            trajectory, center, ligand, cutoff, backend=backend, ncore=ncore
        ),
    }


def _parse_ncores(values: Sequence[int], parser: argparse.ArgumentParser) -> tuple[int, ...]:
    ncores = tuple(dict.fromkeys(int(value) for value in values))
    if not ncores or any(value < 2 or value > 24 for value in ncores):
        parser.error("--ncores values must be unique integers between 2 and 24")
    return ncores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--center", required=True)
    parser.add_argument("--ligand", required=True)
    parser.add_argument("--cutoff", required=True, type=float)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--ncores", type=int, nargs="+", default=(2, 4, 8, 16, 24))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("coordination", "rad", "angles", "cutoff_clusters", "bridging_clusters"),
        default=("coordination", "rad", "angles", "cutoff_clusters", "bridging_clusters"),
    )
    parser.add_argument("--format", help="Explicit MDAnalysis coordinate format")
    parser.add_argument("--atom-attribute", default="auto")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    ncores = _parse_ncores(args.ncores, parser)

    with mdt.load_trajectory(
        args.trajectory,
        frames=slice(0, args.frames),
        format=args.format,
        atom_attribute=args.atom_attribute,
    ) as trajectory:
        available = _methods(trajectory, args.center, args.ligand, args.cutoff)
        started = time.perf_counter()
        trajectory.prepare_for_multiprocessing()
        index_seconds = time.perf_counter() - started
        print(f"frames: {len(trajectory)}")
        print(f"atoms/frame: {trajectory.n_atoms}")
        print(f"MDAnalysis random-access indexing: {index_seconds:.3f} s")

        for name in args.methods:
            calculation = available[name]
            baseline, serial_seconds = _time(lambda: calculation("serial", 1))
            print(f"{name}: serial, 1 core: {serial_seconds:.3f} s")
            for ncore in ncores:
                candidate, elapsed = _time(
                    lambda ncore=ncore: calculation("multiprocessing", ncore)
                )
                _assert_equivalent(candidate, baseline)
                print(
                    f"{name}: multiprocessing, {ncore} cores: {elapsed:.3f} s "
                    f"({serial_seconds / elapsed:.3f}x)"
                )


if __name__ == "__main__":
    main()
