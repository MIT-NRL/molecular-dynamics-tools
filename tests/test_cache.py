"""Persistent analysis-cache behavior and serialization tests."""

from __future__ import annotations

import tempfile
import time
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from molecular_dynamics_tools import load_trajectory
from molecular_dynamics_tools.angles import AngleDefinition
from molecular_dynamics_tools.cache import (
    AnalysisCache,
    CacheCorruptionError,
    CacheMissError,
)
from molecular_dynamics_tools.clustering import SharedNeighborClusterResult

CALLS = 0


def _result_calculation(
    scale: float = 1.0,
    *,
    ncore: int = 1,
    backend: str = "serial",
    show_progress: bool = False,
) -> SharedNeighborClusterResult:
    global CALLS
    CALLS += 1
    base = pd.DataFrame(
        {
            "cluster_size": pd.Series([1, 2], dtype="int64"),
            "probability": np.asarray([0.25, 0.75], dtype=np.float64) * scale,
            "label": pd.Series(["small", "large"], dtype="string"),
        }
    )
    base.attrs = {
        "definition": AngleDefinition("A", "B", "A", 2.5, 2.5),
        "settings": {"pairs": (("A", "B"),), "empty": None},
    }
    return SharedNeighborClusterResult(
        cluster_distribution=base,
        metadata={
            "scale": scale,
            "execution": {
                "ncore": ncore,
                "backend": backend,
                "show_progress": show_progress,
            },
        },
        sharing_distribution=base.copy(),
        frame_summary=pd.DataFrame({"frame": [0], "components": [2]}),
        percolation_cluster_distribution=base.iloc[:, :2].copy(),
        percolation_summary=pd.DataFrame({"percolates": [False], "directions": ["none"]}),
    )


def _trajectory_summary(
    trajectory,
    *,
    ncore: int = 1,
    backend: str = "serial",
    show_progress: bool = False,
) -> pd.DataFrame:
    del ncore, backend, show_progress
    return pd.DataFrame(
        {
            "frames": [len(trajectory)],
            "atoms": [trajectory.n_atoms],
            "box_x": [trajectory.frame(0).dimensions[0]],
        }
    )


def _unsupported_result() -> object:
    return object()


def _slow_result() -> pd.DataFrame:
    global CALLS
    CALLS += 1
    time.sleep(0.2)
    return pd.DataFrame({"value": [42]})


def _transform_result(value: float, *, backend: str = "direct") -> pd.DataFrame:
    return pd.DataFrame({"value": [value], "backend": [backend]})


def _write_xyz(path: Path) -> None:
    path.write_text(
        "2\n"
        'Lattice="10 0 0 0 10 0 0 0 10" Origin="-5 -5 -5" '
        "Properties=species:S:1:pos:R:3\n"
        "A 0 0 0\n"
        "B 1 0 0\n"
        "2\n"
        'Lattice="11 0 0 0 11 0 0 0 11" Origin="-4 -4 -4" '
        "Properties=species:S:1:pos:R:3\n"
        "A 0.1 0 0\n"
        "B 1.1 0 0\n",
        encoding="utf-8",
    )


class AnalysisCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        global CALLS
        CALLS = 0
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_hit_round_trips_nested_result_and_dataframe_metadata(self) -> None:
        cache = AnalysisCache(self.root / "cache", storage="json")
        produced = cache.get_or_compute(_result_calculation, 2.0, ncore=2)
        restored = cache.get_or_compute(_result_calculation, 2.0, ncore=8)

        self.assertEqual(CALLS, 1)
        self.assertTrue(cache.last_info.hit)
        self.assertEqual(restored.metadata["execution"]["ncore"], 2)
        self.assertEqual(restored.cluster_distribution.attrs, produced.cluster_distribution.attrs)
        assert_frame_equal(restored.cluster_distribution, produced.cluster_distribution)
        assert_frame_equal(restored.frame_summary, produced.frame_summary)
        self.assertIsInstance(restored, SharedNeighborClusterResult)

    def test_keys_exclude_execution_settings_but_retain_algorithm_backend(self) -> None:
        cache = AnalysisCache(self.root / "cache")
        left = cache.cache_key(
            _result_calculation,
            ncore=1,
            backend="serial",
            show_progress=False,
        )
        same = cache.cache_key(
            _result_calculation,
            ncore=24,
            backend="serial",
            show_progress=True,
        )
        other_backend = cache.cache_key(
            _result_calculation,
            ncore=1,
            backend="multiprocessing",
            show_progress=False,
        )
        self.assertEqual(left, same)
        self.assertEqual(left, other_backend)
        direct = cache.cache_key(_transform_result, 1.0, backend="direct")
        zoomfft = cache.cache_key(_transform_result, 1.0, backend="zoomfft")
        self.assertNotEqual(direct, zoomfft)

    def test_cache_modes(self) -> None:
        cache = AnalysisCache(self.root / "cache", storage="json")
        with self.assertRaises(CacheMissError):
            cache.get_or_compute(_result_calculation, cache_mode="read_only")
        cache.get_or_compute(_result_calculation)
        self.assertEqual(CALLS, 1)
        cache.get_or_compute(_result_calculation, cache_mode="refresh")
        self.assertEqual(CALLS, 2)
        cache.get_or_compute(_result_calculation, cache_mode="off")
        self.assertEqual(CALLS, 3)
        self.assertIsNone(cache.last_info.key)

    def test_corrupt_entry_raises_read_only_and_is_replaced_in_use_mode(self) -> None:
        cache = AnalysisCache(self.root / "cache", storage="json")
        cache.get_or_compute(_result_calculation)
        path = cache.last_info.path
        path.write_bytes(b"not a zip archive")

        with self.assertRaises(CacheCorruptionError):
            cache.get_or_compute(_result_calculation, cache_mode="read_only")
        restored = cache.get_or_compute(_result_calculation)
        self.assertEqual(CALLS, 2)
        self.assertFalse(cache.last_info.hit)
        self.assertEqual(restored.metadata["scale"], 1.0)

    def test_checksum_failure_is_detected(self) -> None:
        cache = AnalysisCache(self.root / "cache", storage="json")
        cache.get_or_compute(_result_calculation)
        path = cache.last_info.path
        replacement = path.with_suffix(".replacement")
        with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(replacement, "w") as target:
            for name in source.namelist():
                contents = source.read(name)
                if name.startswith("result/"):
                    contents += b"corrupt"
                target.writestr(name, contents)
        replacement.replace(path)
        with self.assertRaises(CacheCorruptionError):
            cache.get_or_compute(_result_calculation, cache_mode="read_only")

    def test_failed_serialization_leaves_no_entry_or_lock(self) -> None:
        cache = AnalysisCache(self.root / "cache")
        with self.assertRaises(TypeError):
            cache.get_or_compute(_unsupported_result)
        self.assertEqual(list((self.root / "cache").rglob("*.mdtc")), [])
        self.assertEqual(list((self.root / "cache").rglob("*.lock")), [])

    def test_concurrent_callers_share_one_completed_entry(self) -> None:
        cache = AnalysisCache(self.root / "cache", storage="json")
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(lambda _index: cache.get_or_compute(_slow_result), range(2))
            )
        self.assertEqual(CALLS, 1)
        assert_frame_equal(results[0], results[1])
        self.assertEqual(results[0].loc[0, "value"], 42)

    def test_trajectory_source_frames_box_and_file_changes_invalidate_key(self) -> None:
        path = self.root / "trajectory.xyz"
        _write_xyz(path)
        cache = AnalysisCache(self.root / "cache")
        full = load_trajectory(path)
        first = load_trajectory(path, frames=slice(0, 1))
        overridden = load_trajectory(path, frames=slice(0, 1), box=20.0)
        shifted = load_trajectory(path, shift_by_origin=True)
        try:
            keys = {
                cache.cache_key(_trajectory_summary, full),
                cache.cache_key(_trajectory_summary, first),
                cache.cache_key(_trajectory_summary, overridden),
                cache.cache_key(_trajectory_summary, shifted),
            }
            self.assertEqual(len(keys), 4)
            before = cache.cache_key(_trajectory_summary, full)
        finally:
            for trajectory in (full, first, overridden, shifted):
                trajectory.close()
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        changed = load_trajectory(path)
        try:
            self.assertNotEqual(before, cache.cache_key(_trajectory_summary, changed))
        finally:
            changed.close()

    def test_sha256_fingerprint_is_reusable_after_source_move(self) -> None:
        first_path = self.root / "one.xyz"
        second_path = self.root / "two.xyz"
        _write_xyz(first_path)
        second_path.write_bytes(first_path.read_bytes())
        cache = AnalysisCache(self.root / "cache", fingerprint="sha256")
        first = load_trajectory(first_path)
        second = load_trajectory(second_path)
        try:
            self.assertEqual(
                cache.cache_key(_trajectory_summary, first),
                cache.cache_key(_trajectory_summary, second),
            )
        finally:
            first.close()
            second.close()

    def test_in_memory_trajectory_requires_explicit_source_id(self) -> None:
        universe = mda.Universe.empty(2)
        universe.add_TopologyAttr("names", ["A", "B"])
        universe.load_new(
            np.asarray([[[0, 0, 0], [1, 0, 0]]], dtype=np.float32),
            dimensions=np.asarray([[10, 10, 10, 90, 90, 90]], dtype=np.float32),
        )
        trajectory = load_trajectory(universe, atom_attribute="names")
        cache = AnalysisCache(self.root / "cache")
        with self.assertRaisesRegex(ValueError, "cache_source_id"):
            cache.cache_key(_trajectory_summary, trajectory)
        key = cache.cache_key(_trajectory_summary, trajectory, cache_source_id="synthetic-v1")
        self.assertEqual(len(key), 64)


if __name__ == "__main__":
    unittest.main()
