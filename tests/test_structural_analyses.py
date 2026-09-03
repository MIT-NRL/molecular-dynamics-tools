"""Correctness and shared-backend tests for migrated structural analyses."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from molecular_dynamics_tools import (
    analyze_bridging_clusters,
    compute_bond_angles,
    compute_coordination,
    compute_cutoff_clusters,
    compute_rad_coordination,
    load_trajectory,
    summarize_coordination,
)


def write_network_xyz(path: Path, *, frame_count: int = 8) -> None:
    species = ("C", "C", "C", "L", "L", "L")
    positions = np.asarray(
        [
            [-3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [-1.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [1.5, 0.5, 0.0],
        ]
    )
    lines: list[str] = []
    for frame in range(frame_count):
        shifted = positions + np.asarray([0.0, 0.0, frame * 0.001])
        lines.extend(
            [
                str(len(species)),
                'Lattice="20 0 0 0 20 0 0 0 20" Properties=species:S:1:pos:R:3',
            ]
        )
        lines.extend(
            f"{atom} {position[0]} {position[1]} {position[2]}"
            for atom, position in zip(species, shifted, strict=True)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_wrapping_xyz(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "4",
                'Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3',
                "C -4 0 0",
                "C 4 0 0",
                "L -5 0 0",
                "L 0 0 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


class StructuralAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary_directory.name) / "network.xyz"
        write_network_xyz(self.path)
        self.trajectory = load_trajectory(self.path)

    def tearDown(self) -> None:
        self.trajectory.universe.trajectory.close()
        self.temporary_directory.cleanup()

    def test_cutoff_coordination_values_and_summary(self) -> None:
        result = compute_coordination(
            self.trajectory,
            [("C", "L", 1.7)],
            backend="serial",
            ncore=2,
        )
        np.testing.assert_allclose(result["C-L"].to_numpy(), [0.0, 1 / 3, 1 / 3, 1 / 3])
        summary = summarize_coordination(result)
        self.assertAlmostEqual(summary.loc[0, "mean"], 2.0)
        self.assertEqual(result.attrs["execution"]["coordinate_precache"], False)

    def test_all_migrated_calculators_match_multiprocessing(self) -> None:
        calculations = (
            lambda backend: compute_coordination(
                self.trajectory,
                [("C", "L", 1.7), ("L", "C", 1.7)],
                ncore=4,
                backend=backend,
            ),
            lambda backend: compute_rad_coordination(
                self.trajectory,
                [("C", "L"), ("L", "C")],
                ncore=4,
                backend=backend,
            ),
            lambda backend: compute_bond_angles(
                self.trajectory,
                [("C", "L", "C", 1.7, 1.7)],
                bins=36,
                ncore=4,
                backend=backend,
            ),
            lambda backend: compute_cutoff_clusters(
                self.trajectory,
                [("C", "L", 1.7), ("C", "C", 3.1)],
                ncore=4,
                backend=backend,
            ),
        )
        for calculation in calculations:
            with self.subTest(calculation=calculation):
                serial = calculation("serial")
                parallel = calculation("multiprocessing")
                np.testing.assert_allclose(parallel.to_numpy(), serial.to_numpy())
                self.assertEqual(parallel.attrs["execution"]["worker_count"], 4)
                self.assertFalse(parallel.attrs["execution"]["coordinate_precache"])

    def test_cutoff_and_bridging_strategies_are_distinct_and_clear(self) -> None:
        cutoff = compute_cutoff_clusters(
            self.trajectory,
            [("C", "L", 1.7)],
            backend="serial",
        )
        self.assertAlmostEqual(cutoff.loc[cutoff["cluster_size"] == 3, "C-L"].iloc[0], 1.0)
        self.assertEqual(cutoff.attrs["method"], "cutoff")

        serial = analyze_bridging_clusters(
            self.trajectory,
            "C",
            "L",
            1.7,
            backend="serial",
            ncore=4,
        )
        parallel = analyze_bridging_clusters(
            self.trajectory,
            "C",
            "L",
            1.7,
            backend="multiprocessing",
            ncore=4,
        )
        np.testing.assert_allclose(
            parallel.cluster_distribution.to_numpy(),
            serial.cluster_distribution.to_numpy(),
        )
        np.testing.assert_allclose(
            parallel.percolation_cluster_distribution.to_numpy(),
            serial.percolation_cluster_distribution.to_numpy(),
        )
        self.assertEqual(
            serial.sharing_distribution["sharing_type"].tolist(), ["corner", "edge"]
        )
        self.assertTrue(np.all(serial.frame_summary["corner_links"] == 1))
        self.assertTrue(np.all(serial.frame_summary["edge_links"] == 1))
        self.assertTrue(np.all(serial.frame_summary["face_links"] == 0))
        self.assertTrue(np.all(serial.frame_summary["largest_cluster_size"] == 3))
        self.assertFalse(serial.frame_summary["wrap_any"].any())
        self.assertEqual(parallel.metadata["execution"]["worker_count"], 4)
        self.assertFalse(parallel.metadata["execution"]["coordinate_precache"])

    def test_bridging_network_detects_periodic_wrapping_cycles(self) -> None:
        path = Path(self.temporary_directory.name) / "wrapping.xyz"
        write_wrapping_xyz(path)
        trajectory = load_trajectory(path)
        result = analyze_bridging_clusters(
            trajectory,
            "C",
            "L",
            4.5,
            backend="serial",
        )
        self.assertTrue(result.frame_summary.loc[0, "wrap_x"])
        self.assertTrue(result.frame_summary.loc[0, "wrap_any"])
        self.assertEqual(result.frame_summary.loc[0, "percolation_strength"], 1.0)
        trajectory.close()


if __name__ == "__main__":
    unittest.main()
