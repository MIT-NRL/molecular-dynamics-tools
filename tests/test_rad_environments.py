"""RAD environment, distance, and persistence tests."""

import tempfile
import unittest
from pathlib import Path

import freud
import numpy as np
import pandas as pd

from molecular_dynamics_tools import compute_rad_environments, load_trajectory
from molecular_dynamics_tools._rad import rad_neighbors


def write_rad_xyz(path: Path, *, frame_count: int = 4) -> None:
    species = ("U", "U", "Te", "Te")
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ]
    )
    lines: list[str] = []
    for frame in range(frame_count):
        shifted = positions + np.asarray([0.0, frame * 0.01, 0.0])
        lines.extend(
            [
                str(len(species)),
                'Lattice="20 0 0 0 20 0 0 0 20" Properties=species:S:1:pos:R:3',
            ]
        )
        lines.extend(
            f"{name} {point[0]} {point[1]} {point[2]}"
            for name, point in zip(species, shifted, strict=True)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_periodic_xyz(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "3",
                'Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3',
                "U 4.8 0 0",
                "Te -4.8 0 0",
                "Cl 0 0 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


class RADEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary_directory.name) / "rad.xyz"
        write_rad_xyz(self.path)
        self.trajectory = load_trajectory(self.path)

    def tearDown(self) -> None:
        self.trajectory.close()
        self.temporary_directory.cleanup()

    def test_species_counts_distances_and_persistence(self) -> None:
        result = compute_rad_environments(
            self.trajectory,
            center_species="U",
            neighbor_species=("Te", "U"),
            frames=(0, 2),
            backend="serial",
        )
        self.assertEqual(
            result.environments["neighbor_species"].iloc[:2].tolist(), ["Te", "U"]
        )
        speciation = result.speciation(("Te", "U"))
        self.assertEqual(speciation.loc[0, "label"], "UTe1U0")
        self.assertEqual(speciation.loc[0, "Te"], 1)
        self.assertEqual(speciation.loc[0, "U"], 0)

        with self.assertRaisesRegex(ValueError, "stable atom indices"):
            result.occupancy("U", "Te")
        occupancy = result.occupancy(
            "U", "Te", assume_stable_atom_identity=True
        )
        self.assertEqual(len(occupancy), 4)
        self.assertEqual((occupancy["occupancy"] == 0.0).sum(), 2)
        self.assertEqual((occupancy["occupancy"] == 1.0).sum(), 2)

        lifetimes = result.lifetimes(
            "U", "Te", assume_stable_atom_identity=True
        )
        self.assertTrue(np.all(lifetimes["lifetime_samples"] == 2))
        self.assertTrue(np.all(lifetimes["logical_frame_span"] == 3))
        self.assertFalse(result.metadata["frame_selection_consecutive"])

    def test_periodic_distance_and_mutual_classification(self) -> None:
        path = Path(self.temporary_directory.name) / "periodic.xyz"
        write_periodic_xyz(path)
        trajectory = load_trajectory(path)
        result = compute_rad_environments(
            trajectory,
            center_species=("U", "Te"),
            neighbor_species=("Te", "U"),
            backend="serial",
        )
        u_te = result.contacts_between("U", "Te")
        self.assertEqual(len(u_te), 1)
        self.assertAlmostEqual(u_te.loc[0, "distance"], 0.4, places=5)
        self.assertTrue(u_te.loc[0, "is_mutual"])
        self.assertEqual(len(result.contacts_between("Te", "U", bond_mode="mutual")), 1)
        trajectory.close()

    def test_serial_and_multiprocessing_outputs_match(self) -> None:
        serial = compute_rad_environments(
            self.trajectory,
            center_species=("U", "Te"),
            neighbor_species=("Te", "U"),
            ncore=2,
            backend="serial",
        )
        parallel = compute_rad_environments(
            self.trajectory,
            center_species=("U", "Te"),
            neighbor_species=("Te", "U"),
            ncore=2,
            backend="multiprocessing",
        )
        pd.testing.assert_frame_equal(parallel.contacts, serial.contacts)
        pd.testing.assert_frame_equal(parallel.environments, serial.environments)
        self.assertEqual(parallel.metadata["execution"]["worker_count"], 2)
        self.assertFalse(parallel.metadata["execution"]["coordinate_precache"])

    def test_closed_kernel_matches_freud_filter_rad(self) -> None:
        rng = np.random.default_rng(4815)
        box = freud.Box.cube(10)
        positions = rng.uniform(-5, 5, size=(25, 3)).astype(np.float32)
        reference = freud.locality.FilterRAD(
            allow_incomplete_shell=True, terminate_after_blocked=True
        ).compute((box, positions))
        neighbor_list = reference.filtered_nlist
        query_indices = np.asarray(neighbor_list.query_point_indices)
        point_indices = np.asarray(neighbor_list.point_indices)
        for center in range(len(positions)):
            expected = set(point_indices[query_indices == center].tolist())
            actual = set(rad_neighbors(box, positions, center).tolist())
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
