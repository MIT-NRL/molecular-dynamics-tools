"""Tests for MDAnalysis-backed trajectory loading."""

from pathlib import Path
import tempfile
import unittest

import MDAnalysis as mda
import numpy as np

from molecular_dynamics_tools import load_trajectory


def write_synthetic_xyz(path: Path, *, frame_count: int = 8) -> None:
    rng = np.random.default_rng(20260825)
    base_species = np.asarray(["A"] * 12 + ["B"] * 8)
    lines: list[str] = []
    for _frame_index in range(frame_count):
        positions = rng.uniform(-5.0, 5.0, size=(len(base_species), 3))
        order = np.arange(len(base_species))
        lines.append(str(len(base_species)))
        lines.append(
            'Lattice="10 0 0 0 10 0 0 0 10" Origin="-5 -5 -5" '
            "Properties=species:S:1:pos:R:3"
        )
        for atom_index in order:
            x, y, z = positions[atom_index]
            lines.append(f"{base_species[atom_index]} {x:.8f} {y:.8f} {z:.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TrajectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_directory.name) / "synthetic.xyz"
        write_synthetic_xyz(self.path)

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_bounded_source_slice_indexes_only_requested_frames(self) -> None:
        trajectory = load_trajectory(self.path, frames=slice(1, 7, 2))
        self.assertEqual(len(trajectory), 3)
        self.assertEqual(trajectory.n_atoms, 20)
        self.assertEqual(trajectory.species, ("A", "B"))
        self.assertEqual(trajectory.atom_counts, {"A": 12, "B": 8})
        self.assertEqual(
            tuple(frame.source_index for frame in trajectory.source.frames),
            (1, 3, 5),
        )
        self.assertFalse(hasattr(trajectory.source, "positions"))
        self.assertIsInstance(trajectory.universe, mda.Universe)
        self.assertFalse(hasattr(trajectory.universe.trajectory, "_offsets"))

    def test_frame_labels_follow_the_universe_topology(self) -> None:
        trajectory = load_trajectory(self.path, frames=slice(0, 3))
        for frame in trajectory.iter_frames():
            self.assertEqual(len(frame.positions_of("A")), 12)
            self.assertEqual(len(frame.positions_of("B")), 8)
            self.assertEqual(frame.positions.dtype, np.float32)

    def test_existing_universe_is_preserved_and_selectable(self) -> None:
        universe = mda.Universe(self.path, format="XYZ")
        trajectory = load_trajectory(universe, frames=slice(0, 2))
        self.assertIs(trajectory.universe, universe)
        self.assertEqual(len(trajectory.select_atoms("name A")), 12)
        self.assertEqual(trajectory.source.atom_attribute, "elements")

    def test_in_memory_universe_uses_reader_dimensions(self) -> None:
        universe = mda.Universe.empty(4)
        universe.add_TopologyAttr("names", ["A", "A", "B", "B"])
        coordinates = np.arange(36, dtype=np.float32).reshape(3, 4, 3) / 10
        dimensions = np.tile([20, 21, 22, 90, 90, 90], (3, 1))
        universe.load_new(coordinates, order="fac", dimensions=dimensions)
        trajectory = load_trajectory(
            universe, frames=slice(1, 3), atom_attribute="names"
        )
        self.assertEqual(len(trajectory), 2)
        self.assertEqual(trajectory.species, ("A", "B"))
        self.assertEqual(trajectory.frame(0).source_index, 1)
        np.testing.assert_allclose(trajectory.frame(0).dimensions, dimensions[1])

    def test_origin_shift_is_applied_to_every_loaded_frame(self) -> None:
        unshifted = load_trajectory(self.path, frames=slice(0, 2))
        shifted = load_trajectory(
            self.path,
            frames=slice(0, 2),
            shift_by_origin=True,
        )
        for left, right in zip(unshifted.iter_frames(), shifted.iter_frames()):
            np.testing.assert_allclose(right.positions, left.positions + 5.0)

    def test_standard_xyz_accepts_explicit_box(self) -> None:
        path = Path(self.temp_directory.name) / "standard.xyz"
        path.write_text("2\ncomment\nA 0 0 0\nB 1 0 0\n", encoding="utf-8")
        trajectory = load_trajectory(path, box=8.0)
        np.testing.assert_allclose(trajectory.dimensions, [8, 8, 8, 90, 90, 90])


if __name__ == "__main__":
    unittest.main()
