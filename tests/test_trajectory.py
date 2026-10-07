"""Tests for MDAnalysis-backed trajectory loading."""

import gc
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import MDAnalysis as mda
import numpy as np

from molecular_dynamics_tools import load_trajectory, normalize_xyz_species_order


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
        gc.collect()
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

    def test_pos_first_extended_xyz_is_canonicalized_before_loading(self) -> None:
        path = Path(self.temp_directory.name) / "pos_first.xyz"
        path.write_text(
            "2\n"
            'Lattice="10 0 0 0 10 0 0 0 10" Properties=pos:R:3:species:S:1:id:I:1\n'
            "1 2 3 B 4\n"
            "4 5 6 A 5\n"
            "2\n"
            'Lattice="10 0 0 0 10 0 0 0 10" Properties=pos:R:3:species:S:1:id:I:1\n'
            "7 8 9 A 6\n"
            "10 11 12 B 7\n",
            encoding="utf-8",
        )
        message = io.StringIO()
        with redirect_stdout(message):
            trajectory = load_trajectory(path)
        self.assertEqual(trajectory.species, ("A", "B"))
        self.assertEqual(trajectory.atom_counts, {"B": 1, "A": 1})
        np.testing.assert_allclose(trajectory.frame(0).positions_of("B"), [[1, 2, 3]])
        np.testing.assert_allclose(trajectory.frame(1).positions_of("A"), [[7, 8, 9]])
        cache_path = trajectory.filename
        self.assertIsNotNone(cache_path)
        self.assertIn("Normalizing full XYZ trajectory", message.getvalue())
        self.assertIn(str(cache_path.resolve()), message.getvalue())
        self.assertIn(f"{cache_path.stat().st_size:,} bytes", message.getvalue())

        message = io.StringIO()
        with redirect_stdout(message):
            reused = load_trajectory(path)
        self.assertEqual(reused.filename, cache_path)
        self.assertIn("Reusing normalized XYZ cache", message.getvalue())
        self.assertIn(f"{cache_path.stat().st_size:,} bytes", message.getvalue())

    def test_failed_xyz_normalization_does_not_leave_a_cache_file(self) -> None:
        path = Path(self.temp_directory.name) / "incomplete.xyz"
        destination = Path(self.temp_directory.name) / "normalized.xyz"
        path.write_text("1\ncomment\nA 0 0 0\n1\ncomment\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "incomplete atom block"):
            normalize_xyz_species_order(path, destination)
        self.assertFalse(destination.exists())
        self.assertEqual(list(destination.parent.glob(f".{destination.name}.*.tmp")), [])

    def test_standard_xyz_accepts_explicit_box(self) -> None:
        path = Path(self.temp_directory.name) / "standard.xyz"
        path.write_text("2\ncomment\nA 0 0 0\nB 1 0 0\n", encoding="utf-8")
        trajectory = load_trajectory(path, box=8.0)
        np.testing.assert_allclose(trajectory.dimensions, [8, 8, 8, 90, 90, 90])

    def test_late_sparse_and_reordered_frames_parse_only_selected_coordinates(self):
        for indices in ((7,), (4, 5, 6), (1, 5, 7), (6, 2, 5)):
            with self.subTest(indices=indices):
                with load_trajectory(self.path, frames=slice(0, 8)) as trajectory:
                    reader = trajectory.universe.trajectory
                    self.assertFalse(hasattr(reader, "_offsets"))
                    parsed = []
                    original = reader._read_next_timestep

                    def observed(*args, **kwargs):
                        timestep = original(*args, **kwargs)
                        parsed.append(timestep.frame)
                        return timestep

                    with patch.object(reader, "_read_next_timestep", side_effect=observed):
                        frames = list(trajectory.iter_frames(indices))
                    self.assertEqual(parsed, list(indices))
                    self.assertEqual([frame.source_index for frame in frames], list(indices))
                    self.assertTrue(hasattr(reader, "_offsets"))

    def test_contiguous_prefix_streams_without_building_random_access_index(self):
        with load_trajectory(self.path, frames=slice(0, 8)) as trajectory:
            reader = trajectory.universe.trajectory
            self.assertFalse(hasattr(reader, "_offsets"))
            with patch.object(reader, "_read_frame", wraps=reader._read_frame) as seek:
                frames = list(trajectory.iter_frames(slice(0, 3)))
            self.assertEqual([frame.source_index for frame in frames], [0, 1, 2])
            seek.assert_not_called()
            self.assertFalse(hasattr(reader, "_offsets"))

    def test_indexed_selection_preserves_variable_cell_species_origin_and_source_window(self):
        path = Path(self.temp_directory.name) / "variable.xyz"
        lines = []
        rng = np.random.default_rng(4513)
        for index in range(8):
            cell = np.diag([10 + index, 11 + index, 12 + index]).astype(float)
            if index % 2:
                cell[1, 0], cell[2, 0], cell[2, 1] = 1.5, -0.7, 0.8
            origin = np.asarray([-2 + index, 3 - index, 0.5 * index])
            species = np.roll(np.asarray(["A", "A", "B", "B"]), index)
            positions = rng.random((4, 3)) @ cell + origin
            lines.extend([
                "4",
                'Lattice="' + " ".join(str(value) for value in cell.ravel())
                + '" Origin="' + " ".join(str(value) for value in origin) + '"',
            ])
            lines.extend(
                f"{name} {x:.8f} {y:.8f} {z:.8f}"
                for name, (x, y, z) in zip(species, positions)
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        for source_selection in (slice(0, 8), (1, 2, 5, 6)):
            for shift in (False, True):
                with self.subTest(source_selection=source_selection, shift=shift):
                    options = dict(
                        frames=source_selection, normalize_species_order=False,
                        shift_by_origin=shift,
                    )
                    with (
                        load_trajectory(path, **options) as trajectory,
                        load_trajectory(path, **options) as reference,
                    ):
                        self.assertIsNotNone(trajectory.source.frame_species_codes)
                        self.assertFalse(hasattr(trajectory.universe.trajectory, "_offsets"))
                        for indices in ((len(trajectory) - 1,), (1, 3), (3, 0, 2)):
                            frames = list(trajectory.iter_frames(indices))
                            for logical_index, actual in zip(indices, frames):
                                expected = reference.frame(logical_index)
                                self.assertEqual(actual.index, logical_index)
                                self.assertEqual(actual.source_index, expected.source_index)
                                np.testing.assert_array_equal(actual.positions, expected.positions)
                                np.testing.assert_array_equal(actual.species, expected.species)
                                np.testing.assert_array_equal(
                                    actual.dimensions, expected.dimensions
                                )
                                np.testing.assert_array_equal(actual.origin, expected.origin)


if __name__ == "__main__":
    unittest.main()
