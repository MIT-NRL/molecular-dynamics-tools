"""RDF correctness and multiprocessing tests."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from molecular_dynamics_tools import (
    compute_scattering,
    compute_rdfs,
    compute_spectral_rdfs,
    load_trajectory,
)
from molecular_dynamics_tools._execution import (
    available_cpu_ids,
    freud_thread_limit,
    plan_execution,
)
from threadpoolctl import threadpool_info


LARGE_TRAJECTORY = Path(
    "/home/sfayfar/Python/Cluster_MD_Calcs/Trajectory/"
    "FlibeCsF_510C_64x_train1_nvt_equi.xyz"
)


def write_rdf_xyz(path: Path, *, frame_count: int = 16) -> None:
    rng = np.random.default_rng(4815162342)
    species = np.asarray(["A"] * 24 + ["B"] * 16)
    lines: list[str] = []
    for _frame_index in range(frame_count):
        positions = rng.uniform(-6.0, 6.0, size=(len(species), 3))
        order = np.arange(len(species))
        lines.append(str(len(species)))
        lines.append(
            'Lattice="12 0 0 0 12 0 0 0 12" '
            "Properties=species:S:1:pos:R:3"
        )
        for atom_index in order:
            x, y, z = positions[atom_index]
            lines.append(f"{species[atom_index]} {x:.8f} {y:.8f} {z:.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class RDFTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_directory.name) / "rdf.xyz"
        write_rdf_xyz(self.path)
        self.trajectory = load_trajectory(self.path)

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_serial_and_multiprocessing_are_numerically_identical(self) -> None:
        options = {
            "pairs": [("A", "A"), ("A", "B"), ("B", "B")],
            "step": 0.2,
            "r_range": (0.0, 5.0),
            "show_progress": False,
        }
        serial = compute_rdfs(
            self.trajectory,
            ncore=4,
            backend="serial",
            **options,
        )
        parallel = compute_rdfs(
            self.trajectory,
            ncore=4,
            backend="multiprocessing",
            **options,
        )

        np.testing.assert_allclose(parallel.to_numpy(), serial.to_numpy(), rtol=1e-13)
        execution = parallel.attrs["execution"]
        self.assertEqual(execution["ncore"], 4)
        self.assertEqual(execution["worker_count"], 4)
        self.assertEqual(execution["threads_per_worker"], 1)
        self.assertFalse(execution["coordinate_precache"])
        self.assertEqual(execution["trajectory_backend"], "MDAnalysis.Universe")
        self.assertEqual(execution["universe_transfer"], "worker initializer")
        self.assertEqual(execution["worker_affinity_counts"], (4,))

    def test_step_sets_a_consistent_bin_width(self) -> None:
        result = compute_rdfs(
            self.trajectory,
            pairs=[("A", "B")],
            step=0.3,
            r_range=(0.0, 4.0),
            ncore=1,
            show_progress=False,
        )
        np.testing.assert_allclose(np.diff(result["r"]), 0.3)
        self.assertEqual(len(result), 13)
        self.assertEqual(result.attrs["binning"]["bins"], 13)
        self.assertAlmostEqual(result.attrs["binning"]["step"], 0.3)
        self.assertAlmostEqual(result.attrs["binning"]["r_max"], 3.9)
        self.assertEqual(result.attrs["system"]["atom_counts"], {"A": 24, "B": 16})
        self.assertAlmostEqual(result.attrs["system"]["number_density"], 40 / 12**3)

    def test_bins_and_step_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "either bins or step"):
            compute_rdfs(
                self.trajectory,
                pairs=[("A", "B")],
                bins=20,
                step=0.2,
                r_range=(0.0, 4.0),
                ncore=1,
                show_progress=False,
            )

    def test_all_unique_pairs_are_the_default(self) -> None:
        result = compute_rdfs(
            self.trajectory,
            bins=10,
            r_range=(0.0, 4.0),
            ncore=1,
            show_progress=False,
        )
        self.assertEqual(list(result.columns), ["r", "A-A", "A-B", "B-B"])
        self.assertTrue(np.all(np.isfinite(result.to_numpy())))

    def test_rdf_metadata_drives_scattering_without_reentered_composition(self) -> None:
        rdfs = compute_rdfs(
            self.trajectory,
            step=0.1,
            r_range=(0.0, 5.0),
            ncore=1,
            show_progress=False,
        )
        scattering = compute_scattering(
            rdfs,
            elements={"A": "Li", "B": "F"},
            probes=("neutron",),
            q_range=(0.0, 10.0),
            q_step=0.2,
        )
        self.assertEqual(scattering.composition.amounts, {"A": 24.0, "B": 16.0})
        self.assertIsNotNone(scattering.neutron)
        self.assertIsNone(scattering.xray)
        self.assertTrue(
            np.all(np.isfinite(scattering.neutron.structure_factor.to_numpy()))
        )

    def test_spectral_serial_and_multiprocessing_are_identical(self) -> None:
        options = {
            "pairs": [("A", "A"), ("A", "B"), ("B", "B")],
            "modes": {("A", "A"): 8, ("B", "A"): 10, ("B", "B"): 12},
            "step": 0.1,
            "r_range": (0.2, 5.0),
            "show_progress": False,
        }
        serial = compute_spectral_rdfs(
            self.trajectory, ncore=4, backend="serial", **options
        )
        parallel = compute_spectral_rdfs(
            self.trajectory, ncore=4, backend="multiprocessing", **options
        )

        np.testing.assert_allclose(parallel.to_numpy(), serial.to_numpy(), rtol=1e-12)
        self.assertEqual(
            parallel.attrs["spectral"]["selected_modes"],
            {"A-A": 8, "A-B": 10, "B-B": 12},
        )
        self.assertEqual(parallel.attrs["method"], "spectral")
        self.assertFalse(parallel.attrs["execution"]["coordinate_precache"])
        self.assertEqual(parallel.attrs["execution"]["worker_count"], 4)

    def test_spectral_coefficients_match_direct_minimum_image_reference(self) -> None:
        r_min, r_max = 0.2, 5.0
        mode_count = 12
        result = compute_spectral_rdfs(
            self.trajectory,
            pairs=[("A", "B")],
            modes=mode_count,
            step=0.1,
            r_range=(r_min, r_max),
            ncore=1,
            show_progress=False,
        )

        interval = r_max - r_min
        coefficient_sums = np.zeros(mode_count + 1, dtype=np.float64)
        scale = 0.0
        for frame in self.trajectory.iter_frames():
            left = np.asarray(frame.positions_of("A"), dtype=np.float64)
            right = np.asarray(frame.positions_of("B"), dtype=np.float64)
            lengths = np.asarray(frame.dimensions[:3], dtype=np.float64)
            displacements = right[None, :, :] - left[:, None, :]
            displacements -= lengths * np.rint(displacements / lengths)
            distances = np.linalg.norm(displacements, axis=2).ravel()
            distances = distances[(distances >= r_min) & (distances < r_max)]
            weights = 1.0 / (4.0 * np.pi * distances**2)
            theta = np.pi * (distances - r_min) / interval
            coefficient_sums[0] += weights.sum() / np.sqrt(interval)
            for mode in range(1, mode_count + 1):
                coefficient_sums[mode] += np.sqrt(2.0 / interval) * np.dot(
                    weights, np.cos(mode * theta)
                )
            scale += len(left) * len(right) / np.prod(lengths)
        expected = coefficient_sums / scale
        observed = np.asarray(
            result.attrs["spectral"]["coefficients"]["A-B"], dtype=np.float64
        )
        np.testing.assert_allclose(observed, expected, rtol=2e-5, atol=5e-6)

        radii = result["r"].to_numpy()
        reconstructed = np.full_like(radii, expected[0] / np.sqrt(interval))
        reconstructed += np.sqrt(2.0 / interval) * (
            np.cos(
                np.pi
                * np.outer(radii - r_min, np.arange(1, mode_count + 1))
                / interval
            )
            @ expected[1:]
        )
        np.testing.assert_allclose(result["A-B"], reconstructed, rtol=2e-5)

    def test_spectral_mode_mapping_must_cover_requested_pairs(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing B-B"):
            compute_spectral_rdfs(
                self.trajectory,
                pairs=[("A", "B"), ("B", "B")],
                modes={("B", "A"): 10},
                r_range=(0.0, 5.0),
                ncore=1,
                show_progress=False,
            )
        with self.assertRaisesRegex(TypeError, "positive integer"):
            compute_spectral_rdfs(
                self.trajectory,
                pairs=[("A", "B")],
                modes=10.5,
                r_range=(0.0, 5.0),
                ncore=1,
                show_progress=False,
            )

    def test_spectral_auto_reuses_pilot_and_matches_fixed_modes(self) -> None:
        pairs = [("A", "A"), ("A", "B"), ("B", "B")]
        options = {
            "pairs": pairs,
            "step": 0.1,
            "r_range": (0.2, 5.0),
            "show_progress": False,
        }
        automatic = compute_spectral_rdfs(
            self.trajectory,
            modes="auto",
            auto_pilot_frames=8,
            auto_min_modes=5,
            auto_max_modes=20,
            ncore=1,
            backend="serial",
            **options,
        )
        selected = automatic.attrs["spectral"]["selected_modes"]
        fixed_modes = {pair: selected["-".join(pair)] for pair in pairs}
        fixed = compute_spectral_rdfs(
            self.trajectory,
            modes=fixed_modes,
            ncore=1,
            backend="serial",
            **options,
        )

        np.testing.assert_allclose(
            automatic.to_numpy(), fixed.to_numpy(), rtol=2e-13, atol=2e-13
        )
        metadata = automatic.attrs["spectral"]["auto"]
        self.assertEqual(metadata["pilot_frame_count"], 8)
        self.assertEqual(metadata["pilot_mode_count"], 20)
        self.assertTrue(metadata["pilot_reused"])
        self.assertEqual(set(metadata["diagnostics"]), {"A-A", "A-B", "B-B"})
        self.assertEqual(len(automatic.attrs["execution"]["phases"]), 2)
        self.assertFalse(automatic.attrs["execution"]["coordinate_precache"])

    def test_spectral_auto_serial_and_multiprocessing_are_identical(self) -> None:
        options = {
            "pairs": [("A", "A"), ("A", "B"), ("B", "B")],
            "modes": "auto",
            "auto_pilot_frames": 8,
            "auto_min_modes": 5,
            "auto_max_modes": 20,
            "step": 0.1,
            "r_range": (0.2, 5.0),
            "show_progress": False,
        }
        serial = compute_spectral_rdfs(
            self.trajectory, ncore=4, backend="serial", **options
        )
        parallel = compute_spectral_rdfs(
            self.trajectory, ncore=4, backend="multiprocessing", **options
        )

        self.assertEqual(
            serial.attrs["spectral"]["selected_modes"],
            parallel.attrs["spectral"]["selected_modes"],
        )
        np.testing.assert_allclose(
            parallel.to_numpy(), serial.to_numpy(), rtol=2e-12, atol=2e-12
        )
        self.assertEqual(parallel.attrs["execution"]["ncore"], 4)
        self.assertEqual(parallel.attrs["execution"]["worker_count"], 4)

    def test_spectral_auto_controls_are_validated(self) -> None:
        options = {
            "pairs": [("A", "B")],
            "modes": "auto",
            "r_range": (0.0, 5.0),
            "ncore": 1,
            "show_progress": False,
        }
        with self.assertRaisesRegex(ValueError, "exceeds"):
            compute_spectral_rdfs(
                self.trajectory, auto_pilot_frames=17, **options
            )
        with self.assertRaisesRegex(ValueError, "at least five greater"):
            compute_spectral_rdfs(
                self.trajectory,
                auto_pilot_frames=8,
                auto_min_modes=10,
                auto_max_modes=14,
                **options,
            )
        with self.assertRaisesRegex(ValueError, "must be auto"):
            compute_spectral_rdfs(
                self.trajectory,
                pairs=[("A", "B")],
                modes="automatic",
                r_range=(0.0, 5.0),
                ncore=1,
                show_progress=False,
            )

    def test_native_thread_pool_limit_is_applied_and_restored(self) -> None:
        before = [
            entry["num_threads"]
            for entry in threadpool_info()
            if entry["user_api"] == "blas"
        ]
        if not before:
            self.skipTest("no BLAS thread pool is loaded")

        limit = min(2, len(available_cpu_ids()))
        with freud_thread_limit(limit):
            inside = [
                entry["num_threads"]
                for entry in threadpool_info()
                if entry["user_api"] == "blas"
            ]
            self.assertTrue(inside)
            self.assertTrue(all(thread_count <= limit for thread_count in inside))

        after = [
            entry["num_threads"]
            for entry in threadpool_info()
            if entry["user_api"] == "blas"
        ]
        self.assertEqual(after, before)

    def test_128_core_plan_is_half_of_this_server(self) -> None:
        available = len(available_cpu_ids())
        if available < 128:
            self.skipTest("requires at least 128 allocated logical CPUs")
        plan = plan_execution(256, ncore=128, backend="multiprocessing")
        self.assertEqual(plan.ncore, 128)
        self.assertEqual(plan.worker_count, 128)
        self.assertEqual(plan.threads_per_worker, 1)
        self.assertEqual(len(plan.cpu_ids), 128)
        if available == 256:
            self.assertEqual(plan.ncore / available, 0.5)

    @unittest.skipUnless(LARGE_TRAJECTORY.is_file(), "large local trajectory not available")
    def test_large_trajectory_four_frame_subset(self) -> None:
        trajectory = load_trajectory(LARGE_TRAJECTORY, frames=slice(0, 4))
        options = {
            "pairs": [("Be", "F")],
            "bins": 16,
            "r_range": (0.0, 5.0),
            "show_progress": False,
        }
        serial = compute_rdfs(
            trajectory,
            ncore=2,
            backend="serial",
            **options,
        )
        self.assertEqual(len(trajectory), 4)
        self.assertEqual(trajectory.n_atoms, 6080)
        self.assertIsNotNone(trajectory.universe)
        self.assertTrue(np.all(np.isfinite(serial.to_numpy())))
        self.assertFalse(hasattr(trajectory.universe.trajectory, "_offsets"))


if __name__ == "__main__":
    unittest.main()
