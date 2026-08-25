"""Smoke tests for the public package surface."""

from importlib import import_module
import unittest

import molecular_dynamics_tools as mdt


class PackageTests(unittest.TestCase):
    def test_version(self) -> None:
        self.assertEqual(mdt.__version__, "0.1.0")

    def test_public_trajectory_and_rdf_exports(self) -> None:
        self.assertTrue(callable(mdt.load_trajectory))
        self.assertTrue(callable(mdt.compute_rdfs))
        self.assertTrue(callable(mdt.compute_spectral_rdfs))
        self.assertTrue(callable(mdt.compute_scattering))
        self.assertTrue(callable(mdt.compute_partial_structure_factors))
        self.assertTrue(callable(mdt.compute_scattering_weights))
        self.assertTrue(callable(mdt.plot_structure_factor))
        self.assertTrue(callable(mdt.plot_weighted_rdf))
        self.assertTrue(callable(mdt.ScatteringComposition))

    def test_scaffold_modules_import(self) -> None:
        modules = (
            "angles",
            "coordination",
            "environments",
            "rdf",
            "scattering",
            "trajectory",
            "transforms",
            "clusters.core",
            "clusters.pairs",
            "clusters.polyhedra",
        )
        for module in modules:
            with self.subTest(module=module):
                import_module(f"molecular_dynamics_tools.{module}")


if __name__ == "__main__":
    unittest.main()
