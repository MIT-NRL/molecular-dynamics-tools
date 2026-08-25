"""Scientific and API tests for scattering weights, S(Q), and weighted g(r)."""

import unittest

import numpy as np
import pandas as pd

from molecular_dynamics_tools import (
    ScatteringComposition,
    compute_partial_structure_factors,
    compute_scattering,
    compute_scattering_weights,
    plot_structure_factor,
)
from molecular_dynamics_tools.transforms import spherical_bessel_transform


PAIRS = (("F", "F"), ("F", "Li"), ("Li", "Li"))


def synthetic_rdfs(*, ideal: bool = False) -> pd.DataFrame:
    r = np.linspace(0.0, 10.0, 501)
    if ideal:
        values = {"F-F": np.ones_like(r), "F-Li": np.ones_like(r), "Li-Li": np.ones_like(r)}
    else:
        values = {
            "F-F": 1.0 + 0.25 * np.exp(-((r - 2.1) / 0.45) ** 2),
            "F-Li": 1.0 - 0.18 * np.exp(-((r - 2.7) / 0.60) ** 2),
            "Li-Li": 1.0 + 0.12 * np.exp(-((r - 3.3) / 0.75) ** 2),
        }
    result = pd.DataFrame({"r": r, **values})
    result.attrs["system"] = {
        "atom_counts": {"F": 4, "Li": 2},
        "atomic_fractions": {"F": 2 / 3, "Li": 1 / 3},
        "number_density": 0.065,
        "number_density_units": "atoms/angstrom^3",
        "pairs": PAIRS,
    }
    return result


class TransformTests(unittest.TestCase):
    def test_zoomfft_matches_direct_trapezoidal_transform(self) -> None:
        source = np.linspace(0.01, 8.01, 401)
        targets = np.linspace(0.02, 18.02, 361)
        curves = np.column_stack(
            (np.exp(-source), np.exp(-0.25 * source**2) * np.cos(source))
        )
        direct = spherical_bessel_transform(
            source, curves, targets, backend="direct"
        )
        fast = spherical_bessel_transform(
            source, curves, targets, backend="zoomfft"
        )
        np.testing.assert_allclose(fast, direct, rtol=2e-11, atol=2e-11)

    def test_gaussian_transform_has_the_analytic_normalization(self) -> None:
        source = np.linspace(0.0, 10.0, 2001)
        targets = np.linspace(0.0, 5.0, 101)
        coefficient = 0.5
        observed = spherical_bessel_transform(
            source,
            np.exp(-coefficient * source**2),
            targets,
            backend="zoomfft",
        )
        expected = (
            np.sqrt(np.pi)
            / (4.0 * coefficient**1.5)
            * np.exp(-(targets**2) / (4.0 * coefficient))
        )
        np.testing.assert_allclose(observed, expected, rtol=2e-10, atol=2e-10)


class ScatteringCompositionTests(unittest.TestCase):
    def test_formula_isotopes_charges_and_density(self) -> None:
        composition = ScatteringComposition.from_formula(
            "Li2F",
            isotopes={"Li": {7: 0.75, 6: 0.25}},
            charges={"Li": 1, "F": -1},
        )
        self.assertEqual(composition.species, ("F", "Li"))
        self.assertAlmostEqual(composition.concentrations["Li"], 2 / 3)

        lithium_7 = ScatteringComposition({"Li": 1}, isotopes={"Li": {7: 1.0}})
        lithium_6 = ScatteringComposition({"Li": 1}, isotopes={"Li": {6: 1.0}})
        expected = (
            0.75 * lithium_7.neutron_scattering_lengths()["Li"]
            + 0.25 * lithium_6.neutron_scattering_lengths()["Li"]
        )
        self.assertAlmostEqual(
            composition.neutron_scattering_lengths()["Li"], expected
        )
        self.assertGreater(composition.number_density(2.0), 0.0)

        q = np.linspace(0.0, 10.0, 51)
        ionic = composition.xray_form_factors(q)
        neutral = ScatteringComposition.from_formula("Li2F").xray_form_factors(q)
        self.assertFalse(np.allclose(ionic["Li"], neutral["Li"]))

    def test_isotope_fractions_must_sum_to_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to one"):
            ScatteringComposition({"Li": 1}, isotopes={"Li": {6: 0.2, 7: 0.7}})


class ScatteringCalculationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rdfs = synthetic_rdfs()
        self.composition = ScatteringComposition(
            {"F": 4, "Li": 2},
            isotopes={"Li": {7: 0.99, 6: 0.01}},
            charges={"F": -1, "Li": 1},
        )
        self.options = {
            "composition": self.composition,
            "q_range": (0.0, 20.0),
            "q_step": 0.05,
            "xray_window": "lorch",
        }

    def test_faber_ziman_weights_sum_to_one(self) -> None:
        q = np.linspace(0.0, 20.0, 101)
        for probe in ("neutron", "xray"):
            with self.subTest(probe=probe):
                weights = compute_scattering_weights(
                    self.composition,
                    PAIRS,
                    q,
                    probe=probe,
                    convention="faber-ziman",
                )
                np.testing.assert_allclose(
                    weights.drop(columns="Q").sum(axis=1), 1.0, atol=2e-14
                )

    def test_ashcroft_langreth_diagonal_weights_supply_the_baseline(self) -> None:
        q = np.linspace(0.0, 20.0, 101)
        for probe in ("neutron", "xray"):
            with self.subTest(probe=probe):
                weights = compute_scattering_weights(
                    self.composition,
                    PAIRS,
                    q,
                    probe=probe,
                    convention="ashcroft-langreth",
                )
                np.testing.assert_allclose(
                    weights[["F-F", "Li-Li"]].sum(axis=1), 1.0, atol=2e-14
                )

    def test_partial_conventions_have_the_expected_algebraic_relation(self) -> None:
        common = {
            "composition": self.composition,
            "q_range": (0.0, 20.0),
            "q_step": 0.05,
        }
        faber_ziman = compute_partial_structure_factors(
            self.rdfs, convention="faber-ziman", **common
        )
        ashcroft_langreth = compute_partial_structure_factors(
            self.rdfs, convention="ashcroft-langreth", **common
        )
        concentrations = self.composition.concentrations
        for pair in PAIRS:
            label = "-".join(pair)
            baseline = 1.0 if pair[0] == pair[1] else 0.0
            expected = baseline + np.sqrt(
                concentrations[pair[0]] * concentrations[pair[1]]
            ) * (faber_ziman[label] - 1.0)
            np.testing.assert_allclose(
                ashcroft_langreth[label], expected, rtol=1e-12, atol=1e-15
            )

    def test_ideal_gas_has_unit_structure_factor_and_weighted_rdf(self) -> None:
        result = compute_scattering(
            synthetic_rdfs(ideal=True),
            composition=self.composition,
            q_range=(0.0, 20.0),
            q_step=0.05,
        )
        for probe in (result.neutron, result.xray):
            self.assertIsNotNone(probe)
            np.testing.assert_allclose(probe.structure_factor["Total"], 1.0)
            np.testing.assert_allclose(probe.weighted_rdf["Total"], 1.0)
            np.testing.assert_allclose(
                probe.weighted_rdf.drop(columns=["r", "Total"]).sum(axis=1),
                1.0,
            )

    def test_result_separates_unweighted_and_weighted_rdfs(self) -> None:
        result = compute_scattering(self.rdfs, **self.options)
        self.assertIs(result.partial_rdfs, self.rdfs)
        self.assertEqual(result.convention, "faber-ziman")
        self.assertFalse(result.metadata["multiprocessing"])
        self.assertEqual(result.neutron.weighted_rdf.columns[-1], "Total")
        self.assertEqual(result.xray.weighted_rdf.attrs["probe"], "xray")
        self.assertEqual(
            result.xray.weighted_rdf.attrs["normalization"],
            "dimensionless; Total approaches 1",
        )
        np.testing.assert_allclose(
            result.xray.weighted_rdf["Total"],
            result.xray.weighted_rdf[["F-F", "F-Li", "Li-Li"]].sum(axis=1),
        )
        np.testing.assert_allclose(
            result.xray.structure_factor["Total"],
            1.0 + result.xray.structure_factor[["F-F", "F-Li", "Li-Li"]].sum(axis=1),
        )

    def test_neutron_rdf_is_weighted_directly_without_a_window(self) -> None:
        result = compute_scattering(self.rdfs, probes=("neutron",), **self.options)
        pairs = ["F-F", "F-Li", "Li-Li"]
        weights = result.neutron.weights.loc[0, pairs].to_numpy()
        expected = (self.rdfs[pairs].to_numpy() * weights).sum(axis=1)
        np.testing.assert_allclose(result.neutron.weighted_rdf["Total"], expected)
        self.assertEqual(
            result.neutron.weighted_rdf.attrs["method"],
            "direct weighting of partial RDFs",
        )
        self.assertIsNone(result.neutron.weighted_rdf.attrs["window"])

    def test_lorch_window_is_optional_and_xray_only(self) -> None:
        lorch = compute_scattering(self.rdfs, **self.options)
        unwindowed = compute_scattering(
            self.rdfs,
            **{**self.options, "xray_window": None},
        )
        np.testing.assert_allclose(
            lorch.neutron.weighted_rdf,
            unwindowed.neutron.weighted_rdf,
        )
        self.assertFalse(
            np.allclose(lorch.xray.weighted_rdf, unwindowed.xray.weighted_rdf)
        )
        self.assertEqual(lorch.xray.weighted_rdf.attrs["window"], "lorch")
        self.assertIsNone(unwindowed.xray.weighted_rdf.attrs["window"])

    def test_structure_factor_plot_uses_unit_baseline_for_every_curve(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = compute_scattering(self.rdfs, probes=("neutron",), **self.options)
        axis = plot_structure_factor(result.neutron)
        lines = {line.get_label(): line for line in axis.lines}
        np.testing.assert_allclose(
            lines["Total"].get_ydata(), result.neutron.structure_factor["Total"]
        )
        for pair in ("F-F", "F-Li", "Li-Li"):
            np.testing.assert_allclose(
                lines[pair].get_ydata(),
                1.0 + result.neutron.structure_factor[pair],
            )
        self.assertEqual(axis.get_ylabel(), "$S(Q)$")
        plt.close(axis.figure)

    def test_ashcroft_langreth_changes_total_normalization(self) -> None:
        fz = compute_scattering(
            self.rdfs,
            probes=("neutron",),
            convention="faber-ziman",
            **self.options,
        )
        al = compute_scattering(
            self.rdfs,
            probes=("neutron",),
            convention="ashcroft-langreth",
            **self.options,
        )
        concentrations = self.composition.concentrations
        lengths = self.composition.neutron_scattering_lengths()
        mean = sum(concentrations[name] * lengths[name] for name in concentrations)
        mean_square = sum(
            concentrations[name] * lengths[name] ** 2 for name in concentrations
        )
        expected = 1.0 + mean**2 / mean_square * (
            fz.neutron.structure_factor["Total"] - 1.0
        )
        np.testing.assert_allclose(al.neutron.structure_factor["Total"], expected)

    def test_incomplete_pair_set_is_rejected(self) -> None:
        incomplete = self.rdfs.drop(columns="Li-Li")
        incomplete.attrs = {}
        with self.assertRaisesRegex(ValueError, "requires every unique pair"):
            compute_scattering(
                incomplete,
                composition=self.composition,
                number_density=0.065,
                q_range=(0.0, 20.0),
                q_step=0.05,
            )

    def test_minor_legacy_grid_jitter_is_normalized_for_fft(self) -> None:
        jittered = self.rdfs.copy()
        r = jittered["r"].to_numpy().copy()
        r[1:-1] += 2e-7 * np.sin(np.arange(1, len(r) - 1))
        jittered["r"] = r
        result = compute_scattering(jittered, probes=("neutron",), **self.options)
        self.assertTrue(
            np.all(np.isfinite(result.neutron.weighted_rdf.to_numpy()))
        )

    def test_singleton_species_gets_an_explicit_ideal_self_partial(self) -> None:
        r = self.rdfs["r"]
        singleton = pd.DataFrame(
            {
                "r": r,
                "F-F": self.rdfs["F-F"],
                "F-Te": self.rdfs["F-Li"],
            }
        )
        singleton.attrs["system"] = {
            "atom_counts": {"F": 4, "Te": 1},
            "number_density": 0.05,
            "pairs": (("F", "F"), ("F", "Te")),
        }
        result = compute_scattering(
            singleton,
            probes=("neutron",),
            q_range=(0.0, 20.0),
            q_step=0.05,
        )
        np.testing.assert_allclose(result.partial_rdfs["Te-Te"], 1.0)
        np.testing.assert_allclose(result.neutron.structure_factor["Te-Te"], 0.0)
        self.assertEqual(result.metadata["synthetic_ideal_rdf_pairs"], ("Te-Te",))


if __name__ == "__main__":
    unittest.main()
