"""Independent reference checks for carbon peak and size comparisons."""

import math

import numpy as np
import pytest
from scipy.integrate import trapezoid

import molecular_dynamics_tools as mdt


def test_asymmetric_peak_recovers_width_and_apparent_size():
    q = np.linspace(2.5, 3.6, 551)
    center, left, right, amplitude = 2.98, 0.055, 0.09, 7.0
    width = np.where(q < center, left, right)
    signal = amplitude * np.exp(-math.log(2) * ((q - center) / width)**2)
    result = mdt.fit_carbon_diffraction_peak(q, 2 + 0.3 * (q - 3) + signal)
    summary = result["summary"]
    assert summary["status"] == "pass"
    assert summary["q_peak_invA"] == pytest.approx(center, abs=1e-6)
    assert summary["fwhm_invA"] == pytest.approx(left + right, rel=1e-4)
    assert summary["La_A"] == pytest.approx(2 * math.pi / (left + right), rel=1e-4)
    assert summary["asymmetry_right_over_left"] == pytest.approx(right / left, rel=1e-4)
    assert summary["residual_rms"] < 1e-6
    assert summary["integrated_peak_area"] == pytest.approx(
        amplitude * (left + right) * math.sqrt(math.pi) / (2 * math.sqrt(math.log(2))),
        rel=1e-4,
    )


def test_width_correction_and_exact_wavelength_conversion():
    q = np.linspace(1.2, 2.2, 501)
    signal = 3 + 12 * np.exp(-math.log(2) * ((q - 1.86) / 0.055)**2)
    result = mdt.fit_carbon_diffraction_peak(q, signal, reflection="002", wavelength_A=1.5406,
                                           instrument_fwhm_q=0.02)
    summary = result["summary"]
    width = math.sqrt(0.11**2 - 0.02**2)
    theta = math.asin(1.86 * 1.5406 / (4 * math.pi))
    beta = 2 * (math.asin((1.86 + width/2) * 1.5406 / (4 * math.pi)) -
                math.asin((1.86 - width/2) * 1.5406 / (4 * math.pi)))
    assert summary["status"] == "pass"
    assert summary["Lc_A"] == pytest.approx(0.9 * 1.5406 / (beta * math.cos(theta)), rel=1e-4)
    assert summary["La_A"] is None
    assert summary["d_spacing_A"] == pytest.approx(2 * math.pi / 1.86, rel=1e-5)
    assert summary["width_correction"] == "Gaussian_quadrature_approximation"


def test_exact_wavelength_conversion_uses_asymmetric_halfmaximum_locations():
    q = np.linspace(2.5, 3.6, 551)
    center, left, right, wavelength = 2.95, 0.06, 0.11, 1.5406
    signal = 2 + 7 * np.exp(-math.log(2) * ((q-center) /
                                          np.where(q < center, left, right))**2)
    summary = mdt.fit_carbon_diffraction_peak(q, signal, wavelength_A=wavelength)["summary"]
    beta = 2 * (math.asin((center + right) * wavelength / (4 * math.pi)) -
                math.asin((center - left) * wavelength / (4 * math.pi)))
    expected = wavelength / (beta * math.cos(math.asin(center*wavelength/(4*math.pi))))
    assert summary["La_A"] == pytest.approx(expected, rel=1e-4)


def test_missing_and_unresolved_peaks_do_not_report_sizes():
    q = np.linspace(2.5, 3.6, 151)
    for signal in (np.ones(len(q)), 1 + np.exp(-((q - 2.54) / 0.1)**2),
                   1 + np.exp(-((q - 3.0) / 0.003)**2)):
        summary = mdt.fit_carbon_diffraction_peak(q, signal)["summary"]
        assert summary["status"] == "review"
        assert summary["La_A"] is None
        assert summary["d_spacing_A"] is None
        assert summary["reason"]
    sparse = np.full(len(q), np.nan)
    sparse[:6] = 2
    result = mdt.fit_carbon_diffraction_peak(q, sparse)
    assert result["summary"]["status"] == "unavailable"
    assert result["summary"]["La_A"] is None
    assert result["curve"][-1]["observed"] is None


def test_reciprocal_gap_is_preserved_and_blocks_width_measurement():
    q = np.linspace(2.5, 3.6, 301)
    y = 1 + 5 * np.exp(-math.log(2) * ((q - 3.0) / 0.08)**2)
    y[(q > 2.985) & (q < 3.03)] = np.nan
    result = mdt.fit_carbon_diffraction_peak(q, y)
    assert result["summary"]["La_A"] is None
    assert "missing reciprocal bins" in result["summary"]["reason"]
    assert any(row["observed"] is None and row["residual"] is None for row in result["curve"])


def test_size_distribution_uses_exact_individual_weights():
    sizes = np.array([3.5, 3.5, 4, 6, 8])
    result = mdt.summarize_crystallite_size_distribution(sizes, bin_width_A=0.5,
                                                      diffraction_size_A=6)
    summary = result["summary"]
    weights = sizes**2.5
    assert summary["number_mean_A"] == pytest.approx(sizes.mean())
    assert summary["number_median_A"] == pytest.approx(np.median(sizes))
    assert summary["weighted_mean_A"] == pytest.approx(np.dot(sizes, weights) / weights.sum())
    assert summary["number_fraction_below_diffraction_size"] == pytest.approx(3/5)
    assert summary["weighted_fraction_below_diffraction_size"] == pytest.approx(
        weights[sizes < 6].sum() / weights.sum()
    )
    assert sum(row["number_density"] * 0.5 for row in result["curve"]) == pytest.approx(1)
    assert sum(row["weighted_density"] * 0.5 for row in result["curve"]) == pytest.approx(1)
    assert summary["fit_status"] == "unavailable"
    assert summary["exponential_scale_A"] is None


def test_lower_truncated_exponential_and_weighted_mode():
    # Deterministic exponential quantiles independently identify the expected scale.
    lower, scale = 3.95, 4.0
    probability = (np.arange(10000) + 0.5) / 10000
    sizes = lower - scale * np.log(1 - probability)
    result = mdt.summarize_crystallite_size_distribution(sizes, support_min_A=lower)
    summary = result["summary"]
    assert summary["fit_status"] == "pass"
    assert summary["exponential_scale_A"] == pytest.approx(scale, rel=1e-3)
    assert summary["weighted_fit_mode_A"] == pytest.approx(2.5 * scale, rel=1e-3)
    curve = result["fit_curve"]
    assert trapezoid([row["weighted_fit_density"] for row in curve],
                     [row["sqrt_area_size_A"] for row in curve]) == pytest.approx(1, abs=0.01)
    narrow = mdt.summarize_crystallite_size_distribution(lower - 0.1 * np.log(1-probability),
                                                       support_min_A=lower)
    assert narrow["summary"]["weighted_fit_mode_A"] == lower
    discrete = mdt.summarize_crystallite_size_distribution([4, 5, 6] * 100)
    assert discrete["summary"]["fit_status"] == "unavailable"
    nonexponential = np.linspace(4, 5, 100)
    reviewed = mdt.summarize_crystallite_size_distribution(nonexponential, support_min_A=0)
    assert reviewed["summary"]["fit_status"] == "review"
    assert reviewed["summary"]["exponential_scale_A"] is not None
    uniform_weights = mdt.summarize_crystallite_size_distribution(sizes, intensity_exponent=0)
    assert uniform_weights["summary"]["weighted_mean_A"] == pytest.approx(sizes.mean())


def test_debye_matches_independent_pair_sum_and_normalization():
    positions = np.array([[0, 0, 0], [1.42, 0, 0], [0.71, 1.229756, 0], [0, 0, 2.5]])
    q = np.array([0, 0.7, 2.98, 10.0])
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    expected = np.array([np.sinc(qi * distances / math.pi).sum() / len(positions) for qi in q])
    result = mdt.carbon_debye_pattern(positions, q, pair_bin_width_A=1e-5, block_size=2)
    assert result["S_q"] == pytest.approx(expected, abs=3e-5)
    assert result["S_q"][0] == pytest.approx(len(positions))
    f = mdt.carbon_xray_form_factor(q)
    assert f[0] == pytest.approx(6, abs=0.01)
    assert result["I_per_atom"] == pytest.approx(result["S_q"] * f**2)
    assert result["I_per_fragment"] == pytest.approx(len(positions) * result["I_per_atom"])
    alternative = mdt.carbon_debye_pattern(positions, q, pair_bin_width_A=1e-5, block_size=3)
    assert alternative["S_q"] == pytest.approx(result["S_q"], abs=1e-12)


def test_fragment_calibration_is_measured_and_normalization_explicit():
    result = mdt.calibrate_graphene_diffraction((10, 15, 22), q_step=0.01)
    summary = result["summary"]
    assert summary["status"] == "pass"
    assert summary["exponent"] > 2
    assert summary["exponent"] - summary["per_atom_exponent"] == pytest.approx(2, abs=1e-8)
    assert summary["max_atom_count"] < 250
    assert len(result["patterns"]) == 3 * 111
    for fragment in result["fragments"]:
        assert fragment["Imax_per_fragment"] == pytest.approx(
            fragment["atom_count"] * fragment["Imax_per_atom"]
        )
        assert fragment["predicted_Imax_per_fragment"] > 0
        assert 2.5 < fragment["peak_q_invA"] < 3.6


def test_resource_and_input_limits():
    with pytest.raises(ValueError, match="exceeding"):
        mdt.carbon_debye_pattern(np.zeros((5, 3)), [1], max_atoms=4)
    with pytest.raises(ValueError, match="one million"):
        mdt.carbon_debye_pattern([[0, 0, 0], [100, 0, 0]], [1], pair_bin_width_A=1e-6)
    with pytest.raises(ValueError, match="resource limit"):
        mdt.calibrate_graphene_diffraction((10, 20, 1000))
    with pytest.raises(ValueError, match="positive"):
        mdt.summarize_crystallite_size_distribution([0, 2])
    with pytest.raises(ValueError, match="strictly increasing"):
        mdt.fit_carbon_diffraction_peak([1, 1], [2, 2])
    with pytest.raises(ValueError, match="support_min"):
        mdt.summarize_crystallite_size_distribution([3, 4], support_min_A=5)
