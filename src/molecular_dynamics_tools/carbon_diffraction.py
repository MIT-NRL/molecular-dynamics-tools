"""Bounded carbon diffraction and real-space size comparison helpers.

These measurements describe apparent coherent sizes, not unique crystallite
dimensions. The profile is an empirical split pseudo-Voigt, not a Warren model.
All reciprocal coordinates use Q = 4*pi*sin(theta)/wavelength in inverse Å.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist, pdist
from scipy.special import gammaincc, gammaln


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def carbon_xray_form_factor(q: ArrayLike) -> np.ndarray:
    """Neutral-carbon elastic f0(Q), in electrons; no anomalous corrections."""
    import periodictable as pt

    values = np.asarray(q, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Q must contain finite nonnegative values")
    return np.asarray(pt.C.xray.f0(values), dtype=float)


def _profile(q: np.ndarray, p: np.ndarray, reference: float) -> tuple[np.ndarray, np.ndarray]:
    center, amplitude, left, right, eta, intercept, slope = p
    width = np.where(q < center, left, right)
    scaled = (q - center) / width
    peak = amplitude * (
        eta / (1 + scaled**2) + (1 - eta) * np.exp(-math.log(2) * scaled**2)
    )
    background = intercept + slope * (q - reference)
    return peak + background, background


def fit_carbon_diffraction_peak(
    q: ArrayLike,
    intensity: ArrayLike,
    *,
    reflection: str = "10",
    q_window: tuple[float, float] | None = None,
    shape_factor: float | None = None,
    instrument_fwhm_q: float = 0.0,
    wavelength_A: float | None = None,
) -> dict[str, Any]:
    """Fit an asymmetric peak plus linear background, with conservative quality gates.

    NaN intensities remain unavailable in the returned curve. Size is withheld
    for sparse, truncated, unresolved, low-contrast or unidentifiable fits.
    Formal standard errors assume independent equal-variance residuals; they
    do not quantify uncertainty between correlated MD frames or model choice.
    A nonzero instrumental width invokes an explicitly approximate Gaussian
    quadrature subtraction; a full resolution convolution belongs upstream.
    """
    q = np.asarray(q, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    if q.ndim != 1 or intensity.shape != q.shape:
        raise ValueError("Q and intensity must be one-dimensional arrays of equal length")
    if np.any(~np.isfinite(q)) or np.any(q < 0) or np.any(np.diff(q) <= 0):
        raise ValueError("Q must be finite, nonnegative and strictly increasing")
    if reflection not in {"10", "002", "11", "004"}:
        raise ValueError("reflection must be 10, 002, 11 or 004")
    default_windows = {"10": (2.5, 3.6), "002": (1.2, 2.2), "11": (4.5, 5.7),
                       "004": (3.2, 4.2)}
    lo, hi = q_window or default_windows[reflection]
    lo, hi = float(lo), float(hi)
    if not np.isfinite([lo, hi]).all() or not 0 <= lo < hi:
        raise ValueError("q_window must contain finite increasing nonnegative bounds")
    factor = _positive(shape_factor if shape_factor is not None else
                       (0.9 if reflection in {"002", "004"} else 1), "shape_factor")
    instrument = float(instrument_fwhm_q)
    if not np.isfinite(instrument) or instrument < 0:
        raise ValueError("instrument_fwhm_q must be finite and nonnegative")
    wavelength = None if wavelength_A is None else _positive(wavelength_A, "wavelength_A")
    selected = (q >= lo) & (q <= hi)
    valid = selected & np.isfinite(intensity)
    x, y = q[valid], intensity[valid]
    rows = [{"q_invA": float(qi), "observed": float(yi) if np.isfinite(yi) else None,
             "fit": None, "background": None, "residual": None}
            for qi, yi in zip(q[selected], intensity[selected], strict=True)]
    summary: dict[str, Any] = {
        "reflection": reflection, "q_window_min_invA": lo, "q_window_max_invA": hi,
        "shape_factor": factor, "profile": "split_pseudo_voigt_linear_background",
        "status": "unavailable", "reason": "fewer than 12 finite points in fitting window",
        "point_count": int(len(x)), "q_peak_invA": None, "fwhm_invA": None,
        "La_A": None, "La_stderr_A": None, "Lc_A": None, "Lc_stderr_A": None,
        "d_spacing_A": None, "size_A": None, "size_stderr_A": None,
        "instrument_fwhm_invA": instrument, "wavelength_A": wavelength,
        "width_correction": "none" if instrument == 0 else "Gaussian_quadrature_approximation",
        "uncertainty_scope": "formal_fit_only_equal_variance_independent_residuals",
        "size_scope": "apparent_coherence_length_not_validated_physical_crystallite_size",
    }
    if len(x) < 12:
        return {"summary": summary, "curve": rows}
    span, reference = float(x[-1] - x[0]), float((x[-1] + x[0]) / 2)
    step = float(np.median(np.diff(x)))
    baseline = np.polyfit(np.r_[x[:3], x[-3:]] - reference, np.r_[y[:3], y[-3:]], 1)
    signal = y - np.polyval(baseline, x - reference)
    scale = max(float(np.ptp(y)), float(np.max(np.abs(y))) * 1e-8, 1e-12)
    minimum_width = max(step * 0.4, span * 1e-5)
    lower = [x[0], 0, minimum_width, minimum_width, 0, -np.inf, -np.inf]
    upper = [x[-1], scale * 50, span * 2, span * 2, 1, np.inf, np.inf]
    starts = np.unique(np.r_[x[np.argmax(signal)], np.quantile(x, [0.35, 0.55, 0.7])])
    candidates = []
    for center in starts:
        initial = [center, max(float(np.max(signal)), scale * 0.1), span / 8, span / 8,
                   0.5, baseline[1], baseline[0]]
        result = least_squares(
            lambda p: (_profile(x, p, reference)[0] - y) / scale,
            initial, bounds=(lower, upper), max_nfev=2500,
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
        )
        candidates.append(result)
    result = min(candidates, key=lambda item: float(np.sum(item.fun**2)))
    p = result.x
    center, amplitude, left, right, eta, _, _ = p
    fitted, _ = _profile(x, p, reference)
    residual = y - fitted
    noise = float(np.sqrt(np.mean(residual**2)))
    fwhm = float(left + right)
    errors = np.full(7, np.nan)
    covariance = None
    if np.linalg.matrix_rank(result.jac) == 7:
        covariance = np.linalg.pinv(result.jac.T @ result.jac) * (
            float(np.sum(result.fun**2)) / (len(x) - 7)
        )
        errors = np.sqrt(np.maximum(0, np.diag(covariance)))
    width_error = None if covariance is None else float(math.sqrt(max(
        0, covariance[2, 2] + covariance[3, 3] + 2 * covariance[2, 3]
    )))
    peak_area = float(amplitude * fwhm * (
        eta * math.pi / 2 + (1 - eta) * math.sqrt(math.pi) / (2 * math.sqrt(math.log(2)))
    ))
    reasons = []
    if not result.success:
        reasons.append("optimizer did not converge")
    if amplitude <= max(4 * noise, scale * 0.01):
        reasons.append("peak contrast is not resolved above residuals")
    if fwhm < 3 * step or min(left, right) < step:
        reasons.append("peak width is not resolved by Q sampling")
    if center - left <= x[0] + 2 * step or center + right >= x[-1] - 2 * step:
        reasons.append("half-maximum or background support reaches fitting boundary")
    central = x[(x >= center - left) & (x <= center + right)]
    if len(central) < 4 or (len(central) > 1 and np.max(np.diff(central)) >
                          max(2.5 * step, fwhm / 4)):
        reasons.append("missing reciprocal bins interrupt the peak")
    if not np.isfinite(errors[:4]).all() or (width_error is not None and width_error > fwhm / 2):
        reasons.append("peak parameters are not identifiable")
    if instrument >= fwhm:
        reasons.append("instrument width exceeds fitted width")
    width = math.sqrt(max(0, fwhm**2 - instrument**2))
    size = None
    if not reasons and width > 0:
        if wavelength is None:
            size = 2 * math.pi * factor / width
        elif (center + right * width / fwhm) * wavelength >= 4 * math.pi or \
                center - left * width / fwhm < 0:
            reasons.append("peak lies outside wavelength-accessible angular range")
        else:
            theta = math.asin(center * wavelength / (4 * math.pi))
            beta = 2 * (math.asin((center + right * width / fwhm) * wavelength /
                                 (4 * math.pi)) -
                        math.asin((center - left * width / fwhm) * wavelength /
                                  (4 * math.pi)))
            size = factor * wavelength / (beta * math.cos(theta))
    size_error = None
    if size is not None and covariance is not None:
        def converted_size(parameters: np.ndarray) -> float:
            q_center, half_left, half_right = parameters[:3]
            observed_width = half_left + half_right
            corrected_width = math.sqrt(observed_width**2 - instrument**2)
            if wavelength is None:
                return 2 * math.pi * factor / corrected_width
            conversion = corrected_width / observed_width
            angle = math.asin(q_center * wavelength / (4 * math.pi))
            angular_width = 2 * (
                math.asin((q_center + half_right * conversion) * wavelength / (4 * math.pi)) -
                math.asin((q_center - half_left * conversion) * wavelength / (4 * math.pi))
            )
            return factor * wavelength / (angular_width * math.cos(angle))

        # Propagate the fitted covariance, including center/width correlations
        # for the exact angular conversion. Instrument width is held fixed.
        parameters = np.array([center, left, right])
        gradient = np.zeros(3)
        for index in range(3):
            delta = max(abs(parameters[index]) * 1e-6, 1e-9)
            above, below = parameters.copy(), parameters.copy()
            above[index] += delta
            below[index] -= delta
            gradient[index] = (converted_size(above) - converted_size(below)) / (2 * delta)
        selected_covariance = covariance[np.ix_([0, 2, 3], [0, 2, 3])]
        size_error = float(math.sqrt(max(0, gradient @ selected_covariance @ gradient)))
    summary.update({
        "status": "review" if reasons else "pass", "reason": "; ".join(reasons) or
        "resolved empirical peak; finite-cell and profile-model sensitivity remain",
        "q_peak_invA": float(center), "q_peak_stderr_invA": float(errors[0])
        if np.isfinite(errors[0]) else None, "fwhm_invA": fwhm,
        "fwhm_stderr_invA": width_error, "corrected_fwhm_invA": width,
        "amplitude": float(amplitude), "integrated_peak_area": peak_area,
        "asymmetry_right_over_left": float(right / left), "lorentz_fraction": float(eta),
        "residual_rms": noise, "d_spacing_A": None if reasons else float(2 * math.pi / center),
        "size_A": size, "size_stderr_A": size_error,
        "La_A" if reflection in {"10", "11"} else "Lc_A": size,
        "La_stderr_A" if reflection in {"10", "11"} else "Lc_stderr_A": size_error,
    })
    fitted_all, background_all = _profile(q[selected], p, reference)
    for row, fit, background in zip(rows, fitted_all, background_all, strict=True):
        row.update(fit=float(fit), background=float(background), residual=(
            None if row["observed"] is None else float(row["observed"] - fit)
        ))
    return {"summary": summary, "curve": rows}


def summarize_crystallite_size_distribution(
    sizes_A: ArrayLike,
    *,
    intensity_exponent: float = 2.5,
    minimum_samples: int = 20,
    support_min_A: float | None = None,
    bin_width_A: float = 1.0,
    diffraction_size_A: float | None = None,
) -> dict[str, Any]:
    """Empirical size weighting and lower-truncated exponential maximum likelihood.

    Input sizes must be individual finite domains, including repeated sizes;
    periodic spanning domains have no finite size and must be excluded by the
    caller. Fits are descriptive: pooled correlated frames are not independent
    observations. Histogram probabilities sum to one; densities integrate to
    one. The weighted fitted mode is max(lower support, exponent*scale).
    """
    sizes = np.asarray(sizes_A, dtype=float)
    if sizes.ndim != 1 or np.any(~np.isfinite(sizes)) or np.any(sizes <= 0):
        raise ValueError("sizes_A must contain individual finite positive sizes")
    exponent = float(intensity_exponent)
    if not np.isfinite(exponent) or exponent < 0:
        raise ValueError("intensity_exponent must be finite and nonnegative")
    bin_width = _positive(bin_width_A, "bin_width_A")
    if isinstance(minimum_samples, bool) or int(minimum_samples) != minimum_samples or \
            minimum_samples < 3:
        raise ValueError("minimum_samples must be an integer of at least three")
    support = (float(np.min(sizes)) if len(sizes) else 0.0) if support_min_A is None else \
        float(support_min_A)
    if not np.isfinite(support) or support < 0 or (len(sizes) and support > np.min(sizes) + 1e-9):
        raise ValueError("support_min_A must be nonnegative and no greater than any size")
    if diffraction_size_A is not None:
        diffraction_size_A = _positive(diffraction_size_A, "diffraction_size_A")
    summary: dict[str, Any] = {
        "count": int(len(sizes)), "number_mean_A": None, "number_median_A": None,
        "weighted_mean_A": None, "weighted_empirical_mode_A": None,
        "exponential_scale_A": None, "weighted_fit_mode_A": None,
        "fit_status": "unavailable", "fit_reason": "no finite crystallites",
        "intensity_exponent": exponent, "support_min_A": support,
        "support_origin": "sample_minimum" if support_min_A is None else "caller_specified",
        "bin_width_A": bin_width, "diffraction_size_A": diffraction_size_A,
        "weight_scope": "L_power_intensity_proxy_not_a_calculated_diffraction_pattern",
        "uncertainty_scope": "descriptive_pooled_domains_no_independent_frame_uncertainty",
    }
    if not len(sizes):
        return {"summary": summary, "curve": []}
    first = math.floor(float(np.min(sizes)) / bin_width) * bin_width
    last = (math.floor(float(np.max(sizes)) / bin_width) + 1) * bin_width
    if (last - first) / bin_width > 100_000:
        raise ValueError("distribution exceeds 100000 histogram bins")
    edges = np.arange(first, last + bin_width * 0.5, bin_width)
    counts, _ = np.histogram(sizes, edges)
    weights = np.exp(exponent * (np.log(sizes) - np.log(np.max(sizes))))
    weighted_counts, _ = np.histogram(sizes, edges, weights=weights)
    centers = (edges[1:] + edges[:-1]) / 2
    empirical_mode = float(centers[np.argmax(weighted_counts)])
    summary.update(number_mean_A=float(np.mean(sizes)), number_median_A=float(np.median(sizes)),
                   weighted_mean_A=float(np.dot(sizes, weights) / weights.sum()),
                   weighted_empirical_mode_A=empirical_mode,
                   min_size_A=float(np.min(sizes)), max_size_A=float(np.max(sizes)))
    if diffraction_size_A is not None:
        below = sizes < diffraction_size_A
        summary.update(number_fraction_below_diffraction_size=float(np.mean(below)),
                       weighted_fraction_below_diffraction_size=float(weights[below].sum() /
                                                                      weights.sum()))
    scale = float(np.mean(sizes - support))
    unique = len(np.unique(np.round(sizes, 6)))
    if len(sizes) < minimum_samples:
        reason = f"fewer than {minimum_samples} finite domain observations"
    elif unique < 5 or scale <= max(1e-8, bin_width * 0.05):
        reason = "size support is too narrow or discrete for an exponential fit"
    else:
        reason = ""
    summary.update(fit_status="unavailable" if reason else "pass", fit_reason=reason or
                   "lower-truncated exponential descriptive maximum-likelihood fit")
    if not reason:
        summary.update(exponential_scale_A=scale,
                       weighted_fit_mode_A=max(support, exponent * scale))
        expected = 1 - np.exp(-(np.sort(sizes) - support) / scale)
        empirical = np.arange(1, len(sizes) + 1) / len(sizes)
        summary["exponential_cdf_max_deviation"] = float(max(
            np.max(np.abs(expected - empirical)),
            np.max(np.abs(expected - (empirical - 1 / len(sizes)))),
        ))
        summary["exponential_cdf_review_threshold"] = 0.2
        if summary["exponential_cdf_max_deviation"] > 0.2:
            summary.update(fit_status="review", fit_reason=
                           "exponential CDF deviation exceeds descriptive review threshold 0.2")
    rows = []
    for center, lower, upper, count, weighted in zip(
        centers, edges[:-1], edges[1:], counts, weighted_counts, strict=True
    ):
        number_fit = weighted_fit = None
        if not reason:
            a, b = max(support, float(lower)), max(support, float(upper))
            number_fit = (math.exp(-(a - support) / scale) -
                          math.exp(-(b - support) / scale)) / bin_width
            denominator = gammaincc(exponent + 1, support / scale)
            weighted_fit = float((gammaincc(exponent + 1, a / scale) -
                                  gammaincc(exponent + 1, b / scale)) /
                                 denominator / bin_width)
        rows.append({"sqrt_area_size_A": float(center), "bin_lower_A": float(lower),
                     "bin_upper_A": float(upper), "count": int(count),
                     "number_density": float(count / len(sizes) / bin_width),
                     "weighted_density": float(weighted / weights.sum() / bin_width),
                     "number_fit_density": number_fit, "weighted_fit_density": weighted_fit})
    if not reason:
        # Dense analytic curves include the lower bound and weighted mode, even
        # when the mode lies beyond the largest observed finite domain.
        stop = max(float(np.max(sizes)), exponent * scale, support + 6 * scale)
        dense = np.unique(np.r_[np.linspace(support, stop, 300), support,
                                max(support, exponent * scale)])
        normalizer = (exponent + 1) * math.log(scale) + gammaln(exponent + 1) + \
            math.log(gammaincc(exponent + 1, support / scale))
        fitted_curve = [{"sqrt_area_size_A": float(value),
                         "number_fit_density": math.exp(-(value - support) / scale) / scale,
                         "weighted_fit_density": (math.exp(-normalizer) if value == 0 and
                                                  exponent == 0 else 0.0 if value == 0 else
                                                  float(math.exp(
                             exponent * math.log(value) - value / scale - normalizer)))}
                        for value in dense]
    else:
        fitted_curve = []
    return {"summary": summary, "curve": rows, "fit_curve": fitted_curve}


def carbon_debye_pattern(
    positions: ArrayLike,
    q: ArrayLike,
    *,
    pair_bin_width_A: float = 0.001,
    block_size: int = 256,
    max_atoms: int = 5000,
) -> dict[str, np.ndarray]:
    """Isolated-fragment powder Debye pattern with bounded pair-distance buffers.

    This is a nonperiodic isolated-particle calculation. S(Q)=sum_ij sinc(Qr_ij)/N;
    I_per_atom=fC(Q)^2*S(Q); I_per_fragment=N*I_per_atom. No background,
    polarization, instrument resolution, Compton or anomalous terms are added.
    Histogram midpoint errors are controlled by pair_bin_width_A, not hidden
    smoothing. No N-by-N pair matrix or Q-by-all-pairs matrix is constructed.
    """
    positions = np.asarray(positions, dtype=float)
    q = np.asarray(q, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions) or \
            np.any(~np.isfinite(positions)):
        raise ValueError("positions must be a finite nonempty N-by-3 array")
    if q.ndim != 1 or not len(q) or np.any(~np.isfinite(q)) or np.any(q < 0):
        raise ValueError("Q must be a finite nonnegative one-dimensional array")
    if len(q) > 5000:
        raise ValueError("Debye helper is limited to 5000 Q points")
    for value, name in ((block_size, "block_size"), (max_atoms, "max_atoms")):
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if block_size > 1024 or max_atoms > 20000:
        raise ValueError("bounded calibration requires block_size <=1024 and max_atoms <=20000")
    if len(positions) > max_atoms:
        raise ValueError(f"fragment has {len(positions)} atoms, exceeding max_atoms={max_atoms}")
    step = _positive(pair_bin_width_A, "pair_bin_width_A")
    extent = float(np.linalg.norm(np.ptp(positions, axis=0)))
    bins = max(1, math.floor(extent / step) + 1)
    if bins > 1_000_000:
        raise ValueError("pair distance histogram exceeds one million bins")
    counts = np.zeros(bins, dtype=np.uint64)
    for begin in range(0, len(positions), block_size):
        first = positions[begin:begin + block_size]
        diagonal = pdist(first)
        counts += np.bincount(np.floor(diagonal / step).astype(np.int64), minlength=bins).astype(
            np.uint64
        )
        for second_begin in range(begin + block_size, len(positions), block_size):
            distances = cdist(first, positions[second_begin:second_begin + block_size]).ravel()
            counts += np.bincount(np.floor(distances / step).astype(np.int64),
                                  minlength=bins).astype(np.uint64)
    occupied = np.flatnonzero(counts)
    radii = (occupied + 0.5) * step
    pair_counts = counts[occupied].astype(float)
    sq = np.ones(len(q))
    # Bound spectral scratch space to roughly 4 MiB, regardless of fragment extent.
    for q_begin in range(0, len(q), 32):
        current = q[q_begin:q_begin + 32]
        accumulated = np.zeros(len(current))
        for r_begin in range(0, len(radii), 4096):
            distances = radii[r_begin:r_begin + 4096]
            kernel = np.sinc(current[:, None] * distances[None, :] / math.pi)
            accumulated += kernel @ pair_counts[r_begin:r_begin + 4096]
        sq[q_begin:q_begin + 32] += 2 * accumulated / len(positions)
    f = carbon_xray_form_factor(q)
    per_atom = f**2 * sq
    return {"q_invA": q.copy(), "S_q": sq, "I_per_atom": per_atom,
            "I_per_fragment": len(positions) * per_atom}


def _graphene_disc(target_size: float, bond: float, max_atoms: int) -> tuple[np.ndarray, float]:
    atom_area = 3 * math.sqrt(3) * bond**2 / 4
    radius = target_size / math.sqrt(math.pi)
    if target_size**2 / atom_area > max_atoms * 1.2:
        raise ValueError("requested fragment would exceed atom resource limit")
    basis = np.array([[1.5 * bond, math.sqrt(3) * bond / 2],
                      [1.5 * bond, -math.sqrt(3) * bond / 2]])
    bound = math.ceil(2 * radius / (math.sqrt(3) * bond)) + 2
    points = []
    center = np.array([bond / 2, math.sqrt(3) * bond / 2])
    for first in range(-bound, bound + 1):
        for second in range(-bound, bound + 1):
            origin = np.array([first, second]) @ basis - center
            for offset in (np.zeros(2), np.array([bond, 0])):
                point = origin + offset
                if np.dot(point, point) <= radius**2 + 1e-10:
                    points.append([float(point[0]), float(point[1]), 0.0])
    if not points or len(points) > max_atoms:
        raise ValueError("fragment is empty or exceeds atom resource limit")
    return np.asarray(points), atom_area


def _power_law(sizes: np.ndarray, intensities: np.ndarray) -> dict[str, float]:
    x, y = np.log(sizes), np.log(intensities)
    design = np.column_stack((np.ones(len(x)), x))
    intercept, exponent = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ np.array([intercept, exponent])
    covariance = np.linalg.inv(design.T @ design) * float(np.dot(residual, residual)) / (len(x) - 2)
    variance = float(np.sum((y - np.mean(y))**2))
    return {"coefficient": float(math.exp(intercept)), "exponent": float(exponent),
            "exponent_stderr": float(math.sqrt(max(0, covariance[1, 1]))),
            "log_r_squared": 1 - float(np.dot(residual, residual)) / variance if variance else 0.0}


def calibrate_graphene_diffraction(
    sizes_A: ArrayLike = (8, 10, 15, 22, 30, 40),
    *,
    bond_length_A: float = 1.42,
    q_window: tuple[float, float] = (2.5, 3.6),
    q_step: float = 0.005,
    pair_bin_width_A: float = 0.001,
    max_atoms: int = 5000,
) -> dict[str, Any]:
    """Small Fig. 3-style isolated graphene-disc intensity calibration.

    Fit the power-law exponent freely, using the raw maximum (10) intensity
    per fragment; also report the per-atom exponent to expose normalization.
    L=sqrt(N*bulk area per atom) approximates the disc area, including edge
    atoms; this convention differs from a sum of full interior hexagon areas.
    Small discs are a calibration, not a reproduction of the paper's 50 nm
    example or evidence for a universal exponent.
    """
    sizes = np.asarray(sizes_A, dtype=float)
    if sizes.ndim != 1 or len(sizes) < 3 or len(sizes) > 20 or \
            np.any(~np.isfinite(sizes)) or np.any(sizes <= 0):
        raise ValueError("sizes_A must contain 3 to 20 finite positive sizes")
    bond = _positive(bond_length_A, "bond_length_A")
    step = _positive(q_step, "q_step")
    lo, hi = map(float, q_window)
    if not np.isfinite([lo, hi]).all() or not 0 < lo < hi:
        raise ValueError("q_window must contain finite positive increasing bounds")
    bins = math.ceil((hi - lo) / step)
    if bins > 4999:
        raise ValueError("calibration exceeds 5000 Q points")
    q = np.linspace(lo, hi, bins + 1)
    fragments, patterns = [], []
    for target in sizes:
        positions, atom_area = _graphene_disc(float(target), bond, max_atoms)
        actual = math.sqrt(len(positions) * atom_area)
        pattern = carbon_debye_pattern(positions, q, pair_bin_width_A=pair_bin_width_A,
                                      max_atoms=max_atoms)
        peak = int(np.argmax(pattern["I_per_fragment"]))
        fragments.append({"L_A": actual, "requested_L_A": float(target),
                          "atom_count": len(positions), "area_A2": len(positions) * atom_area,
                          "Imax_per_atom": float(pattern["I_per_atom"][peak]),
                          "Imax_per_fragment": float(pattern["I_per_fragment"][peak]),
                          "peak_q_invA": float(q[peak]), "peak_at_boundary": peak in {0, len(q)-1}})
        for index, qi in enumerate(q):
            patterns.append({"L_A": actual, "q_invA": float(qi),
                             "S_q": float(pattern["S_q"][index]),
                             "I_per_atom": float(pattern["I_per_atom"][index]),
                             "I_per_fragment": float(pattern["I_per_fragment"][index])})
    actual_sizes = np.array([row["L_A"] for row in fragments])
    distinct = len(np.unique(actual_sizes)) >= 3
    bounded = any(row["peak_at_boundary"] for row in fragments)
    summary: dict[str, Any] = {
        "status": "review" if bounded or not distinct else "pass",
        "reason": "peak reaches Q window boundary" if bounded else
        ("fewer than three distinct atomistic fragment sizes" if not distinct else
         "small isolated-disc calibration; size-range convergence remains required"),
        "bond_length_A": bond, "pair_bin_width_A": float(pair_bin_width_A),
        "q_min_invA": lo, "q_max_invA": hi, "q_step_invA": float(q[1]-q[0]),
        "size_convention": "sqrt_atom_count_times_bulk_graphene_area_per_atom",
        "intensity_convention": "coherent_raw_Debye_per_fragment_in_electron_squared",
        "reference_exponent": 2.5, "exponent": None, "coefficient": None,
        "per_atom_exponent": None, "fragment_count": len(fragments),
        "max_atom_count": max(row["atom_count"] for row in fragments),
        "reference": "doi:10.1016/j.carbon.2023.03.040; arXiv:2212.06354 Fig.3",
    }
    if distinct and not bounded:
        fragment_fit = _power_law(actual_sizes, np.array([
            row["Imax_per_fragment"] for row in fragments
        ]))
        atom_fit = _power_law(actual_sizes, np.array([row["Imax_per_atom"] for row in fragments]))
        summary.update(fragment_fit)
        summary.update(per_atom_exponent=atom_fit["exponent"],
                       per_atom_exponent_stderr=atom_fit["exponent_stderr"],
                       per_atom_log_r_squared=atom_fit["log_r_squared"])
    for row in fragments:
        row["predicted_Imax_per_fragment"] = None if summary["exponent"] is None else \
            summary["coefficient"] * row["L_A"]**summary["exponent"]
    return {"summary": summary, "fragments": fragments, "patterns": patterns}
