"""Partial structure factors and weighted neutron/X-ray PDFs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

from molecular_dynamics_tools.transforms import (
    IntegrationMethod,
    TransformBackend,
    spherical_bessel_transform,
)

from .composition import IsotopeMixture, ScatteringComposition
from .weights import (
    Pair,
    ScatteringProbe,
    StructureFactorConvention,
    compute_scattering_weights,
    normalize_pairs,
    pair_label,
    unique_pairs,
)


CompositionInput = ScatteringComposition | Mapping[str, float] | str | None


@dataclass(frozen=True, slots=True)
class ProbeScatteringResult:
    """Weighted reciprocal- and real-space outputs for one radiation probe."""

    probe: ScatteringProbe
    weights: pd.DataFrame
    structure_factor: pd.DataFrame
    weighted_rdf: pd.DataFrame


@dataclass(frozen=True, slots=True)
class ScatteringResult:
    """Complete scattering calculation while retaining unweighted inputs."""

    partial_rdfs: pd.DataFrame
    partial_structure_factors: pd.DataFrame
    composition: ScatteringComposition
    convention: StructureFactorConvention
    neutron: ProbeScatteringResult | None
    xray: ProbeScatteringResult | None
    metadata: dict


def _coerce_composition(
    rdfs: pd.DataFrame,
    composition: CompositionInput,
    isotopes: Mapping[str, IsotopeMixture] | None,
    charges: Mapping[str, int] | None,
    elements: Mapping[str, str] | None,
) -> ScatteringComposition:
    if isinstance(composition, ScatteringComposition):
        if isotopes or charges or elements:
            raise ValueError(
                "isotopes, charges, and elements must be stored in the supplied "
                "ScatteringComposition"
            )
        return composition
    if composition is None:
        system = rdfs.attrs.get("system", {})
        composition = system.get("atom_counts")
        if composition is None:
            raise ValueError(
                "composition is required when RDF metadata has no atom_counts"
            )
    if isinstance(composition, str):
        if elements:
            raise ValueError("elements cannot be combined with a formula composition")
        return ScatteringComposition.from_formula(
            composition, isotopes=isotopes, charges=charges
        )
    if isinstance(composition, Mapping):
        return ScatteringComposition(
            composition,
            isotopes=isotopes or {},
            charges=charges or {},
            elements=elements or {},
        )
    raise TypeError("composition must be a mapping, formula, or ScatteringComposition")


def _resolve_number_density(rdfs: pd.DataFrame, number_density: float | None) -> float:
    if number_density is None:
        number_density = rdfs.attrs.get("system", {}).get("number_density")
        if number_density is None:
            raise ValueError(
                "number_density in atoms/angstrom^3 is required when RDF metadata "
                "does not contain it"
            )
    value = float(number_density)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("number_density must be finite and positive")
    return value


def _complete_singleton_partials(
    rdfs: pd.DataFrame, composition: ScatteringComposition
) -> pd.DataFrame:
    """Supply g_ii(r)=1 only when an RDF has exactly one atom of species i."""

    system = rdfs.attrs.get("system", {})
    atom_counts = system.get("atom_counts")
    if not isinstance(atom_counts, Mapping):
        return rdfs
    synthetic = []
    for species in composition.species:
        label = pair_label((species, species))
        if label not in rdfs and atom_counts.get(species) == 1:
            synthetic.append(label)
    if not synthetic:
        return rdfs

    completed = rdfs.copy()
    for label in synthetic:
        completed[label] = 1.0
    ordered_pairs = unique_pairs(composition.species)
    completed = completed[["r", *[pair_label(pair) for pair in ordered_pairs]]]
    completed.attrs = dict(rdfs.attrs)
    completed_system = dict(system)
    completed_system["pairs"] = ordered_pairs
    completed_system["synthetic_ideal_rdf_pairs"] = tuple(synthetic)
    completed.attrs["system"] = completed_system
    return completed


def _resolve_pairs(
    rdfs: pd.DataFrame,
    composition: ScatteringComposition,
    pairs: Sequence[Pair] | None,
) -> tuple[Pair, ...]:
    if "r" not in rdfs:
        raise ValueError("RDF input must contain an 'r' column")
    if pairs is None:
        stored = rdfs.attrs.get("system", {}).get("pairs")
        if stored is not None:
            pairs = tuple(tuple(pair) for pair in stored)
        else:
            parsed: list[Pair] = []
            for column in rdfs.columns:
                if column == "r":
                    continue
                pieces = str(column).split("-")
                if len(pieces) != 2:
                    raise ValueError(
                        "cannot infer pair identities from RDF columns; pass pairs=..."
                    )
                parsed.append((pieces[0], pieces[1]))
            pairs = tuple(parsed)
    normalized = normalize_pairs(pairs, composition)
    expected = set(unique_pairs(composition.species))
    supplied = set(normalized)
    if supplied != expected:
        missing = sorted(pair_label(pair) for pair in expected - supplied)
        extra = sorted(pair_label(pair) for pair in supplied - expected)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise ValueError(
            "a total scattering calculation requires every unique pair: "
            + "; ".join(details)
        )
    for pair in normalized:
        label = pair_label(pair)
        if label not in rdfs:
            raise ValueError(f"RDF input is missing canonical pair column {label!r}")
    return normalized


def _resolve_q_grid(
    r: np.ndarray,
    q_range: tuple[float, float | None],
    q_step: float | None,
) -> np.ndarray:
    if len(q_range) != 2:
        raise ValueError("q_range must contain (q_min, q_max)")
    steps = np.diff(r)
    if np.any(steps <= 0):
        raise ValueError("RDF r values must be strictly increasing")
    if not np.allclose(steps, steps[0], rtol=1e-8):
        raise ValueError("the ZoomFFT scattering path requires a uniform RDF grid")
    nyquist = np.pi / float(steps[0])
    q_min = float(q_range[0])
    q_max = min(25.0, nyquist) if q_range[1] is None else float(q_range[1])
    if not np.isfinite(q_min) or q_min < 0:
        raise ValueError("q_min must be finite and nonnegative")
    if not np.isfinite(q_max) or q_max <= q_min:
        raise ValueError("q_max must be finite and greater than q_min")
    if q_max > nyquist * (1.0 + 1e-12):
        raise ValueError(
            f"q_max={q_max:g} exceeds the RDF Nyquist limit {nyquist:g} 1/angstrom"
        )
    if q_step is None:
        q_step = np.pi / (r[-1] - r[0])
    step = float(q_step)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("q_step must be finite and positive")
    point_count = int(np.floor((q_max - q_min) / step + 1e-12)) + 1
    if point_count < 2:
        raise ValueError("q_range and q_step must produce at least two Q points")
    return q_min + step * np.arange(point_count, dtype=np.float64)


def _prepare_rdfs(rdfs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(rdfs, pd.DataFrame):
        raise TypeError("rdfs must be a pandas DataFrame")
    if "r" not in rdfs:
        raise ValueError("RDF input must contain an 'r' column")
    r = rdfs["r"].to_numpy(dtype=np.float64)
    if r.ndim != 1 or len(r) < 2 or np.any(~np.isfinite(r)):
        raise ValueError("RDF r values must be a finite one-dimensional grid")
    steps = np.diff(r)
    if np.any(steps <= 0):
        raise ValueError("RDF r values must be strictly increasing")
    median_step = float(np.median(steps))
    relative_jitter = float(np.max(np.abs(steps - median_step)) / median_step)
    if relative_jitter > 1e-3:
        raise ValueError(
            "RDF grid is materially nonuniform; resample it or use an exact "
            "uniform-grid MDT RDF"
        )
    if not np.allclose(steps, median_step, rtol=1e-8, atol=0.0):
        r = np.linspace(r[0], r[-1], len(r), dtype=np.float64)
        steps = np.diff(r)
    return r, steps


def compute_partial_structure_factors(
    rdfs: pd.DataFrame,
    *,
    composition: CompositionInput = None,
    number_density: float | None = None,
    pairs: Sequence[Pair] | None = None,
    convention: StructureFactorConvention = "faber-ziman",
    q_range: tuple[float, float | None] = (0.0, None),
    q_step: float | None = None,
    backend: TransformBackend = "zoomfft",
    integration: IntegrationMethod = "uniform",
    chunk_size: int = 512,
) -> pd.DataFrame:
    """Transform partial RDFs into Faber–Ziman or Ashcroft–Langreth partials."""

    if convention not in {"faber-ziman", "ashcroft-langreth"}:
        raise ValueError("convention must be 'faber-ziman' or 'ashcroft-langreth'")
    resolved_composition = _coerce_composition(rdfs, composition, None, None, None)
    completed_rdfs = _complete_singleton_partials(rdfs, resolved_composition)
    r, _ = _prepare_rdfs(completed_rdfs)
    resolved_pairs = _resolve_pairs(completed_rdfs, resolved_composition, pairs)
    density = _resolve_number_density(completed_rdfs, number_density)
    q = _resolve_q_grid(r, q_range, q_step)
    labels = [pair_label(pair) for pair in resolved_pairs]
    h_values = completed_rdfs[labels].to_numpy(dtype=np.float64) - 1.0
    if np.any(~np.isfinite(h_values)):
        raise ValueError("partial RDF values must be finite")
    transformed = spherical_bessel_transform(
        r,
        h_values,
        q,
        backend=backend,
        integration=integration,
        chunk_size=chunk_size,
    )
    faber_ziman_delta = 4.0 * np.pi * density * transformed
    concentrations = resolved_composition.concentrations

    result = pd.DataFrame({"Q": q})
    for pair_index, pair in enumerate(resolved_pairs):
        if convention == "faber-ziman":
            baseline = 1.0
            delta = faber_ziman_delta[:, pair_index]
        else:
            baseline = 1.0 if pair[0] == pair[1] else 0.0
            delta = (
                np.sqrt(concentrations[pair[0]] * concentrations[pair[1]])
                * faber_ziman_delta[:, pair_index]
            )
        result[pair_label(pair)] = baseline + delta

    result.attrs["convention"] = convention
    result.attrs["number_density"] = density
    result.attrs["pairs"] = resolved_pairs
    result.attrs["q_step"] = float(q[1] - q[0])
    result.attrs["transform"] = {
        "backend": backend,
        "integration": integration,
        "chunk_size": chunk_size,
    }
    return result


def _modification_function(q: np.ndarray, window: str | None) -> np.ndarray:
    if window is None:
        return np.ones_like(q)
    if window == "lorch":
        return np.sinc(q / q[-1])
    raise ValueError("window must be None or 'lorch'")


def _probe_result(
    rdfs: pd.DataFrame,
    source_r: np.ndarray,
    partials: pd.DataFrame,
    composition: ScatteringComposition,
    pairs: tuple[Pair, ...],
    probe: ScatteringProbe,
    convention: StructureFactorConvention,
    number_density: float,
    r_output: np.ndarray,
    xray_window: str | None,
    backend: TransformBackend,
    integration: IntegrationMethod,
    chunk_size: int,
) -> ProbeScatteringResult:
    q = partials["Q"].to_numpy(dtype=np.float64)
    weights = compute_scattering_weights(
        composition, pairs, q, probe=probe, convention=convention
    )
    labels = [pair_label(pair) for pair in pairs]
    correlation = np.empty((len(q), len(pairs)), dtype=np.float64)
    for pair_index, pair in enumerate(pairs):
        baseline = (
            1.0
            if convention == "faber-ziman" or pair[0] == pair[1]
            else 0.0
        )
        correlation[:, pair_index] = partials[labels[pair_index]] - baseline
    contributions = correlation * weights[labels].to_numpy(dtype=np.float64)

    structure_factor = pd.DataFrame(contributions, columns=labels)
    structure_factor.insert(0, "Q", q)
    structure_factor["Total"] = 1.0 + contributions.sum(axis=1)
    structure_factor.attrs["pair_columns"] = "contributions to S(Q) - 1"
    structure_factor.attrs["probe"] = probe
    structure_factor.attrs["convention"] = convention

    # Allocate the unit baseline across pairs using their convention-specific
    # baseline at the first sampled Q. This keeps pair columns additive while
    # ensuring that their sum is a normalized total g(r) approaching one.
    partial_baselines = np.asarray(
        [
            1.0
            if convention == "faber-ziman" or pair[0] == pair[1]
            else 0.0
            for pair in pairs
        ],
        dtype=np.float64,
    )
    pair_baselines = (
        weights.loc[weights.index[0], labels].to_numpy(dtype=np.float64)
        * partial_baselines
    )
    if probe == "neutron":
        if r_output[0] < source_r[0] or r_output[-1] > source_r[-1]:
            raise ValueError(
                "r_grid must remain within the input RDF range for direct "
                "neutron weighting"
            )
        source_values = rdfs[labels].to_numpy(dtype=np.float64)
        if np.array_equal(r_output, source_r):
            interpolated = source_values
        else:
            interpolated = np.column_stack(
                [
                    np.interp(r_output, source_r, source_values[:, index])
                    for index in range(len(labels))
                ]
            )
        concentrations = composition.concentrations
        correlation_scales = np.asarray(
            [
                1.0
                if convention == "faber-ziman"
                else np.sqrt(concentrations[pair[0]] * concentrations[pair[1]])
                for pair in pairs
            ],
            dtype=np.float64,
        )
        radial_correlations = (
            (interpolated - 1.0)
            * weights.loc[weights.index[0], labels].to_numpy(dtype=np.float64)
            * correlation_scales
        )
        real_space_method = "direct weighting of partial RDFs"
        applied_window = None
    else:
        modified = contributions * _modification_function(
            q, xray_window
        )[:, np.newaxis]
        transformed = spherical_bessel_transform(
            q,
            modified,
            r_output,
            backend=backend,
            integration=integration,
            chunk_size=chunk_size,
        )
        # Invert weighted S(Q)-1 to dimensionless h(r)=g(r)-1. The transform
        # evaluates integral q^2 F(q) sinc(qr) dq.
        radial_correlations = transformed / (2.0 * np.pi**2 * number_density)
        real_space_method = "inverse transform of Q-dependent X-ray weighting"
        applied_window = xray_window

    weighted_values = radial_correlations + pair_baselines[np.newaxis, :]
    weighted_rdf = pd.DataFrame(weighted_values, columns=labels)
    weighted_rdf.insert(0, "r", r_output)
    weighted_rdf["Total"] = weighted_values.sum(axis=1)
    weighted_rdf.attrs["definition"] = (
        "g(r) - 1 = [1/(2*pi^2*rho)] integral "
        "Q^2[S(Q)-1] sinc(Qr) dQ"
    )
    weighted_rdf.attrs["pair_columns"] = (
        "additive weighted contributions to g(r), including allocated baselines"
    )
    weighted_rdf.attrs["pair_baselines"] = dict(zip(labels, pair_baselines))
    weighted_rdf.attrs["normalization"] = "dimensionless; Total approaches 1"
    weighted_rdf.attrs["probe"] = probe
    weighted_rdf.attrs["convention"] = convention
    weighted_rdf.attrs["method"] = real_space_method
    weighted_rdf.attrs["window"] = applied_window
    weighted_rdf.attrs["q_max"] = float(q[-1])
    weighted_rdf.attrs["number_density"] = number_density
    return ProbeScatteringResult(
        probe, weights, structure_factor, weighted_rdf
    )


def compute_scattering(
    rdfs: pd.DataFrame,
    *,
    composition: CompositionInput = None,
    number_density: float | None = None,
    pairs: Sequence[Pair] | None = None,
    isotopes: Mapping[str, IsotopeMixture] | None = None,
    charges: Mapping[str, int] | None = None,
    elements: Mapping[str, str] | None = None,
    probes: Iterable[ScatteringProbe] = ("neutron", "xray"),
    convention: StructureFactorConvention = "faber-ziman",
    q_range: tuple[float, float | None] = (0.0, None),
    q_step: float | None = None,
    r_grid: ArrayLike | None = None,
    xray_window: str | None = "lorch",
    backend: TransformBackend = "zoomfft",
    integration: IntegrationMethod = "uniform",
    chunk_size: int = 512,
) -> ScatteringResult:
    """Calculate normalized neutron and/or X-ray ``S(Q)`` and weighted ``g(r)``.

    The input partial RDFs remain unweighted. Pair columns in each probe's
    structure-factor table are weighted contributions to ``S(Q)-1``; pair
    columns in each weighted-RDF table are additive contributions to a
    dimensionless total ``g(r)`` with a unit long-range baseline.
    """

    resolved_composition = _coerce_composition(
        rdfs, composition, isotopes, charges, elements
    )
    completed_rdfs = _complete_singleton_partials(rdfs, resolved_composition)
    r, _ = _prepare_rdfs(completed_rdfs)
    resolved_pairs = _resolve_pairs(completed_rdfs, resolved_composition, pairs)
    density = _resolve_number_density(completed_rdfs, number_density)
    requested_probes = tuple(str(probe) for probe in probes)
    if not requested_probes or len(set(requested_probes)) != len(requested_probes):
        raise ValueError("probes must contain one or both unique values")
    if any(probe not in {"neutron", "xray"} for probe in requested_probes):
        raise ValueError("probes must contain only 'neutron' and 'xray'")

    partials = compute_partial_structure_factors(
        completed_rdfs,
        composition=resolved_composition,
        number_density=density,
        pairs=resolved_pairs,
        convention=convention,
        q_range=q_range,
        q_step=q_step,
        backend=backend,
        integration=integration,
        chunk_size=chunk_size,
    )
    if r_grid is None:
        r_output = r.copy()
    else:
        r_output = np.asarray(r_grid, dtype=np.float64)
        if r_output.ndim != 1 or len(r_output) < 2:
            raise ValueError("r_grid must be a one-dimensional grid with two points")
        if np.any(~np.isfinite(r_output)) or np.any(np.diff(r_output) <= 0):
            raise ValueError("r_grid must contain finite, strictly increasing values")

    results = {
        probe: _probe_result(
            completed_rdfs,
            r,
            partials,
            resolved_composition,
            resolved_pairs,
            probe,
            convention,
            density,
            r_output,
            xray_window,
            backend,
            integration,
            chunk_size,
        )
        for probe in requested_probes
    }
    metadata = {
        "number_density": density,
        "pairs": resolved_pairs,
        "q_range": (float(partials["Q"].iloc[0]), float(partials["Q"].iloc[-1])),
        "q_step": float(partials["Q"].iloc[1] - partials["Q"].iloc[0]),
        "real_space": {
            "neutron": {
                "method": "direct weighting of partial RDFs",
                "window": None,
            },
            "xray": {
                "method": "inverse transform of Q-dependent X-ray weighting",
                "window": xray_window,
            },
        },
        "transform_backend": backend,
        "integration": integration,
        "multiprocessing": False,
        "synthetic_ideal_rdf_pairs": completed_rdfs.attrs.get("system", {}).get(
            "synthetic_ideal_rdf_pairs", ()
        ),
    }
    return ScatteringResult(
        partial_rdfs=completed_rdfs,
        partial_structure_factors=partials,
        composition=resolved_composition,
        convention=convention,
        neutron=results.get("neutron"),
        xray=results.get("xray"),
        metadata=metadata,
    )


__all__ = [
    "CompositionInput",
    "ProbeScatteringResult",
    "ScatteringResult",
    "compute_partial_structure_factors",
    "compute_scattering",
]
