"""Faber–Ziman and Ashcroft–Langreth scattering weights."""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

from .composition import ScatteringComposition


Pair = tuple[str, str]
ScatteringProbe = Literal["neutron", "xray"]
StructureFactorConvention = Literal["faber-ziman", "ashcroft-langreth"]


def pair_label(pair: Pair) -> str:
    return f"{pair[0]}-{pair[1]}"


def canonical_pair(pair: Pair, species_order: dict[str, int]) -> Pair:
    left, right = pair
    if left not in species_order or right not in species_order:
        unknown = left if left not in species_order else right
        raise ValueError(f"pair contains species {unknown!r} absent from composition")
    return pair if left <= right else (right, left)


def unique_pairs(species: Iterable[str]) -> tuple[Pair, ...]:
    names = tuple(sorted(species))
    return tuple(
        (names[left], names[right])
        for left in range(len(names))
        for right in range(left, len(names))
    )


def normalize_pairs(
    pairs: Iterable[Pair], composition: ScatteringComposition
) -> tuple[Pair, ...]:
    order = {species: index for index, species in enumerate(composition.species)}
    normalized: list[Pair] = []
    seen: set[Pair] = set()
    for raw_pair in pairs:
        if len(raw_pair) != 2:
            raise ValueError("each pair must contain exactly two species")
        pair = canonical_pair((str(raw_pair[0]), str(raw_pair[1])), order)
        if pair in seen:
            raise ValueError(f"duplicate pair {pair_label(pair)!r}")
        normalized.append(pair)
        seen.add(pair)
    if not normalized:
        raise ValueError("at least one atomic pair is required")
    return tuple(normalized)


def _scattering_factors(
    composition: ScatteringComposition,
    q: np.ndarray,
    probe: ScatteringProbe,
) -> dict[str, np.ndarray]:
    if probe == "neutron":
        return {
            species: np.full(len(q), length, dtype=np.float64)
            for species, length in composition.neutron_scattering_lengths().items()
        }
    if probe == "xray":
        table = composition.xray_form_factors(q)
        return {
            species: table[species].to_numpy(dtype=np.float64)
            for species in composition.species
        }
    raise ValueError("probe must be 'neutron' or 'xray'")


def compute_scattering_weights(
    composition: ScatteringComposition,
    pairs: Iterable[Pair],
    q: ArrayLike,
    *,
    probe: ScatteringProbe,
    convention: StructureFactorConvention = "faber-ziman",
) -> pd.DataFrame:
    """Calculate pair coefficients for a partial-structure-factor convention.

    Faber–Ziman weights multiply ``S_ij - 1``. Ashcroft–Langreth weights
    multiply ``S_ij - delta_ij``. Pair columns therefore always represent
    contributions to the total ``S(Q) - 1``.
    """

    if not isinstance(composition, ScatteringComposition):
        raise TypeError("composition must be a ScatteringComposition")
    q_values = np.asarray(q, dtype=np.float64)
    if q_values.ndim != 1 or len(q_values) < 2:
        raise ValueError("q must be a one-dimensional grid with at least two points")
    if np.any(~np.isfinite(q_values)) or np.any(q_values < 0):
        raise ValueError("q values must be finite and nonnegative")
    if np.any(np.diff(q_values) <= 0):
        raise ValueError("q must be strictly increasing")
    if convention not in {"faber-ziman", "ashcroft-langreth"}:
        raise ValueError("convention must be 'faber-ziman' or 'ashcroft-langreth'")

    normalized_pairs = normalize_pairs(pairs, composition)
    concentrations = composition.concentrations
    factors = _scattering_factors(composition, q_values, probe)
    factor_matrix = np.column_stack(
        [factors[species] for species in composition.species]
    )
    concentration_array = np.asarray(
        [concentrations[species] for species in composition.species]
    )

    if convention == "faber-ziman":
        mean_factor = factor_matrix @ concentration_array
        scale = np.sum(
            np.abs(factor_matrix) * concentration_array[np.newaxis, :], axis=1
        )
        unstable = np.abs(mean_factor) <= np.finfo(float).eps * np.maximum(scale, 1.0)
        if np.any(unstable):
            first = int(np.flatnonzero(unstable)[0])
            raise ValueError(
                f"Faber-Ziman normalization is singular at Q={q_values[first]:g}; "
                "the concentration-weighted mean scattering factor is zero"
            )
        denominator = mean_factor**2
    else:
        denominator = (factor_matrix**2) @ concentration_array
        if np.any(denominator <= 0):
            raise ValueError("Ashcroft-Langreth mean-square factor must be positive")

    values: dict[str, np.ndarray] = {}
    for left, right in normalized_pairs:
        multiplicity = 1.0 if left == right else 2.0
        if convention == "faber-ziman":
            numerator = (
                multiplicity
                * concentrations[left]
                * concentrations[right]
                * factors[left]
                * factors[right]
            )
        else:
            numerator = (
                multiplicity
                * np.sqrt(concentrations[left] * concentrations[right])
                * factors[left]
                * factors[right]
            )
        values[pair_label((left, right))] = numerator / denominator

    result = pd.DataFrame(values)
    result.insert(0, "Q", q_values)
    result.attrs["probe"] = probe
    result.attrs["convention"] = convention
    result.attrs["pair_role"] = "coefficient multiplying partial correlation"
    result.attrs["normalization"] = (
        "mean scattering factor squared"
        if convention == "faber-ziman"
        else "mean squared scattering factor"
    )
    return result


__all__ = [
    "Pair",
    "ScatteringProbe",
    "StructureFactorConvention",
    "compute_scattering_weights",
    "normalize_pairs",
    "pair_label",
    "unique_pairs",
]
