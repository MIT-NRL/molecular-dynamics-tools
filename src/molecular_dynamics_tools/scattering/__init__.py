"""Neutron and X-ray weighting of molecular-dynamics partial RDFs."""

from .calculation import (
    CompositionInput,
    ProbeScatteringResult,
    ScatteringResult,
    compute_partial_structure_factors,
    compute_scattering,
)
from .composition import IsotopeMixture, ScatteringComposition
from .plotting import plot_structure_factor, plot_weighted_rdf
from .weights import (
    Pair,
    ScatteringProbe,
    StructureFactorConvention,
    compute_scattering_weights,
)

__all__ = [
    "CompositionInput",
    "IsotopeMixture",
    "Pair",
    "ProbeScatteringResult",
    "ScatteringComposition",
    "ScatteringProbe",
    "ScatteringResult",
    "StructureFactorConvention",
    "compute_partial_structure_factors",
    "compute_scattering",
    "plot_structure_factor",
    "plot_weighted_rdf",
    "compute_scattering_weights",
]
