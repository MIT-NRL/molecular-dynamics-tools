"""Reusable molecular-dynamics trajectory and structural-analysis tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


__version__ = "0.1.0"

if TYPE_CHECKING:
    from .rdf import compute_rdfs, compute_spectral_rdfs
    from .scattering import (
        ProbeScatteringResult,
        ScatteringComposition,
        ScatteringResult,
        compute_partial_structure_factors,
        compute_scattering,
        compute_scattering_weights,
        plot_structure_factor,
        plot_weighted_rdf,
    )
    from .trajectory import Trajectory, TrajectoryFrame, load_trajectory


_LAZY_EXPORTS = {
    "compute_rdfs": ("molecular_dynamics_tools.rdf", "compute_rdfs"),
    "compute_spectral_rdfs": (
        "molecular_dynamics_tools.rdf", "compute_spectral_rdfs"
    ),
    "compute_partial_structure_factors": (
        "molecular_dynamics_tools.scattering",
        "compute_partial_structure_factors",
    ),
    "compute_scattering": (
        "molecular_dynamics_tools.scattering",
        "compute_scattering",
    ),
    "plot_structure_factor": (
        "molecular_dynamics_tools.scattering",
        "plot_structure_factor",
    ),
    "plot_weighted_rdf": (
        "molecular_dynamics_tools.scattering",
        "plot_weighted_rdf",
    ),
    "compute_scattering_weights": (
        "molecular_dynamics_tools.scattering",
        "compute_scattering_weights",
    ),
    "ProbeScatteringResult": (
        "molecular_dynamics_tools.scattering",
        "ProbeScatteringResult",
    ),
    "ScatteringComposition": (
        "molecular_dynamics_tools.scattering",
        "ScatteringComposition",
    ),
    "ScatteringResult": (
        "molecular_dynamics_tools.scattering",
        "ScatteringResult",
    ),
    "load_trajectory": ("molecular_dynamics_tools.trajectory", "load_trajectory"),
    "Trajectory": ("molecular_dynamics_tools.trajectory", "Trajectory"),
    "TrajectoryFrame": ("molecular_dynamics_tools.trajectory", "TrajectoryFrame"),
}


def __getattr__(name: str) -> Any:
    """Load scientific dependencies only when their public API is requested."""

    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc

    from importlib import import_module

    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = [
    "Trajectory",
    "TrajectoryFrame",
    "__version__",
    "ProbeScatteringResult",
    "ScatteringComposition",
    "ScatteringResult",
    "compute_partial_structure_factors",
    "compute_rdfs",
    "compute_scattering",
    "plot_structure_factor",
    "plot_weighted_rdf",
    "compute_scattering_weights",
    "compute_spectral_rdfs",
    "load_trajectory",
]
