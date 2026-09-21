"""Reusable molecular-dynamics trajectory and structural-analysis tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.2.0"

if TYPE_CHECKING:
    from . import cache, clustering
    from .angles import AngleDefinition, compute_bond_angles
    from .coordination import (
        CoordinationDefinition,
        compute_coordination,
        compute_rad_coordination,
        summarize_coordination,
    )
    from .environments import RADEnvironmentResult, compute_rad_environments
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
    from .trajectory import (
        Trajectory,
        TrajectoryFrame,
        load_trajectory,
        normalize_xyz_species_order,
        xyz_has_variable_species_order,
    )


_LAZY_EXPORTS = {
    "AngleDefinition": ("molecular_dynamics_tools.angles", "AngleDefinition"),
    "compute_bond_angles": (
        "molecular_dynamics_tools.angles",
        "compute_bond_angles",
    ),
    "CoordinationDefinition": (
        "molecular_dynamics_tools.coordination",
        "CoordinationDefinition",
    ),
    "compute_coordination": (
        "molecular_dynamics_tools.coordination",
        "compute_coordination",
    ),
    "compute_rad_coordination": (
        "molecular_dynamics_tools.coordination",
        "compute_rad_coordination",
    ),
    "summarize_coordination": (
        "molecular_dynamics_tools.coordination",
        "summarize_coordination",
    ),
    "RADEnvironmentResult": (
        "molecular_dynamics_tools.environments",
        "RADEnvironmentResult",
    ),
    "compute_rad_environments": (
        "molecular_dynamics_tools.environments",
        "compute_rad_environments",
    ),
    "compute_rdfs": ("molecular_dynamics_tools.rdf", "compute_rdfs"),
    "compute_spectral_rdfs": ("molecular_dynamics_tools.rdf", "compute_spectral_rdfs"),
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
    "normalize_xyz_species_order": (
        "molecular_dynamics_tools.trajectory",
        "normalize_xyz_species_order",
    ),
    "xyz_has_variable_species_order": (
        "molecular_dynamics_tools.trajectory",
        "xyz_has_variable_species_order",
    ),
    "Trajectory": ("molecular_dynamics_tools.trajectory", "Trajectory"),
    "TrajectoryFrame": ("molecular_dynamics_tools.trajectory", "TrajectoryFrame"),
}

_LAZY_MODULES = {
    "cache": "molecular_dynamics_tools.cache",
    "clustering": "molecular_dynamics_tools.clustering",
}


def __getattr__(name: str) -> Any:
    """Load scientific dependencies only when their public API is requested."""

    module_name = _LAZY_MODULES.get(name)
    if module_name is not None:
        from importlib import import_module

        value = import_module(module_name)
        globals()[name] = value
        return value

    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc

    from importlib import import_module

    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = [
    "AngleDefinition",
    "CoordinationDefinition",
    "Trajectory",
    "TrajectoryFrame",
    "__version__",
    "ProbeScatteringResult",
    "RADEnvironmentResult",
    "ScatteringComposition",
    "ScatteringResult",
    "cache",
    "clustering",
    "compute_partial_structure_factors",
    "compute_bond_angles",
    "compute_coordination",
    "compute_rad_coordination",
    "compute_rad_environments",
    "compute_rdfs",
    "compute_scattering",
    "plot_structure_factor",
    "plot_weighted_rdf",
    "compute_scattering_weights",
    "compute_spectral_rdfs",
    "load_trajectory",
    "normalize_xyz_species_order",
    "xyz_has_variable_species_order",
    "summarize_coordination",
]
