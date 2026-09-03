"""Chemical composition and scattering-factor lookup."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import periodictable as pt
from numpy.typing import ArrayLike, NDArray
from scipy.constants import Avogadro

if TYPE_CHECKING:
    from molecular_dynamics_tools.trajectory import Trajectory


IsotopeMixture = Mapping[int | str, float]


def _element(symbol: str):
    try:
        return pt.elements.symbol(symbol)
    except ValueError as exc:
        raise ValueError(f"unsupported element symbol {symbol!r}") from exc


def _isotope_mass_number(label: int | str, expected_symbol: str) -> int:
    if isinstance(label, (int, np.integer)):
        return int(label)
    compact = str(label).replace("-", "").replace(" ", "")
    match = re.fullmatch(r"(?:(\d+)([A-Z][a-z]?)|([A-Z][a-z]?)(\d+))", compact)
    if match is None:
        raise ValueError(
            f"invalid isotope {label!r}; use a mass number, '7Li', or 'Li7'"
        )
    leading_mass, leading_symbol, trailing_symbol, trailing_mass = match.groups()
    symbol = leading_symbol if leading_symbol is not None else trailing_symbol
    mass = leading_mass if leading_mass is not None else trailing_mass
    if symbol != expected_symbol:
        raise ValueError(
            f"isotope {label!r} does not match element {expected_symbol!r}"
        )
    return int(mass)


@dataclass(frozen=True, slots=True)
class ScatteringComposition:
    """Composition, isotope mixtures, and ionic X-ray states.

    ``amounts`` are relative species counts and need not be normalized. Species
    names are the labels used in RDF pair columns. ``elements`` maps non-element
    trajectory labels onto chemical symbols when necessary.
    """

    amounts: Mapping[str, float]
    isotopes: Mapping[str, IsotopeMixture] = field(default_factory=dict)
    charges: Mapping[str, int] = field(default_factory=dict)
    elements: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        amounts = {str(key): float(value) for key, value in self.amounts.items()}
        if not amounts:
            raise ValueError("composition must contain at least one species")
        if any(not np.isfinite(value) or value <= 0 for value in amounts.values()):
            raise ValueError("composition amounts must be finite and positive")

        elements = {str(key): str(value) for key, value in self.elements.items()}
        unknown_elements = set(elements) - set(amounts)
        if unknown_elements:
            raise ValueError(
                "element mappings contain unknown species: "
                + ", ".join(sorted(unknown_elements))
            )
        for species in amounts:
            _element(elements.get(species, species))

        isotopes = {
            str(species): dict(mixture)
            for species, mixture in self.isotopes.items()
        }
        charges = {str(species): int(charge) for species, charge in self.charges.items()}
        for supplied, name in ((isotopes, "isotope"), (charges, "charge")):
            unknown = set(supplied) - set(amounts)
            if unknown:
                raise ValueError(
                    f"{name} settings contain unknown species: "
                    + ", ".join(sorted(unknown))
                )

        for species, mixture in isotopes.items():
            if not mixture:
                raise ValueError(f"isotope mixture for {species!r} is empty")
            fractions = np.asarray(list(mixture.values()), dtype=np.float64)
            if np.any(~np.isfinite(fractions)) or np.any(fractions < 0):
                raise ValueError("isotope fractions must be finite and nonnegative")
            if not np.isclose(fractions.sum(), 1.0, rtol=0.0, atol=1e-8):
                raise ValueError(f"isotope fractions for {species!r} must sum to one")
            symbol = elements.get(species, species)
            for isotope in mixture:
                mass = _isotope_mass_number(isotope, symbol)
                try:
                    pt.elements.symbol(symbol)[mass]
                except (KeyError, ValueError) as exc:
                    raise ValueError(f"unknown isotope {mass}{symbol}") from exc

        object.__setattr__(self, "amounts", amounts)
        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "isotopes", isotopes)
        object.__setattr__(self, "charges", charges)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        *,
        isotopes: Mapping[str, IsotopeMixture] | None = None,
        charges: Mapping[str, int] | None = None,
    ) -> ScatteringComposition:
        """Build a composition using periodictable's chemical formula parser."""

        try:
            atoms = pt.formula(formula).atoms
        except Exception as exc:
            raise ValueError(f"unable to parse composition formula {formula!r}") from exc
        amounts: dict[str, float] = {}
        for atom, amount in atoms.items():
            symbol = atom.symbol
            amounts[symbol] = amounts.get(symbol, 0.0) + float(amount)
        return cls(amounts, isotopes=isotopes or {}, charges=charges or {})

    @classmethod
    def from_trajectory(
        cls,
        trajectory: Trajectory,
        *,
        isotopes: Mapping[str, IsotopeMixture] | None = None,
        charges: Mapping[str, int] | None = None,
        elements: Mapping[str, str] | None = None,
    ) -> ScatteringComposition:
        """Build a composition from the fixed species counts in a trajectory."""

        return cls(
            trajectory.atom_counts,
            isotopes=isotopes or {},
            charges=charges or {},
            elements=elements or {},
        )

    @property
    def species(self) -> tuple[str, ...]:
        return tuple(sorted(self.amounts))

    @property
    def concentrations(self) -> dict[str, float]:
        total = sum(self.amounts.values())
        return {species: amount / total for species, amount in self.amounts.items()}

    def element_symbol(self, species: str) -> str:
        if species not in self.amounts:
            raise KeyError(f"species {species!r} is not present in the composition")
        return self.elements.get(species, species)

    def neutron_scattering_lengths(self) -> dict[str, float]:
        """Return real coherent neutron scattering lengths in femtometres."""

        lengths: dict[str, float] = {}
        for species in self.species:
            symbol = self.element_symbol(species)
            element = _element(symbol)
            if species in self.isotopes:
                value = 0.0
                for isotope, fraction in self.isotopes[species].items():
                    mass = _isotope_mass_number(isotope, symbol)
                    neutron = element[mass].neutron
                    if neutron is None or neutron.b_c is None:
                        raise ValueError(
                            f"no coherent neutron length is available for {mass}{symbol}"
                        )
                    value += float(fraction) * float(np.real(neutron.b_c))
            else:
                neutron = element.neutron
                if neutron is None or neutron.b_c is None:
                    raise ValueError(
                        f"no coherent neutron length is available for {symbol}"
                    )
                value = float(np.real(neutron.b_c))
            lengths[species] = value
        return lengths

    def xray_form_factors(self, q: ArrayLike) -> pd.DataFrame:
        """Return neutral or ionic Cromer–Mann ``f0(Q)`` form factors."""

        q_values = np.asarray(q, dtype=np.float64)
        if q_values.ndim != 1 or not len(q_values):
            raise ValueError("q must be a nonempty one-dimensional grid")
        if np.any(~np.isfinite(q_values)) or np.any(q_values < 0):
            raise ValueError("q values must be finite and nonnegative")
        factors: dict[str, NDArray[np.float64]] = {}
        for species in self.species:
            element = _element(self.element_symbol(species))
            charge = self.charges.get(species, 0)
            try:
                scatterer = element if charge == 0 else element.ion[charge]
                values = np.asarray(scatterer.xray.f0(q_values), dtype=np.float64)
            except Exception as exc:
                label = f"{element.symbol}{charge:+d}" if charge else element.symbol
                raise ValueError(f"unable to calculate X-ray form factors for {label}") from exc
            if np.any(~np.isfinite(values)):
                first = int(np.flatnonzero(~np.isfinite(values))[0])
                raise ValueError(
                    f"X-ray form factor for {species!r} is unavailable at "
                    f"Q={q_values[first]:g} 1/angstrom"
                )
            factors[species] = values
        return pd.DataFrame(factors, index=pd.Index(q_values, name="Q"))

    def atomic_masses(self) -> dict[str, float]:
        masses: dict[str, float] = {}
        for species in self.species:
            symbol = self.element_symbol(species)
            element = _element(symbol)
            if species in self.isotopes:
                masses[species] = sum(
                    float(fraction)
                    * float(element[_isotope_mass_number(isotope, symbol)].mass)
                    for isotope, fraction in self.isotopes[species].items()
                )
            else:
                masses[species] = float(element.mass)
        return masses

    def number_density(self, mass_density: float) -> float:
        """Convert mass density in g/cm^3 to total atoms/angstrom^3."""

        density = float(mass_density)
        if not np.isfinite(density) or density <= 0:
            raise ValueError("mass_density must be finite and positive")
        concentrations = self.concentrations
        mean_atomic_mass = sum(
            concentrations[name] * mass
            for name, mass in self.atomic_masses().items()
        )
        return float(Avogadro * density / mean_atomic_mass / 1e24)

    def table(self) -> pd.DataFrame:
        """Return a human-readable summary of the composition."""

        concentrations = self.concentrations
        masses = self.atomic_masses()
        lengths = self.neutron_scattering_lengths()
        return pd.DataFrame(
            {
                "element": [self.element_symbol(name) for name in self.species],
                "amount": [self.amounts[name] for name in self.species],
                "concentration": [concentrations[name] for name in self.species],
                "atomic_mass": [masses[name] for name in self.species],
                "neutron_b_fm": [lengths[name] for name in self.species],
                "xray_charge": [self.charges.get(name, 0) for name in self.species],
            },
            index=pd.Index(self.species, name="species"),
        )


__all__ = ["IsotopeMixture", "ScatteringComposition"]
