"""Species-resolved relative-angular-distance (RAD) environments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._execution import Backend, execute_frame_chunks, plan_execution
from ._geometry import freud_box
from ._rad import (
    RADBondMode,
    normalize_rad_bond_mode,
    normalize_rad_variant,
    rad_neighbors_with_distances,
)
from .trajectory import Trajectory

_CONTACT_COLUMNS = (
    "frame",
    "source_frame",
    "center_index",
    "center_species",
    "neighbor_index",
    "neighbor_species",
    "distance",
    "is_mutual",
)
_ENVIRONMENT_COLUMNS = (
    "frame",
    "source_frame",
    "center_index",
    "center_species",
    "neighbor_species",
    "coordination",
    "mean_distance",
    "min_distance",
    "max_distance",
)


def _species_tuple(
    value: str | Sequence[str] | None,
    *,
    available: Sequence[str],
    name: str,
) -> tuple[str, ...]:
    if value is None:
        resolved = tuple(str(species) for species in available)
    elif isinstance(value, str):
        resolved = (value,)
    else:
        resolved = tuple(str(species) for species in value)
    if not resolved:
        raise ValueError(f"{name} must contain at least one species")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} must not contain duplicate species")
    missing = sorted(set(resolved) - set(available))
    if missing:
        raise ValueError(f"unknown species in {name}: {missing}")
    return resolved


@dataclass(slots=True)
class RADEnvironmentResult:
    """Per-contact and per-center results from :func:`compute_rad_environments`.

    Atom-index occupancy and lifetime summaries assume that a trajectory index
    identifies the same physical atom in every analyzed frame. This is normal
    for MD trajectories, but cannot be proven from coordinate-only files.
    """

    contacts: pd.DataFrame
    environments: pd.DataFrame
    metadata: dict[str, Any]

    def contacts_between(
        self,
        center_species: str | None = None,
        neighbor_species: str | None = None,
        *,
        bond_mode: str = "directed",
    ) -> pd.DataFrame:
        """Return stored contacts filtered by species and reciprocity."""

        mode = normalize_rad_bond_mode(bond_mode)
        if self.metadata["bond_mode"] == "mutual" and mode == "directed":
            raise ValueError(
                "directed contacts were not retained by a bond_mode='mutual' "
                "calculation; recompute with bond_mode='directed'"
            )
        result = self.contacts
        mask = np.ones(len(result), dtype=bool)
        if center_species is not None:
            mask &= result["center_species"].to_numpy() == str(center_species)
        if neighbor_species is not None:
            mask &= result["neighbor_species"].to_numpy() == str(neighbor_species)
        if mode == "mutual":
            mask &= result["is_mutual"].to_numpy(dtype=bool)
        return result.loc[mask].reset_index(drop=True)

    def speciation(self, species: str | Sequence[str] | None = None) -> pd.DataFrame:
        """Return one species-count signature per center atom and frame."""

        requested = _species_tuple(
            species,
            available=self.metadata["neighbor_species"],
            name="species",
        )
        index = ["frame", "source_frame", "center_index", "center_species"]
        selected = self.environments[
            self.environments["neighbor_species"].isin(requested)
        ]
        table = selected.pivot(index=index, columns="neighbor_species", values="coordination")
        table = table.reindex(columns=requested, fill_value=0).reset_index()
        table.columns.name = None
        for name in requested:
            table[name] = table[name].fillna(0).astype(np.int64)
        table["label"] = [
            str(center) + "".join(f"{name}{int(row[name])}" for name in requested)
            for center, (_, row) in zip(
                table["center_species"], table.iterrows(), strict=True
            )
        ]
        return table

    def speciation_distribution(
        self, species: str | Sequence[str] | None = None
    ) -> pd.DataFrame:
        """Return the probability of each center-environment signature."""

        table = self.speciation(species)
        counts = (
            table.groupby(["center_species", "label"], sort=False)
            .size()
            .rename("count")
            .reset_index()
        )
        totals = counts.groupby("center_species")["count"].transform("sum")
        counts["probability"] = counts["count"] / totals
        return counts

    def occupancy(
        self,
        center_species: str | None = None,
        neighbor_species: str | None = None,
        *,
        bond_mode: str = "directed",
        assume_stable_atom_identity: bool = False,
    ) -> pd.DataFrame:
        """Return observed-sample occupancy for each atom-index pair.

        Set ``assume_stable_atom_identity=True`` to acknowledge that trajectory
        atom indices identify the same physical atoms in every analyzed frame.
        """

        if not assume_stable_atom_identity:
            raise ValueError(
                "occupancy requires stable atom indices across frames; pass "
                "assume_stable_atom_identity=True after verifying the trajectory"
            )

        columns = [
            "center_index",
            "center_species",
            "neighbor_index",
            "neighbor_species",
        ]
        contacts = self.contacts_between(
            center_species, neighbor_species, bond_mode=bond_mode
        )
        atom_species = tuple(str(value) for value in self.metadata["atom_species"])
        center_labels = (
            (str(center_species),)
            if center_species is not None
            else tuple(self.metadata["center_species"])
        )
        neighbor_labels = (
            (str(neighbor_species),)
            if neighbor_species is not None
            else tuple(self.metadata["neighbor_species"])
        )
        center_indices = [
            index for index, label in enumerate(atom_species) if label in center_labels
        ]
        neighbor_indices = [
            index for index, label in enumerate(atom_species) if label in neighbor_labels
        ]
        eligible = pd.DataFrame.from_records(
            [
                (center, atom_species[center], neighbor, atom_species[neighbor])
                for center in center_indices
                for neighbor in neighbor_indices
                if center != neighbor
            ],
            columns=columns,
        )
        if eligible.empty:
            return pd.DataFrame(columns=[*columns, "observations", "occupancy"])
        observed = (
            contacts.groupby(columns, sort=False)["frame"]
            .nunique()
            .rename("observations")
            .reset_index()
        )
        result = eligible.merge(observed, on=columns, how="left", sort=False)
        result["observations"] = result["observations"].fillna(0).astype(np.int64)
        result["occupancy"] = result["observations"] / int(self.metadata["frame_count"])
        return result

    def lifetimes(
        self,
        center_species: str | None = None,
        neighbor_species: str | None = None,
        *,
        bond_mode: str = "directed",
        assume_stable_atom_identity: bool = False,
    ) -> pd.DataFrame:
        """Return contiguous contact runs in analyzed-frame samples.

        Runs are measured in adjacent selected samples, not elapsed simulation
        time. Set ``assume_stable_atom_identity=True`` after verifying that atom
        indices retain physical identity across frames.
        """

        if not assume_stable_atom_identity:
            raise ValueError(
                "lifetimes require stable atom indices across frames; pass "
                "assume_stable_atom_identity=True after verifying the trajectory"
            )

        contacts = self.contacts_between(
            center_species, neighbor_species, bond_mode=bond_mode
        )
        columns = [
            "center_index",
            "center_species",
            "neighbor_index",
            "neighbor_species",
            "start_frame",
            "end_frame",
            "start_source_frame",
            "end_source_frame",
            "lifetime_samples",
            "logical_frame_span",
            "source_frame_span",
        ]
        if contacts.empty:
            return pd.DataFrame(columns=columns)

        frame_order = {
            int(frame): order for order, frame in enumerate(self.metadata["frames"])
        }
        keys = [
            "center_index",
            "center_species",
            "neighbor_index",
            "neighbor_species",
        ]
        rows: list[dict[str, Any]] = []
        for key, group in contacts.groupby(keys, sort=False):
            ordered = group.assign(
                _order=group["frame"].map(frame_order).astype(np.int64)
            ).sort_values("_order")
            orders = ordered["_order"].to_numpy(dtype=np.int64)
            starts = np.r_[0, np.flatnonzero(np.diff(orders) != 1) + 1]
            stops = np.r_[starts[1:], len(ordered)]
            for start, stop in zip(starts, stops, strict=True):
                run = ordered.iloc[int(start) : int(stop)]
                first = run.iloc[0]
                last = run.iloc[-1]
                rows.append(
                    {
                        **dict(zip(keys, key, strict=True)),
                        "start_frame": int(first["frame"]),
                        "end_frame": int(last["frame"]),
                        "start_source_frame": int(first["source_frame"]),
                        "end_source_frame": int(last["source_frame"]),
                        "lifetime_samples": len(run),
                        "logical_frame_span": int(last["frame"])
                        - int(first["frame"])
                        + 1,
                        "source_frame_span": int(last["source_frame"])
                        - int(first["source_frame"])
                        + 1,
                    }
                )
        return pd.DataFrame(rows, columns=columns)


ContactRow = tuple[int, int, int, str, int, str, float, bool]
EnvironmentRow = tuple[int, int, int, str, str, int, float, float, float]


def _rad_environment_chunk(
    trajectory: Trajectory,
    frame_indices: tuple[int, ...],
    center_species: tuple[str, ...],
    neighbor_species: tuple[str, ...],
    bond_mode: RADBondMode,
    variant: str,
) -> tuple[list[ContactRow], list[EnvironmentRow]]:
    contact_rows: list[ContactRow] = []
    environment_rows: list[EnvironmentRow] = []
    allowed_neighbors = set(neighbor_species)

    for frame in trajectory.iter_frames(frame_indices):
        box = freud_box(frame.dimensions)
        shell_cache: dict[int, tuple[NDArray[np.int64], NDArray[np.float64]]] = {}
        shell_sets: dict[int, set[int]] = {}

        def shell(index: int) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
            if index not in shell_cache:
                shell_cache[index] = rad_neighbors_with_distances(
                    box, frame.positions, index, variant=variant
                )
            return shell_cache[index]

        def shell_set(index: int) -> set[int]:
            if index not in shell_sets:
                shell_sets[index] = set(shell(index)[0].tolist())
            return shell_sets[index]

        centers = np.flatnonzero(np.isin(frame.species, center_species))
        for center_raw in centers:
            center = int(center_raw)
            center_label = str(frame.species[center])
            neighbors, distances = shell(center)
            grouped: dict[str, list[float]] = {name: [] for name in neighbor_species}
            for neighbor_raw, distance_raw in zip(neighbors, distances, strict=True):
                neighbor = int(neighbor_raw)
                neighbor_label = str(frame.species[neighbor])
                if neighbor_label not in allowed_neighbors:
                    continue
                is_mutual = center in shell_set(neighbor)
                if bond_mode == "mutual" and not is_mutual:
                    continue
                distance = float(distance_raw)
                grouped[neighbor_label].append(distance)
                contact_rows.append(
                    (
                        frame.index,
                        frame.source_index,
                        center,
                        center_label,
                        neighbor,
                        neighbor_label,
                        distance,
                        is_mutual,
                    )
                )
            for neighbor_label in neighbor_species:
                values = grouped[neighbor_label]
                environment_rows.append(
                    (
                        frame.index,
                        frame.source_index,
                        center,
                        center_label,
                        neighbor_label,
                        len(values),
                        float(np.mean(values)) if values else float("nan"),
                        float(np.min(values)) if values else float("nan"),
                        float(np.max(values)) if values else float("nan"),
                    )
                )
    return contact_rows, environment_rows


def _contact_table(
    rows: list[ContactRow], frame_indices: Sequence[int]
) -> pd.DataFrame:
    result = pd.DataFrame.from_records(rows, columns=_CONTACT_COLUMNS)
    result = result.astype(
        {
            "frame": "int64",
            "source_frame": "int64",
            "center_index": "int64",
            "center_species": "string",
            "neighbor_index": "int64",
            "neighbor_species": "string",
            "distance": "float64",
            "is_mutual": "bool",
        }
    )
    order = {int(frame): index for index, frame in enumerate(frame_indices)}
    result["_frame_order"] = result["frame"].map(order).astype(np.int64)
    return (
        result.sort_values(
            ["_frame_order", "center_index", "neighbor_index"], kind="stable"
        )
        .drop(columns="_frame_order")
        .reset_index(drop=True)
    )


def _environment_table(
    rows: list[EnvironmentRow],
    frame_indices: Sequence[int],
    neighbor_species: Sequence[str],
) -> pd.DataFrame:
    result = pd.DataFrame.from_records(rows, columns=_ENVIRONMENT_COLUMNS)
    result = result.astype(
        {
            "frame": "int64",
            "source_frame": "int64",
            "center_index": "int64",
            "center_species": "string",
            "neighbor_species": "string",
            "coordination": "int64",
            "mean_distance": "float64",
            "min_distance": "float64",
            "max_distance": "float64",
        }
    )
    frame_order = {int(frame): index for index, frame in enumerate(frame_indices)}
    species_order = {str(species): index for index, species in enumerate(neighbor_species)}
    result["_frame_order"] = result["frame"].map(frame_order).astype(np.int64)
    result["_species_order"] = (
        result["neighbor_species"].map(species_order).astype(np.int64)
    )
    return (
        result.sort_values(
            ["_frame_order", "center_index", "_species_order"], kind="stable"
        )
        .drop(columns=["_frame_order", "_species_order"])
        .reset_index(drop=True)
    )


def compute_rad_environments(
    trajectory: Trajectory,
    center_species: str | Sequence[str],
    *,
    neighbor_species: str | Sequence[str] | None = None,
    bond_mode: str = "directed",
    variant: str = "closed",
    frames: slice | Sequence[int] | None = None,
    ncore: int | None = 1,
    backend: Backend = "auto",
    show_progress: bool = False,
) -> RADEnvironmentResult:
    """Compute species-resolved RAD shells and periodic neighbor distances.

    ``bond_mode='directed'`` retains every neighbor in each selected center's
    RAD-closed shell and records reciprocity in ``contacts.is_mutual``.
    ``bond_mode='mutual'`` retains only reciprocal contacts. Omitting
    ``neighbor_species`` records every trajectory species. Coordinates are
    streamed through the shared execution backend and are never precached.

    Persistent-contact helpers assume stable atom identities across frames.
    """

    centers = _species_tuple(
        center_species, available=trajectory.species, name="center_species"
    )
    neighbors = _species_tuple(
        neighbor_species, available=trajectory.species, name="neighbor_species"
    )
    resolved_mode = normalize_rad_bond_mode(bond_mode)
    resolved_variant = normalize_rad_variant(variant)
    frame_indices = trajectory.resolve_frame_indices(frames)
    if not frame_indices:
        raise ValueError("frames must select at least one trajectory frame")
    source_frame_indices = tuple(
        trajectory.source.frames[index].source_index for index in frame_indices
    )
    first_frame = trajectory.frame(frame_indices[0])
    center_count = sum(trajectory.atom_counts[species] for species in centers)
    plan = plan_execution(
        len(frame_indices),
        ncore=ncore,
        backend=backend,
        workload=len(frame_indices) * trajectory.n_atoms * center_count,
        auto_multiprocessing_threshold=15_000_000,
    )
    partials, execution = execute_frame_chunks(
        trajectory,
        frame_indices,
        plan,
        _rad_environment_chunk,
        (centers, neighbors, resolved_mode, resolved_variant),
        show_progress=show_progress,
        description=f"RAD environments ({resolved_mode})",
    )
    contact_rows = [row for contacts, _ in partials for row in contacts]
    environment_rows = [row for _, environments in partials for row in environments]
    metadata: dict[str, Any] = {
        "method": "relative_angular_distance",
        "variant": resolved_variant,
        "bond_mode": resolved_mode,
        "center_species": centers,
        "neighbor_species": neighbors,
        "frames": frame_indices,
        "source_frames": source_frame_indices,
        "frame_count": len(frame_indices),
        "atom_species": tuple(str(value) for value in first_frame.species),
        "frame_selection_consecutive": all(
            right == left + 1
            for left, right in zip(source_frame_indices, source_frame_indices[1:])
        ),
        "stable_atom_identity_required": True,
        "atom_identity_verified": False,
        "execution": execution,
    }
    return RADEnvironmentResult(
        contacts=_contact_table(contact_rows, frame_indices),
        environments=_environment_table(environment_rows, frame_indices, neighbors),
        metadata=metadata,
    )


__all__ = ["RADEnvironmentResult", "compute_rad_environments"]
