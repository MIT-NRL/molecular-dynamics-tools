"""MDAnalysis-backed trajectory loading and package metadata handling."""

from __future__ import annotations

import hashlib
import re
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.XYZ import XYZReader
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.lib import util
from numpy.typing import ArrayLike, NDArray

_LATTICE_RE = re.compile(r'Lattice="([^"]+)"')
_ORIGIN_RE = re.compile(r'Origin="([^"]+)"')


@dataclass(frozen=True, slots=True)
class FrameLocation:
    source_index: int
    dimensions: tuple[float, float, float, float, float, float] | None
    origin: tuple[float, float, float] | None


@dataclass(frozen=True, slots=True)
class TrajectorySource:
    frames: tuple[FrameLocation, ...]
    topology: str | None
    coordinates: tuple[str, ...]
    format: str | None
    atom_attribute: str
    shift_by_origin: bool
    frame_species_codes: NDArray[np.uint8] | None = None
    species_labels: tuple[str, ...] = ()


@dataclass(slots=True)
class TrajectoryFrame:
    index: int
    source_index: int
    species: NDArray[np.str_]
    positions: NDArray[np.float32]
    dimensions: NDArray[np.float64]
    origin: NDArray[np.float64] | None

    def positions_of(self, species: str) -> NDArray[np.float32]:
        return self.positions[self.species == species]


@dataclass(slots=True)
class Trajectory:
    """Common analysis input backed by a public MDAnalysis Universe."""

    universe: mda.Universe
    source: TrajectorySource
    _atom_labels: NDArray[np.str_]

    def __len__(self) -> int:
        return len(self.source.frames)

    def __enter__(self) -> Trajectory:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying MDAnalysis trajectory reader."""

        self.universe.trajectory.close()

    @property
    def n_atoms(self) -> int:
        return int(self.universe.atoms.n_atoms)

    @property
    def species(self) -> tuple[str, ...]:
        return tuple(sorted(str(value) for value in np.unique(self._atom_labels)))

    @property
    def atom_counts(self) -> dict[str, int]:
        values, counts = np.unique(self._atom_labels, return_counts=True)
        return {str(value): int(count) for value, count in zip(values, counts)}

    @property
    def filename(self) -> Path | None:
        if len(self.source.coordinates) != 1:
            return None
        return Path(self.source.coordinates[0])

    @property
    def dimensions(self) -> NDArray[np.float64]:
        return self.frame(0).dimensions

    def select_atoms(self, selection: str, *others: str, **kwargs: Any):
        return self.universe.select_atoms(selection, *others, **kwargs)

    def resolve_frame_indices(
        self, frames: slice | Sequence[int] | None = None
    ) -> tuple[int, ...]:
        if frames is None:
            return tuple(range(len(self)))
        if isinstance(frames, slice):
            return tuple(range(len(self))[frames])
        resolved: list[int] = []
        for raw_index in frames:
            index = int(raw_index)
            if index < 0:
                index += len(self)
            if index < 0 or index >= len(self):
                raise IndexError(
                    f"frame {raw_index} is outside the trajectory range 0 to {len(self) - 1}"
                )
            resolved.append(index)
        if len(set(resolved)) != len(resolved):
            raise ValueError("frames must not contain duplicate indices")
        return tuple(resolved)

    def _materialize(self, logical_index: int, timestep: Any) -> TrajectoryFrame:
        location = self.source.frames[logical_index]
        dimensions = location.dimensions
        if dimensions is None and timestep.dimensions is not None:
            dimensions = _normalize_box(timestep.dimensions)
        if dimensions is None:
            raise ValueError(
                f"frame {location.source_index} has no periodic cell; pass box=... "
                "or use a trajectory format that stores dimensions"
            )
        timestep.dimensions = np.asarray(dimensions, dtype=np.float32)
        positions = np.asarray(timestep.positions, dtype=np.float32).copy()
        origin = (
            None if location.origin is None else np.asarray(location.origin, dtype=np.float64)
        )
        if self.source.shift_by_origin and origin is not None:
            positions -= origin.astype(np.float32)
        labels = self._atom_labels
        if self.source.frame_species_codes is not None:
            codes = self.source.frame_species_codes[logical_index]
            labels = np.asarray(self.source.species_labels, dtype=str)[codes]
        return TrajectoryFrame(
            logical_index,
            location.source_index,
            labels,
            positions,
            np.asarray(dimensions, dtype=np.float64),
            origin,
        )

    def frame(self, index: int) -> TrajectoryFrame:
        logical_index = self.resolve_frame_indices((index,))[0]
        source_index = self.source.frames[logical_index].source_index
        return self._materialize(logical_index, self.universe.trajectory[source_index])

    def iter_frames(
        self, frames: slice | Sequence[int] | None = None
    ) -> Iterator[TrajectoryFrame]:
        """Stream frames through MDAnalysis without retaining coordinates."""

        logical_indices = self.resolve_frame_indices(frames)
        if not logical_indices:
            return
        source_indices = tuple(
            self.source.frames[index].source_index for index in logical_indices
        )
        increasing = all(
            left < right for left, right in zip(source_indices, source_indices[1:])
        )
        if increasing:
            requested = dict(zip(source_indices, logical_indices))
            final_source_index = source_indices[-1]
            for timestep in self.universe.trajectory:
                source_index = int(timestep.frame)
                if source_index in requested:
                    yield self._materialize(requested[source_index], timestep)
                if source_index >= final_source_index:
                    break
            return
        for logical_index, source_index in zip(logical_indices, source_indices):
            yield self._materialize(logical_index, self.universe.trajectory[source_index])

    def prepare_for_multiprocessing(self) -> None:
        """Build random-access state once before spawned workers start."""

        frame_count = len(self.universe.trajectory)
        highest = max(location.source_index for location in self.source.frames)
        if highest >= frame_count:
            raise IndexError(
                f"source frame {highest} does not exist; trajectory has {frame_count} frames"
            )


def _normalize_box(box: ArrayLike) -> tuple[float, float, float, float, float, float]:
    values = np.asarray(box, dtype=float)
    if values.ndim == 0:
        values = np.repeat(values, 3)
    values = values.ravel()
    if len(values) == 3:
        values = np.concatenate((values, np.full(3, 90.0)))
    if len(values) != 6:
        raise ValueError("box must be a scalar, a 3-vector, or a 6-vector")
    if not np.all(np.isfinite(values)) or np.any(values[:3] <= 0):
        raise ValueError("box lengths must be finite and positive")
    return tuple(float(value) for value in values)


def _angle_degrees(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    cosine = np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _header_box(
    header: str,
    override: tuple[float, float, float, float, float, float] | None,
) -> tuple[float, float, float, float, float, float] | None:
    if override is not None:
        return override
    match = _LATTICE_RE.search(header)
    if match is None:
        return None
    values = np.asarray([float(value) for value in match.group(1).split()])
    if len(values) == 3:
        return _normalize_box(values)
    if len(values) != 9:
        raise ValueError("Lattice must contain either 3 or 9 numbers")
    a_vector, b_vector, c_vector = values.reshape(3, 3)
    return _normalize_box(
        (
            np.linalg.norm(a_vector),
            np.linalg.norm(b_vector),
            np.linalg.norm(c_vector),
            _angle_degrees(b_vector, c_vector),
            _angle_degrees(a_vector, c_vector),
            _angle_degrees(a_vector, b_vector),
        )
    )


def _header_origin(header: str) -> tuple[float, float, float] | None:
    match = _ORIGIN_RE.search(header)
    if match is None:
        return None
    values = tuple(float(value) for value in match.group(1).split())
    if len(values) != 3:
        raise ValueError("Origin must contain exactly 3 numbers")
    return values


def _source_frame_indices(reader: Any, frames: slice | Sequence[int] | None) -> tuple[int, ...]:
    if frames is None:
        selected = tuple(range(len(reader)))
    elif isinstance(frames, slice):
        start = 0 if frames.start is None else int(frames.start)
        stop = frames.stop
        step = 1 if frames.step is None else int(frames.step)
        if start < 0 or (stop is not None and int(stop) < 0) or step < 1:
            raise ValueError("source frame slices require nonnegative bounds and step > 0")
        selected = tuple(range(start, len(reader) if stop is None else int(stop), step))
    else:
        selected = tuple(int(index) for index in frames)
        if any(index < 0 for index in selected):
            raise ValueError("source frame indices must be nonnegative")
    if not selected:
        raise ValueError("frames must select at least one source frame")
    if len(set(selected)) != len(selected):
        raise ValueError("frames must not contain duplicate indices")
    return selected


def _read_xyz_headers_sequentially(
    reader: XYZReader, source_indices: Sequence[int]
) -> dict[int, str]:
    requested = set(source_indices)
    headers: dict[int, str] = {}
    with util.anyopen(reader.filename) as handle:
        for source_index in range(max(requested) + 1):
            count_line = handle.readline()
            if not count_line:
                break
            try:
                atom_count = int(count_line)
            except ValueError as exc:
                raise ValueError(f"invalid atom count at XYZ frame {source_index}") from exc
            header = handle.readline().rstrip("\r\n")
            if source_index in requested:
                headers[source_index] = header
            for _ in range(atom_count):
                if not handle.readline():
                    raise ValueError(f"incomplete XYZ atom block at frame {source_index}")
    missing = requested.difference(headers)
    if missing:
        raise IndexError(f"source frames do not exist: {sorted(missing)}")
    return headers


def _read_xyz_headers(reader: XYZReader, source_indices: Sequence[int]) -> dict[int, str]:
    offsets = getattr(reader, "_offsets", None)
    if offsets is None:
        return _read_xyz_headers_sequentially(reader, source_indices)
    headers: dict[int, str] = {}
    with util.anyopen(reader.filename) as handle:
        for source_index in source_indices:
            try:
                handle.seek(offsets[source_index])
            except IndexError as exc:
                raise IndexError(f"source frame {source_index} does not exist") from exc
            handle.readline()
            headers[source_index] = handle.readline().rstrip("\r\n")
    return headers


def _read_xyz_species_codes(
    reader: XYZReader,
    source_indices: Sequence[int],
) -> tuple[NDArray[np.uint8] | None, tuple[str, ...]]:
    """Return compact per-frame labels when XYZ row order changes.

    Labels are streamed directly into a compact code matrix.  This avoids
    retaining a Python string object for every atom in every selected frame.
    """
    offsets = getattr(reader, "_offsets", None)
    row_for_source = {source_index: row for row, source_index in enumerate(source_indices)}
    codes: NDArray[np.uint8] | None = None
    reference_codes: NDArray[np.uint8] | None = None
    species_labels: tuple[str, ...] = ()
    lookup: dict[str, int] = {}
    dynamic_order = False

    def read_labels(handle, source_index: int) -> list[str]:
        try:
            atom_count = int(handle.readline())
        except ValueError as exc:
            raise ValueError(f"invalid atom count at XYZ frame {source_index}") from exc
        handle.readline()
        labels: list[str] = []
        for _ in range(atom_count):
            fields = handle.readline().split(maxsplit=1)
            if not fields:
                raise ValueError(f"incomplete XYZ atom block at frame {source_index}")
            labels.append(fields[0])
        return labels

    def store_labels(labels: list[str], row: int) -> None:
        nonlocal codes, reference_codes, species_labels, lookup, dynamic_order
        if codes is None:
            species_labels = tuple(sorted(set(labels)))
            if len(species_labels) > np.iinfo(np.uint8).max + 1:
                raise ValueError("XYZ trajectory contains too many species for compact labels")
            lookup = {label: index for index, label in enumerate(species_labels)}
            codes = np.empty((len(source_indices), len(labels)), dtype=np.uint8)
        if len(labels) != codes.shape[1] or set(labels) != set(species_labels):
            raise ValueError("XYZ frame species must be consistent across the selected frames")
        row_codes = np.fromiter((lookup[label] for label in labels), dtype=np.uint8, count=len(labels))
        codes[row] = row_codes
        if reference_codes is None:
            reference_codes = row_codes.copy()
        elif not np.array_equal(row_codes, reference_codes):
            dynamic_order = True

    with util.anyopen(reader.filename) as handle:
        if offsets is None:
            requested = set(source_indices)
            for source_index in range(max(requested) + 1):
                if source_index in requested:
                    store_labels(read_labels(handle, source_index), row_for_source[source_index])
                else:
                    try:
                        atom_count = int(handle.readline())
                    except ValueError as exc:
                        raise ValueError(f"invalid atom count at XYZ frame {source_index}") from exc
                    handle.readline()
                    for _ in range(atom_count):
                        if not handle.readline():
                            raise ValueError(f"incomplete XYZ atom block at frame {source_index}")
        else:
            for source_index in source_indices:
                try:
                    handle.seek(offsets[source_index])
                except IndexError as exc:
                    raise IndexError(f"source frame {source_index} does not exist") from exc
                store_labels(read_labels(handle, source_index), row_for_source[source_index])
    if codes is None:
        raise ValueError("XYZ trajectory did not provide any selected frames")
    if not dynamic_order:
        return None, ()
    codes.setflags(write=False)
    return codes, species_labels


def _xyz_species_sequence_for_frame(
    filename: str | Path,
    frame_index: int,
    atom_count: int,
) -> list[str] | None:
    """Read the ordered species labels for one XYZ frame."""

    frame_start = frame_index * (atom_count + 2) + 2
    labels: list[str] = []
    with Path(filename).open("r", encoding="utf-8", errors="replace") as handle:
        for line_index, line in enumerate(handle):
            if line_index < frame_start:
                continue
            if line_index >= frame_start + atom_count:
                break
            fields = line.split(maxsplit=1)
            if not fields:
                return None
            labels.append(fields[0])
    return labels if len(labels) == atom_count else None


def xyz_has_variable_species_order(
    filename: str | Path,
    sample_frames: Sequence[int] = (0, 1, 10, 100),
) -> bool:
    """Quickly detect an XYZ trajectory whose atom rows change species order."""

    source = Path(filename)
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        try:
            atom_count = int(handle.readline().strip())
        except ValueError as exc:
            raise ValueError("XYZ file does not begin with an atom count") from exc
    reference = _xyz_species_sequence_for_frame(source, 0, atom_count)
    if reference is None:
        return False
    return any(
        sequence is not None and sequence != reference
        for sequence in (
            _xyz_species_sequence_for_frame(source, frame_index, atom_count)
            for frame_index in sample_frames[1:]
        )
    )


def normalize_xyz_species_order(
    filename: str | Path,
    output_filename: str | Path | None = None,
) -> str:
    """Create or reuse an XYZ cache with a stable per-frame species order.

    Structural calculations do not require persistent atom identities, but
    MDAnalysis does retain labels from the first XYZ frame.  Reordering every
    frame by the first-frame species sequence makes its static labels valid and
    avoids carrying a full per-frame label table into multiprocessing workers.
    """

    source = Path(filename).expanduser()
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        try:
            atom_count = int(handle.readline().strip())
        except ValueError as exc:
            raise ValueError("XYZ file does not begin with an atom count") from exc
        if not handle.readline():
            raise ValueError("Malformed XYZ file: missing first frame header")
        first_lines = [handle.readline() for _ in range(atom_count)]
    if len(first_lines) != atom_count or any(line == "" for line in first_lines):
        raise ValueError("Malformed XYZ file: incomplete first atom block")

    species_order: list[str] = []
    species_rank: dict[str, int] = {}
    for line in first_lines:
        fields = line.split(maxsplit=1)
        if not fields:
            raise ValueError("Malformed XYZ file: empty atom line")
        species = fields[0]
        if species not in species_rank:
            species_rank[species] = len(species_order)
            species_order.append(species)

    if output_filename is None:
        cache_dir = Path(tempfile.gettempdir()) / "mdt_xyz_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        stat = source.stat()
        digest = hashlib.sha1(
            f"{source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
        ).hexdigest()[:12]
        destination = cache_dir / f"{source.stem}_{digest}_species_ordered{source.suffix}"
    else:
        destination = Path(output_filename).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return str(destination)

    with source.open("r", encoding="utf-8", errors="replace") as fin, destination.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        while True:
            atom_count_line = fin.readline()
            if not atom_count_line:
                break
            header_line = fin.readline()
            if not header_line:
                raise ValueError("Malformed XYZ file: missing frame header")
            try:
                frame_atom_count = int(atom_count_line.strip())
            except ValueError as exc:
                raise ValueError("Malformed XYZ file: invalid atom count") from exc
            atom_lines = [fin.readline() for _ in range(frame_atom_count)]
            if len(atom_lines) != frame_atom_count or any(line == "" for line in atom_lines):
                raise ValueError("Malformed XYZ file: incomplete atom block")
            grouped: dict[str, list[str]] = {}
            for line in atom_lines:
                fields = line.split(maxsplit=1)
                if not fields:
                    raise ValueError("Malformed XYZ file: empty atom line")
                species = fields[0]
                if species not in species_rank:
                    species_rank[species] = len(species_order)
                    species_order.append(species)
                grouped.setdefault(species, []).append(line)
            fout.write(atom_count_line)
            fout.write(header_line)
            for species in species_order:
                fout.writelines(grouped.get(species, ()))
    return str(destination)

def _atom_labels(universe: mda.Universe, requested: str) -> tuple[str, NDArray[np.str_]]:
    attributes = ("elements", "names", "types") if requested == "auto" else (requested,)
    for attribute in attributes:
        try:
            labels = np.asarray(getattr(universe.atoms, attribute), dtype=str)
        except (AttributeError, NoDataError):
            continue
        if len(labels) == universe.atoms.n_atoms and np.all(labels != ""):
            labels.setflags(write=False)
            return attribute, labels
    raise ValueError(
        f"Universe has no usable atom attribute for {requested!r}; "
        "pass atom_attribute='names', 'types', or another topology attribute"
    )


def _coordinate_filenames(universe: mda.Universe) -> tuple[str, ...]:
    filename = getattr(universe.trajectory, "filename", None)
    if filename is None:
        return ()
    if isinstance(filename, (str, Path)):
        return (str(filename),)
    return tuple(str(value) for value in filename)


def load_trajectory(
    topology: str | Path | mda.Universe,
    coordinates: str | Path | Sequence[str | Path] | None = None,
    *,
    frames: slice | Sequence[int] | None = None,
    box: ArrayLike | None = None,
    shift_by_origin: bool = False,
    atom_attribute: str = "auto",
    normalize_species_order: str | bool = "auto",
    format: str | None = None,
    topology_format: str | None = None,
    **universe_kwargs: Any,
) -> Trajectory:
    """Wrap files or an existing Universe for package analyses.

    For extended XYZ, the custom loader preserves ``Lattice`` and ``Origin``
    metadata that MDAnalysis's XYZ reader otherwise discards.
    """

    if normalize_species_order not in ("auto", True, False):
        raise ValueError("normalize_species_order must be 'auto', True, or False")
    normalized_xyz = False
    if isinstance(topology, mda.Universe):
        if coordinates is not None or format is not None or topology_format is not None:
            raise ValueError("coordinates and format arguments cannot accompany a Universe")
        if universe_kwargs:
            raise ValueError("Universe constructor options cannot accompany a Universe")
        if normalize_species_order != "auto":
            raise ValueError("normalize_species_order cannot accompany a Universe")
        universe = topology
    else:
        input_path = Path(topology).expanduser()
        load_path = input_path
        if coordinates is None and input_path.suffix.lower() == ".xyz" and normalize_species_order:
            should_normalize = normalize_species_order is True
            if normalize_species_order == "auto":
                should_normalize = xyz_has_variable_species_order(input_path)
            if should_normalize:
                load_path = Path(normalize_xyz_species_order(input_path))
                normalized_xyz = True
        args: list[Any] = [str(load_path)]
        if coordinates is not None:
            if isinstance(coordinates, (str, Path)):
                args.append(str(Path(coordinates).expanduser()))
            else:
                args.append([str(Path(value).expanduser()) for value in coordinates])
        options = dict(universe_kwargs)
        if format is not None:
            options["format"] = format
        if topology_format is not None:
            options["topology_format"] = topology_format
        universe = mda.Universe(*args, **options)

    source_indices = _source_frame_indices(universe.trajectory, frames)
    box_override = None if box is None else _normalize_box(box)
    reader_format = getattr(universe.trajectory, "format", None)
    if isinstance(universe.trajectory, XYZReader):
        headers = _read_xyz_headers(universe.trajectory, source_indices)
        if normalized_xyz:
            frame_species_codes, species_labels = None, ()
        else:
            frame_species_codes, species_labels = _read_xyz_species_codes(
                universe.trajectory, source_indices
            )
        locations = tuple(
            FrameLocation(
                index,
                _header_box(headers[index], box_override),
                _header_origin(headers[index]),
            )
            for index in source_indices
        )
        if any(location.dimensions is None for location in locations):
            raise ValueError("XYZ frames have no Lattice field; pass box=... to load_trajectory")
    else:
        frame_species_codes = None
        species_labels = ()
        locations = tuple(FrameLocation(index, box_override, None) for index in source_indices)
    if box_override is not None:
        universe.dimensions = np.asarray(box_override, dtype=np.float32)

    resolved_attribute, labels = _atom_labels(universe, atom_attribute)
    topology_filename = getattr(universe, "filename", None)
    source = TrajectorySource(
        frames=locations,
        topology=None if topology_filename is None else str(topology_filename),
        coordinates=_coordinate_filenames(universe),
        format=None if reader_format is None else str(reader_format),
        atom_attribute=resolved_attribute,
        shift_by_origin=bool(shift_by_origin),
        frame_species_codes=frame_species_codes,
        species_labels=species_labels,
    )
    return Trajectory(universe, source, labels)


__all__ = [
    "FrameLocation",
    "Trajectory",
    "TrajectoryFrame",
    "TrajectorySource",
    "load_trajectory",
    "normalize_xyz_species_order",
    "xyz_has_variable_species_order",
]
