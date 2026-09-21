"""Persistent, provenance-aware caches for analysis results.

The cache stores results rather than trajectory coordinates.  Entries are
single, atomically replaced zip files containing a JSON manifest and safe
non-pickle representations of pandas, NumPy, and package dataclass objects.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import inspect
import io
import json
import math
import os
import tempfile
import time
import zipfile
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np
import pandas as pd

CacheMode = Literal["use", "refresh", "read_only", "off"]
FingerprintMode = Literal["stat", "sha256"]
StorageFormat = Literal["auto", "json", "parquet"]

CACHE_SCHEMA_VERSION = 1
_EXECUTION_PARAMETERS = frozenset({"ncore", "show_progress"})
_RESULT_ROOT = "result"
_T = TypeVar("_T")


class CacheError(RuntimeError):
    """Base exception for persistent analysis caches."""


class CacheMissError(CacheError):
    """Raised when a read-only cache does not contain the requested result."""


class CacheCorruptionError(CacheError):
    """Raised when a cache entry is incomplete, corrupt, or incompatible."""


@dataclass(frozen=True, slots=True)
class CacheInfo:
    """Information about the most recent :meth:`AnalysisCache.get_or_compute` call."""

    key: str | None
    path: Path | None
    hit: bool
    mode: CacheMode


def _qualified_name(value: type[Any] | Callable[..., Any]) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _float_token(value: float) -> dict[str, str]:
    if math.isnan(value):
        token = "nan"
    elif math.isinf(value):
        token = "+inf" if value > 0 else "-inf"
    else:
        token = value.hex()
    return {"type": "float", "value": token}


class _Canonicalizer:
    def __init__(self, fingerprint_mode: FingerprintMode, source_id: str | None = None) -> None:
        self.fingerprint_mode = fingerprint_mode
        self.source_id = source_id
        self._files: dict[Path, dict[str, Any]] = {}

    def file(self, path: str | Path) -> dict[str, Any]:
        resolved = Path(path).expanduser().resolve()
        cached = self._files.get(resolved)
        if cached is not None:
            return cached
        if not resolved.is_file():
            raise FileNotFoundError(f"cache source file does not exist: {resolved}")
        stat = resolved.stat()
        if self.fingerprint_mode == "stat":
            result = {
                "type": "file",
                "mode": "stat",
                "path": str(resolved),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        else:
            result = {
                "type": "file",
                "mode": "sha256",
                "size": stat.st_size,
                "sha256": _sha256_file(resolved),
            }
        self._files[resolved] = result
        return result

    def trajectory(self, value: Any) -> dict[str, Any]:
        source = value.source
        paths: list[dict[str, Any]] = []
        seen: set[Path] = set()
        source_paths = []
        if source.topology is not None:
            source_paths.append(("topology", source.topology))
        source_paths.extend(("coordinates", filename) for filename in source.coordinates)
        for role, filename in source_paths:
            resolved = Path(filename).expanduser().resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append({"role": role, "fingerprint": self.file(resolved)})
        reader = value.universe.trajectory
        in_memory = reader.__class__.__name__ == "MemoryReader"
        transformed = bool(getattr(reader, "transformations", ())) or bool(
            getattr(reader, "_transformations", ())
        )
        if (not paths or in_memory or transformed) and self.source_id is None:
            raise ValueError(
                "persistent caching of an in-memory or transformed trajectory "
                "requires cache_source_id=..."
            )
        return {
            "type": "molecular_dynamics_tools.Trajectory",
            "sources": paths,
            "source_id": self.source_id,
            "format": source.format,
            "atom_attribute": source.atom_attribute,
            "shift_by_origin": source.shift_by_origin,
            "frames": self.value(source.frames),
            "frame_species_codes": self.value(source.frame_species_codes),
            "species_labels": self.value(source.species_labels),
            "atom_labels": self.value(value._atom_labels),
            "n_atoms": value.n_atoms,
        }

    def dataframe(self, value: pd.DataFrame) -> dict[str, Any]:
        try:
            hashes = pd.util.hash_pandas_object(value, index=True, categorize=True)
        except (TypeError, ValueError):
            table = value.to_json(orient="table", date_format="iso", double_precision=15).encode(
                "utf-8"
            )
            data_hash = _sha256_bytes(table)
        else:
            data_hash = _sha256_bytes(hashes.to_numpy(dtype=np.uint64).tobytes())
        return {
            "type": "pandas.DataFrame",
            "shape": list(value.shape),
            "columns": self.value(tuple(value.columns.tolist())),
            "index_names": self.value(tuple(value.index.names)),
            "dtypes": [str(dtype) for dtype in value.dtypes],
            "data_sha256": data_hash,
            "attrs": self.value(value.attrs),
        }

    def value(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, str)):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return {"type": "int", "value": str(value)}
        if isinstance(value, float):
            return _float_token(value)
        if isinstance(value, complex):
            return {
                "type": "complex",
                "real": _float_token(value.real),
                "imag": _float_token(value.imag),
            }
        if isinstance(value, np.generic):
            return {
                "type": "numpy.scalar",
                "dtype": str(value.dtype),
                "value": self.value(value.item()),
            }
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                contents = self.value(value.tolist())
                digest = _sha256_bytes(_json_bytes(contents))
            else:
                contiguous = np.ascontiguousarray(value)
                digest = _sha256_bytes(contiguous.view(np.uint8).tobytes())
                contents = None
            return {
                "type": "numpy.ndarray",
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": digest,
                **({"contents": contents} if contents is not None else {}),
            }
        if isinstance(value, pd.DataFrame):
            return self.dataframe(value)
        if (
            value.__class__.__module__ == "molecular_dynamics_tools.trajectory"
            and value.__class__.__name__ == "Trajectory"
        ):
            return self.trajectory(value)
        if isinstance(value, Path):
            return (
                self.file(value)
                if value.is_file()
                else {
                    "type": "path",
                    "value": str(value.expanduser().resolve()),
                }
            )
        if isinstance(value, Enum):
            return {
                "type": "enum",
                "class": _qualified_name(type(value)),
                "value": self.value(value.value),
            }
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                "type": "dataclass",
                "class": _qualified_name(type(value)),
                "fields": [
                    [field.name, self.value(getattr(value, field.name))]
                    for field in dataclasses.fields(value)
                ],
            }
        if isinstance(value, Mapping):
            items = [[self.value(key), self.value(item)] for key, item in value.items()]
            items.sort(key=lambda item: _json_bytes(item[0]))
            return {"type": "mapping", "items": items}
        if isinstance(value, tuple):
            return {"type": "tuple", "items": [self.value(item) for item in value]}
        if isinstance(value, list):
            return {"type": "list", "items": [self.value(item) for item in value]}
        if isinstance(value, (set, frozenset)):
            items = [self.value(item) for item in value]
            items.sort(key=_json_bytes)
            return {
                "type": "frozenset" if isinstance(value, frozenset) else "set",
                "items": items,
            }
        if isinstance(value, slice):
            return {
                "type": "slice",
                "start": self.value(value.start),
                "stop": self.value(value.stop),
                "step": self.value(value.step),
            }
        if isinstance(value, range):
            return {
                "type": "range",
                "start": value.start,
                "stop": value.stop,
                "step": value.step,
            }
        if isinstance(value, (datetime, date)):
            return {"type": type(value).__name__, "value": value.isoformat()}
        if callable(value):
            return {"type": "callable", "name": _qualified_name(value)}
        raise TypeError(f"cannot construct a reproducible cache key for {type(value).__name__}")


class _ResultWriter:
    def __init__(self, storage: StorageFormat) -> None:
        self.storage = self._resolve_storage(storage)
        self.blobs: dict[str, bytes] = {}
        self._counter = 0

    @staticmethod
    def _resolve_storage(storage: StorageFormat) -> Literal["json", "parquet"]:
        if storage == "json":
            return "json"
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            if storage == "parquet":
                raise CacheError(
                    "Parquet cache storage requires pyarrow; install "
                    "molecular-dynamics-tools[cache] or use storage='json'"
                ) from None
            return "json"
        return "parquet"

    def _blob(self, suffix: str, contents: bytes) -> str:
        self._counter += 1
        name = f"{_RESULT_ROOT}/{self._counter:04d}{suffix}"
        self.blobs[name] = contents
        return name

    def value(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, str, int)):
            return value
        if isinstance(value, float):
            return _float_token(value)
        if isinstance(value, complex):
            return {
                "type": "complex",
                "real": _float_token(value.real),
                "imag": _float_token(value.imag),
            }
        if isinstance(value, np.generic):
            return {
                "type": "numpy.scalar",
                "dtype": str(value.dtype),
                "value": self.value(value.item()),
            }
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                return {
                    "type": "numpy.object_array",
                    "shape": list(value.shape),
                    "items": self.value(value.tolist()),
                }
            buffer = io.BytesIO()
            np.save(buffer, value, allow_pickle=False)
            return {"type": "numpy.ndarray", "file": self._blob(".npy", buffer.getvalue())}
        if isinstance(value, pd.DataFrame):
            attrs = self.value(value.attrs)
            if self.storage == "parquet":
                buffer = io.BytesIO()
                # Pandas forwards DataFrame.attrs to Parquet metadata and
                # requires it to be JSON serializable.  MDT attributes may
                # contain definitions/dataclasses, which this cache already
                # serializes safely in the manifest below.  Write a shallow
                # attribute-free view and restore attrs on cache reads.
                table = value.copy(deep=False)
                table.attrs = {}
                table.to_parquet(buffer, index=True, engine="pyarrow")
                filename = self._blob(".parquet", buffer.getvalue())
                storage = "parquet"
            else:
                contents = value.to_json(
                    orient="table", date_format="iso", double_precision=15
                ).encode("utf-8")
                filename = self._blob(".table.json", contents)
                storage = "json-table"
            return {
                "type": "pandas.DataFrame",
                "storage": storage,
                "file": filename,
                "attrs": attrs,
            }
        if isinstance(value, Enum):
            return {
                "type": "enum",
                "class": _qualified_name(type(value)),
                "value": self.value(value.value),
            }
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                "type": "dataclass",
                "class": _qualified_name(type(value)),
                "fields": {
                    field.name: self.value(getattr(value, field.name))
                    for field in dataclasses.fields(value)
                },
            }
        if isinstance(value, Path):
            return {"type": "path", "value": str(value)}
        if isinstance(value, Mapping):
            return {
                "type": "mapping",
                "items": [[self.value(key), self.value(item)] for key, item in value.items()],
            }
        if isinstance(value, tuple):
            return {"type": "tuple", "items": [self.value(item) for item in value]}
        if isinstance(value, list):
            return {"type": "list", "items": [self.value(item) for item in value]}
        if isinstance(value, (set, frozenset)):
            return {
                "type": "frozenset" if isinstance(value, frozenset) else "set",
                "items": [self.value(item) for item in value],
            }
        if isinstance(value, slice):
            return {
                "type": "slice",
                "start": self.value(value.start),
                "stop": self.value(value.stop),
                "step": self.value(value.step),
            }
        if isinstance(value, range):
            return {
                "type": "range",
                "start": value.start,
                "stop": value.stop,
                "step": value.step,
            }
        if isinstance(value, (datetime, date)):
            return {"type": type(value).__name__, "value": value.isoformat()}
        raise TypeError(f"cache cannot safely serialize {type(value).__name__}")


def _decode_float(node: Mapping[str, Any]) -> float:
    value = node["value"]
    if value == "nan":
        return float("nan")
    if value == "+inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    return float.fromhex(value)


def _package_type(qualified_name: str) -> type[Any]:
    module_name, _, name = qualified_name.rpartition(".")
    if not module_name.startswith("molecular_dynamics_tools"):
        raise CacheCorruptionError(
            f"cached type is outside molecular_dynamics_tools: {qualified_name}"
        )
    try:
        module = importlib.import_module(module_name)
        value: Any = module
        for part in name.split("."):
            value = getattr(value, part)
    except (AttributeError, ImportError) as exc:
        raise CacheCorruptionError(f"cached type is unavailable: {qualified_name}") from exc
    if not isinstance(value, type):
        raise CacheCorruptionError(f"cached object is not a type: {qualified_name}")
    return value


class _ResultReader:
    def __init__(self, archive: zipfile.ZipFile) -> None:
        self.archive = archive

    def value(self, node: Any) -> Any:
        if node is None or isinstance(node, (bool, str, int)):
            return node
        if not isinstance(node, dict) or "type" not in node:
            raise CacheCorruptionError("cached result contains an invalid value")
        kind = node["type"]
        if kind == "float":
            return _decode_float(node)
        if kind == "complex":
            return complex(_decode_float(node["real"]), _decode_float(node["imag"]))
        if kind == "numpy.scalar":
            return np.asarray(self.value(node["value"]), dtype=node["dtype"])[()]
        if kind == "numpy.ndarray":
            with self.archive.open(node["file"]) as handle:
                return np.load(io.BytesIO(handle.read()), allow_pickle=False)
        if kind == "numpy.object_array":
            values = self.value(node["items"])
            return np.asarray(values, dtype=object).reshape(node["shape"])
        if kind == "pandas.DataFrame":
            with self.archive.open(node["file"]) as handle:
                contents = handle.read()
            if node["storage"] == "parquet":
                try:
                    result = pd.read_parquet(io.BytesIO(contents), engine="pyarrow")
                except ImportError as exc:
                    raise CacheCorruptionError("reading this cache entry requires pyarrow") from exc
            elif node["storage"] == "json-table":
                result = pd.read_json(io.BytesIO(contents), orient="table")
            else:
                raise CacheCorruptionError(f"unknown DataFrame storage {node['storage']!r}")
            result.attrs = self.value(node["attrs"])
            return result
        if kind == "enum":
            return _package_type(node["class"])(self.value(node["value"]))
        if kind == "dataclass":
            cls = _package_type(node["class"])
            if not dataclasses.is_dataclass(cls):
                raise CacheCorruptionError(f"cached type is not a dataclass: {node['class']}")
            return cls(**{key: self.value(value) for key, value in node["fields"].items()})
        if kind == "path":
            return Path(node["value"])
        if kind == "mapping":
            return {self.value(key): self.value(value) for key, value in node["items"]}
        if kind in {"tuple", "list", "set", "frozenset"}:
            values = [self.value(item) for item in node["items"]]
            constructors = {
                "tuple": tuple,
                "list": list,
                "set": set,
                "frozenset": frozenset,
            }
            return constructors[kind](values)
        if kind == "slice":
            return slice(
                self.value(node["start"]),
                self.value(node["stop"]),
                self.value(node["step"]),
            )
        if kind == "range":
            return range(node["start"], node["stop"], node["step"])
        if kind == "datetime":
            return datetime.fromisoformat(node["value"])
        if kind == "date":
            return date.fromisoformat(node["value"])
        raise CacheCorruptionError(f"unknown cached value type {kind!r}")


def _function_fingerprint(function: Callable[..., Any]) -> dict[str, str]:
    source_file = inspect.getsourcefile(function)
    if source_file is not None and Path(source_file).is_file():
        implementation = _sha256_file(Path(source_file))
    else:
        try:
            source = inspect.getsource(function).encode("utf-8")
        except (OSError, TypeError):
            code = getattr(function, "__code__", None)
            if code is None:
                raise TypeError("cache function must expose Python source or bytecode")
            source = code.co_code
        implementation = _sha256_bytes(source)
    return {
        "name": _qualified_name(function),
        "implementation_sha256": implementation,
        "algorithm_version": str(getattr(function, "__cache_version__", "1")),
    }


def _package_version() -> str:
    from . import __version__

    return __version__


class AnalysisCache:
    """Persistent result cache for deterministic analysis functions.

    Parameters that affect execution but not scientific results (``ncore`` and
    ``show_progress``) are omitted from keys by default. ``backend`` is also
    omitted for calculators that accept ``ncore``; it is retained for transforms
    where it selects a numerical algorithm.
    Additional names can be omitted with ``exclude_parameters``.

    Args:
        root: Directory in which cache entries are stored.
        fingerprint: ``"stat"`` uses source path, size, and modification time;
            ``"sha256"`` hashes source contents and permits reuse after moves.
        storage: DataFrame representation. ``"auto"`` uses Parquet when
            pyarrow is installed and otherwise uses pandas table JSON.
        mode: Default cache behavior for calls.
        exclude_parameters: Additional non-scientific function parameters.
        lock_timeout: Maximum seconds to wait for another writer.
        stale_lock_age: Age in seconds after which an abandoned lock is removed.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        fingerprint: FingerprintMode = "stat",
        storage: StorageFormat = "auto",
        mode: CacheMode = "use",
        exclude_parameters: tuple[str, ...] = (),
        lock_timeout: float = 300.0,
        stale_lock_age: float = 86_400.0,
    ) -> None:
        if fingerprint not in {"stat", "sha256"}:
            raise ValueError("fingerprint must be 'stat' or 'sha256'")
        if storage not in {"auto", "json", "parquet"}:
            raise ValueError("storage must be 'auto', 'json', or 'parquet'")
        if mode not in {"use", "refresh", "read_only", "off"}:
            raise ValueError("mode must be 'use', 'refresh', 'read_only', or 'off'")
        if lock_timeout <= 0 or stale_lock_age <= 0:
            raise ValueError("lock timeouts must be positive")
        self.root = Path(root).expanduser()
        self.fingerprint = fingerprint
        self.storage = storage
        self.mode = mode
        self.exclude_parameters = _EXECUTION_PARAMETERS | frozenset(exclude_parameters)
        self.lock_timeout = float(lock_timeout)
        self.stale_lock_age = float(stale_lock_age)
        self.last_info: CacheInfo | None = None

    def _key_material(
        self,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        source_id: str | None = None,
    ) -> dict[str, Any]:
        if source_id is not None and (not isinstance(source_id, str) or not source_id.strip()):
            raise ValueError("cache_source_id must be a nonempty string")
        try:
            signature = inspect.signature(function)
            bound = signature.bind(*args, **kwargs)
        except TypeError as exc:
            raise TypeError(f"invalid cached analysis call: {exc}") from exc
        bound.apply_defaults()
        canonicalizer = _Canonicalizer(self.fingerprint, source_id)
        excluded = self.exclude_parameters
        if "ncore" in signature.parameters:
            excluded = excluded | {"backend"}
        parameters = {
            name: canonicalizer.value(value)
            for name, value in bound.arguments.items()
            if name not in excluded
        }
        return {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "package_version": _package_version(),
            "function": _function_fingerprint(function),
            "fingerprint_mode": self.fingerprint,
            "parameters": parameters,
        }

    def cache_key(
        self,
        function: Callable[..., Any],
        /,
        *args: Any,
        cache_source_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Return the deterministic key for an analysis call without running it."""

        material = self._key_material(function, args, kwargs, source_id=cache_source_id)
        return _sha256_bytes(_json_bytes(material))

    def _entry_path(self, function: Callable[..., Any], key: str) -> Path:
        name = function.__name__.removeprefix("compute_") or "analysis"
        return self.root / name / f"{key}.mdtc"

    @contextmanager
    def _lock(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.lock_timeout
        while True:
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    age = time.time() - path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age > self.stale_lock_age:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                if time.monotonic() >= deadline:
                    raise CacheError(f"timed out waiting for cache lock {path}")
                time.sleep(0.1)
                continue
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    created = datetime.now(timezone.utc).isoformat()
                    handle.write(f"pid={os.getpid()} created={created}\n")
                yield
            finally:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            return

    def _write_entry(
        self,
        path: Path,
        key: str,
        key_material: dict[str, Any],
        result: Any,
    ) -> None:
        writer = _ResultWriter(self.storage)
        result_node = writer.value(result)
        files = {
            name: {"sha256": _sha256_bytes(contents), "size": len(contents)}
            for name, contents in writer.blobs.items()
        }
        manifest = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "key": key,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "storage": writer.storage,
            "key_material": key_material,
            "result": result_node,
            "files": files,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{key}.", suffix=".tmp", dir=path.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
            ) as archive:
                for name, contents in writer.blobs.items():
                    archive.writestr(name, contents)
                archive.writestr("manifest.json", _json_bytes(manifest))
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _read_entry(path: Path, expected_key: str) -> Any:
        try:
            with zipfile.ZipFile(path, "r") as archive:
                try:
                    manifest = json.loads(archive.read("manifest.json"))
                except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise CacheCorruptionError(f"invalid cache manifest in {path}") from exc
                if manifest.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
                    raise CacheCorruptionError(f"incompatible cache schema in {path}")
                if manifest.get("key") != expected_key:
                    raise CacheCorruptionError(f"cache key mismatch in {path}")
                for name, expected in manifest.get("files", {}).items():
                    try:
                        contents = archive.read(name)
                    except KeyError as exc:
                        raise CacheCorruptionError(
                            f"cache data file {name!r} is missing from {path}"
                        ) from exc
                    wrong_size = len(contents) != expected["size"]
                    wrong_hash = _sha256_bytes(contents) != expected["sha256"]
                    if wrong_size or wrong_hash:
                        raise CacheCorruptionError(f"cache data file {name!r} failed validation")
                try:
                    return _ResultReader(archive).value(manifest["result"])
                except CacheCorruptionError:
                    raise
                except (AttributeError, KeyError, TypeError, ValueError) as exc:
                    raise CacheCorruptionError(
                        f"cached result metadata is invalid in {path}"
                    ) from exc
        except (OSError, zipfile.BadZipFile) as exc:
            raise CacheCorruptionError(f"cannot read cache entry {path}") from exc

    def get_or_compute(
        self,
        function: Callable[..., _T],
        /,
        *args: Any,
        cache_mode: CacheMode | None = None,
        cache_source_id: str | None = None,
        **kwargs: Any,
    ) -> _T:
        """Load an exact result from the cache or execute and store the analysis.

        ``cache_mode`` overrides the instance default. In ``"read_only"`` mode,
        a missing entry raises :class:`CacheMissError`; an invalid entry raises
        :class:`CacheCorruptionError`. In ``"use"`` mode invalid entries are
        safely replaced after recomputation.
        """

        mode = self.mode if cache_mode is None else cache_mode
        if mode not in {"use", "refresh", "read_only", "off"}:
            raise ValueError("cache_mode must be 'use', 'refresh', 'read_only', or 'off'")
        if mode == "off":
            self.last_info = CacheInfo(None, None, False, mode)
            return function(*args, **kwargs)

        key_material = self._key_material(function, args, kwargs, source_id=cache_source_id)
        key = _sha256_bytes(_json_bytes(key_material))
        path = self._entry_path(function, key)
        if mode != "refresh" and path.is_file():
            try:
                result = self._read_entry(path, key)
            except CacheCorruptionError:
                if mode == "read_only":
                    raise
            else:
                self.last_info = CacheInfo(key, path, True, mode)
                return result
        elif mode == "read_only":
            raise CacheMissError(f"no cached result for key {key}")

        lock_path = path.with_suffix(path.suffix + ".lock")
        with self._lock(lock_path):
            if mode == "use" and path.is_file():
                try:
                    result = self._read_entry(path, key)
                except CacheCorruptionError:
                    pass
                else:
                    self.last_info = CacheInfo(key, path, True, mode)
                    return result
            result = function(*args, **kwargs)
            self._write_entry(path, key, key_material, result)
        self.last_info = CacheInfo(key, path, False, mode)
        return result


__all__ = [
    "AnalysisCache",
    "CACHE_SCHEMA_VERSION",
    "CacheCorruptionError",
    "CacheError",
    "CacheInfo",
    "CacheMissError",
    "CacheMode",
    "FingerprintMode",
    "StorageFormat",
]
