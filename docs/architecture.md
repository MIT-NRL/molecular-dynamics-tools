# Architecture and naming

This package is organized around one imported `Trajectory` object. Analysis
functions accept that object rather than separately interpreting filenames,
boxes, species ordering, or worker input.

## Public naming conventions

- Use `snake_case` for functions and arguments and `PascalCase` for classes.
- Prefer a scientific operation over an implementation detail in public names.
- Prefer one function that accepts one or many definitions over separate
  singular and `all_*` functions.
- Give every calculator that needs parallel execution the same `ncore`
  total-core contract; inexpensive vectorized calculators omit the parameter.
- Keep backend and worker orchestration private.
- Return named result objects when an analysis has multiple related tables.
  Simple single-table analyses may return a `pandas.DataFrame` directly.
- Keep plotting optional and separate from numerical calculations.

## Trajectory data flow

```text
load_trajectory(path)
        |
        v
Trajectory
  - public MDAnalysis Universe
  - selected source-frame numbers
  - topology-derived atom labels
  - custom extended-XYZ box/origin metadata
  - optional normalized-XYZ file-backed copy
        |
        +----> serial calculator streams selected frames
        |
        +----> workers receive the indexed Universe once and stream contiguous chunks
```

Coordinate arrays are deliberately absent from `TrajectorySource`. For XYZ,
automatic species-order detection samples a few source frames. If order or
column layout needs normalization, the full file is rewritten to a reusable
temporary copy before frame selection; its location and size are reported.
Any remaining frame-to-frame label changes are represented by compact
per-frame codes.
Multiprocessing asks MDAnalysis to build its random-access index once in the
parent so workers do not repeat the scan. Normalization preserves species
labels, but does not establish persistent atom identities across frames.

## Shared execution contract

`_execution` owns CPU validation, affinity, frame chunking, and the
serial-versus-multiprocessing plan.

- `ncore` is the total process/native-thread budget, defaulting to one.
- Serial: one process, `ncore` native threads.
- Multiprocessing: at most `ncore` processes, one native thread each.
- `backend="auto"` chooses multiprocessing only when the core budget, selected
  frame count, and calculator-specific workload estimate justify process
  startup. Explicit `serial` and `multiprocessing` choices bypass this estimate.
- Worker processes receive the indexed, pickleable Universe once in their
  initializer, not coordinates.
- Progress is opt-in with `show_progress=True`.
- Linux affinity constrains the selected CPUs and is restored after each call;
  on platforms without that API, worker and library thread counts are limited
  but processes are not pinned to particular CPUs.

Trajectory calculators share `plan_execution` and contiguous frame chunks.
RDFs use a dedicated accumulation loop but the same process/thread plan and
worker limits.

## Module responsibilities

### `trajectory`

Implemented:

- `Trajectory` and `TrajectoryFrame`
- `load_trajectory`
- MDAnalysis `Universe` construction from one file or topology/coordinate inputs
- Existing-Universe wrapping and delegated atom selections
- Topology-derived atom labels
- Extended-XYZ lattice and origin metadata supplementation
- Automatic XYZ species-order/column normalization into an atomically written,
  reusable temporary copy when needed
- Compact per-frame species codes when XYZ order varies outside the quick
  normalization sample
- Sequential and indexed worker frame iteration

Formats supported by MDAnalysis are available behind the same `Trajectory`
interface.

### `rdf`

Implemented:

- `compute_rdfs` for histogram RDFs
- `compute_spectral_rdfs` for fixed-mode cosine RDFs
- Explicit or all-unique atom pairs
- Fixed bin-count or exact radial-step histogram resolution
- Global, pair-specific, or pilot-elbow automatic spectral mode cutoffs
- Exact-step spectral output grids
- Streaming coefficient accumulation without distance caching
- Reuse of automatic pilot sums during final accumulation
- Serial and file-backed multiprocessing execution
- Same-species self-exclusion
- Ordered-pair ideal-gas shell normalization
- Variable per-frame cell-volume normalization
- Validation against the safe periodic query radius before neighbor searches

### `scattering`

Implemented as a package:

- `ScatteringComposition` from mappings, formulas, or trajectory counts
- Natural-element and isotope-mixture coherent neutron lengths
- Neutral and ionic Q-dependent X-ray form factors
- Faber–Ziman and Ashcroft–Langreth pair weights
- `compute_partial_structure_factors`
- `compute_scattering` for neutron/X-ray S(Q) and normalized weighted g(r)
- Separate unweighted partial RDFs and weighted radial distributions
- Unity-baseline S(Q) and dimensionless g(r) plotting helpers
- Direct neutron RDF weighting and optional Lorch-windowed X-ray transforms
- A warning and explicit ideal `g_ii(r)=1` placeholder when metadata confirms
  that a missing self-pair belongs to a singleton species
- Single-process vectorized execution without an unnecessary `ncore` argument

### `transforms`

Implemented batched spherical-Bessel transforms with equivalent ZoomFFT and
direct trapezoidal paths on uniform grids; direct Simpson integration is also
available.

### `coordination`

Implemented cutoff and relative-angular-distance coordination calculations:

- `compute_coordination`
- `compute_rad_coordination`
- `summarize_coordination`

Multiple definitions share one streamed pass through each frame chunk. RAD
uses the closed-shell variant by default and supports directed shells and
mutual (`RAD-and`) bonds.

### `_rad`

Internal periodic RAD-closed neighbor geometry shared by coordination and
atom-resolved environment calculations.

### `angles`

Implemented bond-angle distributions:

- `compute_bond_angles`

Equivalent outer atoms are deduplicated for symmetric bond definitions.

### `clustering`

Public clustering calculations are grouped under `mdt.clustering`:

- `compute_by_distance`
- `compute_by_shared_neighbors`

Both return named result objects with a `cluster_distribution` table and
metadata. Shared-neighbor results additionally expose sharing, per-frame, and
percolation tables. By default, shared-neighbor results include the union
`connected` network and mutually exclusive corner-, edge-, and face-sharing
networks. Distance clustering can normalize by component or by atoms of a
selected species; shared-neighbor cluster distributions are center-atom
normalized.

### Internal cluster graph implementation

Implemented union-find and connected-component kernels shared by distance and
shared-neighbor clustering. Periodic-wrapping analysis is used by the
shared-neighbor method.

Sharing and percolation are evaluated from the same per-frame network, so
center-neighbor construction is not repeated. The result groups sharing
distributions, cluster distributions, per-frame wrapping data, finite-cluster
moments, and aggregate percolation statistics.

### `rad` and `environments`

`rad` implements atom-resolved RAD environments with species counts, periodic
contact distances, mutual-contact classification, and speciation. Occupancy
and sample-based contact lifetimes require the caller to confirm stable atom
identities. `environments` re-exports this public API.

### `cache`

Implemented reusable result caching through `AnalysisCache`. Scientific inputs,
the analytical trajectory view, source-file fingerprints, package version, and
analysis implementation identify entries; process counts and progress do not.
Each `.mdtc` entry is an atomically replaced ZIP archive with a checksummed
JSON manifest. Tables use Parquet when PyArrow is installed and pandas table
JSON otherwise; arrays use `.npy`. Cache storage never uses pickle.

## Compatibility policy

This is a clean rebuild, so renamed APIs do not require indefinite aliases.
Numerical regression tests should compare each rebuilt tool with the original
helper or an authoritative backend. Compatibility wrappers should be added only
when an existing production script needs a gradual transition.

## Repository data policy

Package source and portable tests must not depend on workstation-specific data.
Small synthetic trajectories should be generated inside tests. Large production
trajectories remain outside the repository and may be used by explicitly
optional regression tests, benchmarks, or scripts under `experiments/`.

Generated plots, tables, timing data, and comparison metadata are written below
`artifacts/` and are not committed. Local trajectory roots are supplied to
experiments through `MDT_EXPERIMENT_DATA`; `data/` and `.mdtc` caches are ignored.
An experiment should record enough inputs and numerical settings in its output
metadata to make a local result auditable.
