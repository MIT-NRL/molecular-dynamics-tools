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
        |
        +----> serial calculator streams selected frames
        |
        +----> workers receive the indexed Universe once and stream contiguous chunks
```

Coordinate arrays are deliberately absent from `TrajectorySource`. For XYZ, a
bounded serial import such as `frames=slice(0, 10)` reads metadata only through
frame 9. Multiprocessing asks MDAnalysis to build its full random-access index
once in the parent so workers do not repeat the scan.

## Shared execution contract

`_execution` owns CPU validation, affinity, frame chunking, and the
serial-versus-multiprocessing plan.

- `ncore` is a hard total logical-CPU budget.
- Serial: one process, `ncore` native threads.
- Multiprocessing: at most `ncore` processes, one native thread each.
- `backend="auto"` chooses multiprocessing only when the core budget, selected
  frame count, and calculator-specific workload estimate justify process
  startup. Explicit `serial` and `multiprocessing` choices bypass this estimate.
- Worker processes receive the indexed, pickleable Universe once in their
  initializer, not coordinates.
- Linux affinity is applied temporarily and restored after each call.

All future trajectory calculators should use `plan_execution` and contiguous
frame chunks rather than creating analysis-specific multiprocessing policies.

## Module responsibilities

### `trajectory`

Implemented:

- `Trajectory` and `TrajectoryFrame`
- `load_trajectory`
- MDAnalysis `Universe` construction from one file or topology/coordinate inputs
- Existing-Universe wrapping and delegated atom selections
- Topology-derived atom labels
- Extended-XYZ lattice and origin metadata supplementation
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
- Single-process vectorized execution without an unnecessary `ncore` argument

### `transforms`

Implemented batched spherical-Bessel transforms with numerically equivalent
ZoomFFT and direct trapezoidal backends. This module does not depend on
SeanFunctions.

### `coordination`

Implemented cutoff and relative-angular-distance coordination calculations:

- `compute_coordination`
- `compute_rad_coordination`
- `summarize_coordination`

Multiple definitions share one streamed pass through each frame chunk. RAD
supports directed shells and mutual (`RAD-and`) bonds.

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
percolation tables. All shared-neighbor connections are categorized as
connected, corner, edge, and face by default.

### Internal cluster graph implementation

Implemented union-find, connected-component, and periodic-wrapping kernels
shared by distance and shared-neighbor clustering.

Sharing and percolation are evaluated from the same per-frame network, so
center-neighbor construction is not repeated. The result groups sharing
distributions, cluster distributions, per-frame wrapping data, finite-cluster
moments, and aggregate percolation statistics.

### `environments`

Planned specialized local-environment analyses that do not fit a general
coordination or clustering abstraction.

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
`artifacts/` and are not committed. An experiment should record enough inputs
and numerical settings in its output metadata to make a local result auditable.
