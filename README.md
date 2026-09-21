# Molecular Dynamics Tools

`molecular-dynamics-tools` is a standalone Python package for reusable
molecular-dynamics trajectory and structural-analysis tools.

The package is a clean rebuild of the original `MolecularDynamicsTools.py`
helper module. It provides trajectory import, RDFs, scattering, coordination,
bond-angle, and cluster-network calculations behind a consistent public API.

## Implemented

- MDAnalysis `Universe` backend with broad format support
- Custom extended-XYZ lattice and origin metadata handling
- Topology/coordinate file pairs and existing-Universe inputs
- Bounded trajectory import for inexpensive serial testing
- Serial and multiprocessing histogram and spectral RDF calculations
- Faber–Ziman and Ashcroft–Langreth neutron/X-ray scattering calculations
- FFT partial structure factors and normalized weighted radial distributions
- Cutoff and relative-angular-distance (RAD) coordination distributions
- Bond-angle probability densities
- Direct cutoff-bond cluster distributions
- Bridging-ligand corner/edge/face sharing and periodic percolation
- One shared total-core contract for analysis functions
- File-backed workers without coordinate precaching

## Planned

- Specialized local-environment analyses

See [docs/architecture.md](docs/architecture.md) for module boundaries and
naming conventions.

## Installation

Clone the repository and install it into an isolated environment. For a conda
development environment:

Using conda:

```bash
conda env create -f environment.yml
conda activate molecular-dynamics-tools
```

Using an existing Python environment:

```bash
python -m pip install -e ".[dev]"
```

For a runtime-only editable installation, use `python -m pip install -e .`.
Plotting is an optional dependency when the package is installed without the
development extras:

```bash
python -m pip install -e ".[plot]"
```

## Trajectory and RDF workflow

The distribution name uses hyphens, while the Python import uses underscores.
Trajectories can be used as context managers so file handles close promptly:

```python
import molecular_dynamics_tools as mdt

with mdt.load_trajectory("simulation.xyz") as trajectory:
    rdfs = mdt.compute_rdfs(
        trajectory,
        step=0.02,
        r_range=(0.0, None),
        ncore=24,
    )
```

`load_trajectory` returns a package-owned `Trajectory` wrapper whose public
`.universe` attribute is the MDAnalysis `Universe` used for coordinate loading.
The wrapper records frame selection and package metadata, but it does not cache
coordinates. Each frame is read only when requested. It accepts a single file,
a topology plus coordinate file, or an existing Universe:

```python
trajectory = mdt.load_trajectory("topology.psf", "production.dcd")
trajectory = mdt.load_trajectory(existing_universe)
protein = trajectory.universe.select_atoms("protein")
```

A bounded source slice stops indexing early, which is useful when testing a
large file:

```python
sample = mdt.load_trajectory(
    "very-large-simulation.xyz",
    frames=slice(0, 10),
)
```

Atom labels come from the Universe topology (`elements`, then `names`, then
`types` by default). As with MDAnalysis generally, atom identity and ordering
must remain stable across trajectory frames. Use `atom_attribute="names"` or
`atom_attribute="types"` when that better represents the desired RDF labels.
For extended XYZ, the custom loader preserves per-frame `Lattice` and `Origin`
metadata that MDAnalysis does not expose.

RDF resolution can be specified either by total bin count or by an exact radial
step. The latter keeps the sampling interval consistent across box sizes:

```python
rdfs = mdt.compute_rdfs(trajectory, step=0.02, r_range=(0.0, None))
```

If the safe radial range is not an exact multiple of `step`, the incomplete
final bin is omitted. For triclinic cells, the safe radius is half the shortest
perpendicular face separation rather than half the shortest lattice vector.
Explicit ranges are validated against every selected frame before workers are
started. `bins` and `step` cannot be supplied together.

Smooth spectral RDFs are available as a separate calculator:

```python
spectral = mdt.compute_spectral_rdfs(
    trajectory,
    modes={("Na", "Na"): 40, ("Na", "Cl"): 60, ("Cl", "Cl"): 50},
    step=0.02,
    r_range=(0.0, 6.0),
    ncore=32,
)
```

`modes` accepts one positive integer, a complete pair-to-integer mapping, or
`"auto"`. Pair keys are symmetric. Automatic selection calculates a bounded
pilot spectrum through `auto_max_modes=120`, fits the decay-to-noise-floor
elbow separately for every pair, reuses the pilot sums, and processes remaining
frames only through the selected cutoffs:

```python
spectral = mdt.compute_spectral_rdfs(
    trajectory,
    modes="auto",
    auto_pilot_frames=128,  # optional override
    ncore=32,
)
print(spectral.attrs["spectral"]["selected_modes"])
```

Without an override, the pilot targets roughly 100,000 atom-frames and is
clamped to 128--1,024 frames or the available trajectory length. Fit diagnostics
are stored under `result.attrs["spectral"]["auto"]`. Spectral `step` controls
the returned sampling grid; it does not create bins. The standard histogram
calculator remains the default recommendation.

## Coordination and bond-angle workflow

Cutoff coordination accepts one or more `(center, neighbor, r_max)` or
`(center, neighbor, r_min, r_max)` definitions and streams them in one pass:

```python
coordination = mdt.compute_coordination(
    trajectory,
    [
        ("Be", "F", 2.35),
        ("Li", "F", 2.75),
        ("Cs", "F", 3.50),
    ],
    ncore=24,
)
summary = mdt.summarize_coordination(coordination)
```

RAD coordination does not use a distance cutoff. Directed mode asks which
neighbors are in each center atom's RAD shell; mutual mode retains a bond only
when both atoms include one another:

```python
directed = mdt.compute_rad_coordination(
    trajectory,
    [("Be", "F"), ("Li", "F")],
    bond_mode="directed",
    ncore=24,
)
mutual = mdt.compute_rad_coordination(
    trajectory,
    [("Be", "F")],
    bond_mode="mutual",
    ncore=24,
)
```

Bond-angle definitions are `(first, center, third, first_center_max,
center_third_max)`. Equivalent outer atoms are counted as unordered pairs when
their cutoffs match, avoiding duplicate angles:

```python
angles = mdt.compute_bond_angles(
    trajectory,
    [
        ("F", "Be", "F", 2.35, 2.35),
        ("Be", "F", "Be", 2.35, 2.35),
    ],
    bins=180,
    ncore=24,
)
```

All returned tables store execution details in `result.attrs["execution"]`.
Use `CoordinationDefinition` or `AngleDefinition` when two definitions would
otherwise have the same generated column name and need explicit labels.

## Cluster workflow

Clustering is grouped under the `mdt.clustering` namespace. Both calculations
use the same connected-component machinery but differ in how graph connections
are defined.

`compute_by_distance` connects one species to itself or two species to each
other using a distance cutoff:

```python
distance = mdt.clustering.compute_by_distance(
    trajectory,
    species=("Be", "F"),
    cutoff=2.35,
    count_species="Be",
    include_isolated=False,
    ncore=24,
)

distance.cluster_distribution
```

With `count_species="Be"`, cluster size is the number of Be atoms and the
distribution is sampled once per Be atom. Without `count_species`, cluster
size is the total number of selected atoms and the distribution is sampled
once per connected component. Passing a single species, such as
`species="Li"`, constructs a same-species distance graph and excludes
self-pairs. `r_min=0.0` means no positive lower distance cutoff; it does not
make an atom its own neighbor. Set `r_min` to exclude shorter contacts; it must
satisfy `0 <= r_min < cutoff`. By default, size-one components are omitted;
set `include_isolated=True` to retain them.

`compute_by_shared_neighbors` projects a center-neighbor distance graph onto
the centers. The graph is built once per frame and reused for the requested
category distributions—all four by default—and percolation:

```python
network = mdt.clustering.compute_by_shared_neighbors(
    trajectory,
    centers="Be",
    neighbors="F",
    cutoff=2.35,
    ncore=24,
)

network.sharing_distribution              # shared-neighbor count and type
network.cluster_distribution              # connected/corner/edge/face clusters
network.frame_summary                     # links and wrapping per frame
network.percolation_cluster_distribution  # total and finite component counts
network.percolation_summary               # means, deviations, wrap probabilities
```

All four connection outputs are calculated by default; `connections=` can
request a subset. `connected` is the union of all center pairs sharing at
least one neighbor; one shared neighbor is corner sharing, two is edge sharing,
and three or more is face sharing. `min_shared_neighbors` is the minimum
shared-neighbor count for a percolation edge, independently of `connections`;
`3` retains links sharing three or more neighbors. Periodic
percolation is detected from inconsistent image translations around network
cycles, independently along x, y, and z. `include_isolated=False` omits
size-one center components from cluster and percolation component tables.

With `r_min=0.0`, matching cutoff and isolation settings, and `connections`
including `"connected"`, two-species distance clustering with `count_species`
set to the center is numerically equivalent to the `connected` shared-neighbor
distribution. The shared-neighbor calculation adds categorized networks and
percolation details.

## Scattering workflow

RDF results carry atom counts, pair identities, and mean number density, so a
scattering calculation made directly from MDT output does not require the
composition to be entered again:

```python
scattering = mdt.compute_scattering(
    rdfs,
    isotopes={"Li": {7: 0.99, 6: 0.01}},
    charges={"Li": 1, "Be": 2, "F": -1, "Cs": 1},
    q_range=(0.0, 25.0),
    q_step=0.01,
    convention="faber-ziman",
    xray_window="lorch",  # use None for an unwindowed X-ray inverse transform
)

scattering.partial_rdfs                 # unweighted g_ij(r)
scattering.partial_structure_factors    # convention-specific S_ij(Q)
scattering.neutron.structure_factor     # weighted S(Q)
scattering.neutron.weighted_rdf         # dimensionless weighted g(r)
scattering.xray.structure_factor
scattering.xray.weighted_rdf

mdt.plot_structure_factor(scattering.neutron)  # total and pairs use S(Q) baseline
mdt.plot_weighted_rdf(scattering.neutron)
```

Use `convention="ashcroft-langreth"` for Ashcroft–Langreth partials and total
normalization. Faber–Ziman partials approach one at high Q; Ashcroft–Langreth
diagonal partials approach one and cross partials approach zero. Pair columns
in weighted structure-factor tables remain additive contributions to
`S(Q) - 1`; the total column is `S(Q)`. `plot_structure_factor` adds one to
each displayed pair curve so every plotted curve uses the same `S(Q)` unit
baseline.

Real-space output is a dimensionless weighted radial distribution rather than
the reduced PDF `G(r)`. Neutron weights are Q-independent, so neutron `g(r)` is
calculated directly from the partial RDFs without a modification function.
X-ray weights depend on Q and are inverse transformed using
`g(r)-1 = [1/(2 pi^2 rho)] integral Q^2[S(Q)-1] sinc(Qr) dQ`. The optional
`xray_window="lorch"` suppresses termination artifacts from the X-ray form
factors; set it to `None` for an unwindowed transform. Both outputs approach
one, and their pair columns add exactly to `Total`.

For imported CSV data without MDT metadata, supply a formula or amount mapping
and number density explicitly:

```python
scattering = mdt.compute_scattering(
    imported_rdfs,
    composition={"Cs": 1, "Li": 13, "Be": 6.5, "F": 27},
    number_density=0.0805,
    q_range=(0.0, 25.0),
    q_step=0.01,
)
```

`compute_scattering_weights` and `compute_partial_structure_factors` expose the
two lower-level stages independently. Scattering uses one vectorized process:
the FFT workload is small once the trajectory has already been reduced to RDFs,
so these functions intentionally do not accept `ncore`.

If an MDT RDF contains exactly one atom of a species, its unavailable self-pair
is completed as an explicitly recorded ideal partial (`g_ii(r)=1`) for total
scattering. Other missing pairs remain errors.

## Multiprocessing contract

Every calculator that exposes `ncore` uses the same meaning:

- `ncore` is the total logical-CPU budget for the function call.
- Serial execution uses one process and limits native scientific-library threads to `ncore`.
- Multiprocessing uses up to `ncore` workers with freud and BLAS limited to one thread each.
- The process and native-thread backends are never multiplied together.
- A temporary CPU-affinity mask enforces the budget on Linux and is restored
  when the calculation finishes.
- Worker count is limited by available work. For example, four frames cannot
  use more than four workers even if `ncore=128`.
- `backend="auto"` uses conservative, calculator-specific workload thresholds
  because process startup can be slower than serial execution for short slices.

On a 256-logical-CPU server, a sufficiently large multiprocessing calculation
with `ncore=128` is restricted to 128 logical CPUs.

Before spawning, the parent builds the MDAnalysis reader random-access index
once. Each worker receives the indexed Universe once through its initializer and
then streams a contiguous frame chunk through its own file handle. No
parent-process coordinate cache is constructed or transferred. DataFrame
calculators record execution details in `result.attrs["execution"]`; named
results record them in `result.metadata["execution"]`.

## Benchmark

```bash
python benchmarks/benchmark_rdf.py simulation.xyz --frames 128 --ncore 32
python benchmarks/benchmark_structural.py simulation.xyz \
  --center Be --ligand F --cutoff 2.35 --frames 64 --ncores 2 4 8 16 24
```

Both benchmarks verify serial/multiprocessing numerical parity before reporting
timings. The structural benchmark limits requested worker counts to 24. To
regress the migration directly against a local copy of the original helper:

```bash
python experiments/compare_original_structural.py simulation.xyz \
  /path/to/MolecularDynamicsTools.py --center Be --ligand F --cutoff 2.35 \
  --second-center Li --second-cutoff 2.75 --frames 4
```

## Repository layout

- `src/molecular_dynamics_tools/`: installable package source
- `tests/`: portable unit and numerical-regression tests
- `benchmarks/`: bounded command-line performance checks
- `experiments/`: reproducible development and method-comparison scripts
- `docs/`: architecture and contributor-facing design notes

Experiment scripts may refer to large local trajectories that are intentionally
kept outside the repository. Their generated CSVs, figures, and metadata belong
under `artifacts/`, which is ignored by Git. Package tests create synthetic
trajectories as needed; an optional local large-trajectory check runs only when
that external file is present.

## Tests

```bash
python -m pytest
python -m ruff check .
python -m build
```

The test suite includes deterministic serial/multiprocessing comparisons and,
when present, a four-frame subset of the large local CsFLiBe trajectory. CI
runs the portable suite and builds both the source distribution and wheel.

## License

This project is licensed under the [BSD 3-Clause License](LICENSE).
