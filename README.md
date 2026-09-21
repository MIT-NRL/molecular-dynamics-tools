# Molecular Dynamics Tools

`molecular-dynamics-tools` provides reusable trajectory and structural analyses
for molecular-dynamics simulations. It uses MDAnalysis for trajectory I/O and
supports serial and multiprocessing execution.

Main features:

- Histogram and spectral radial distribution functions
- Neutron and X-ray scattering calculations
- Cutoff and relative-angular-distance (RAD) coordination
- Atom-resolved RAD environments and neighbor distances
- Bond-angle distributions
- Distance and shared-neighbor clustering
- Periodic percolation analysis
- Reusable, provenance-aware result caching

## Installation

### Conda (recommended)

```bash
conda create -n mdt -c seanfayfar -c conda-forge molecular-dynamics-tools
conda activate mdt
```

### Pip from GitHub

```bash
python -m pip install "git+https://github.com/MIT-NRL/molecular-dynamics-tools.git"
```

### Clone the repository

```bash
git clone https://github.com/MIT-NRL/molecular-dynamics-tools.git
cd molecular-dynamics-tools
python -m pip install .
```

For development:

```bash
python -m pip install -e ".[dev]"
```

Optional extras are `plot`, `progress`, and `cache` (Parquet serialization via
PyArrow).

## Quick start

```python
import molecular_dynamics_tools as mdt

with mdt.load_trajectory("simulation.xyz") as trajectory:
    rdfs = mdt.compute_rdfs(
        trajectory,
        step=0.02,
        ncore=8,
    )
```

`load_trajectory` accepts a single trajectory, a topology/coordinate pair, or
an existing MDAnalysis `Universe`:

```python
trajectory = mdt.load_trajectory("simulation.xyz")
trajectory = mdt.load_trajectory("topology.psf", "production.dcd")
trajectory = mdt.load_trajectory(universe)
```

Use `frames` to select part of a trajectory:

```python
trajectory = mdt.load_trajectory("simulation.xyz", frames=slice(0, 1000, 10))
```

If an XYZ needs species-order or column normalization, loading creates or
reuses a normalized copy of the full file in the system temporary directory
and prints its location and size. The original file is unchanged.

## Backends and result conventions

- **MDAnalysis** reads trajectory and topology formats and supplies atom
  selections. It is the trajectory I/O layer, not a separate analysis method.
- **freud** supplies periodic simulation boxes, minimum-image distances, and
  neighbor searches used by the structural calculations.
- **NumPy, pandas, and SciPy** provide numerical operations, labeled result
  tables, and scattering transforms.
- **periodictable** supplies atomic masses, coherent neutron scattering
  lengths, and neutral or ionic X-ray form factors. Isotope mixtures and ion
  charges can be overridden in the scattering inputs.

Table-based calculations return pandas `DataFrame` objects: the first column
is the independent coordinate and the remaining columns are labeled results.
Calculation settings and execution details are stored in `DataFrame.attrs`.
More structured analyses return result dataclasses containing named tables and
a `metadata` dictionary.

## Radial distribution functions

```python
rdfs = mdt.compute_rdfs(
    trajectory,
    pairs=[("Na", "Na"), ("Na", "Cl"), ("Cl", "Cl")],
    step=0.02,
    r_range=(0.0, None),
    ncore=8,
)

spectral_rdfs = mdt.compute_spectral_rdfs(
    trajectory,
    modes="auto",
    step=0.02,
    ncore=8,
)
```

Both functions calculate partial pair distribution functions, `g_ij(r)`.
Pair counts are divided by spherical-shell volume and the ideal pair density,
so `g(r)` approaches 1 for a spatially uniform system. `compute_rdfs` uses
distance histograms; `compute_spectral_rdfs` represents the same normalized
quantity with the spectral Monte Carlo cosine expansion of
[Patrone and Rosch, J. Chem. Phys. 146, 094107 (2017)](https://doi.org/10.1063/1.4977516).
The optional automatic mode selector is an MDT heuristic built on that
expansion. Results contain `r` in distance units followed by columns such as
`Na-Cl`; spectral mode selections and other settings are retained in `.attrs`.

## Scattering

RDF metadata supplies the composition and number density when the RDFs were
calculated by this package.

```python
scattering = mdt.compute_scattering(
    rdfs,
    q_range=(0.0, 25.0),
    q_step=0.01,
)

scattering.partial_structure_factors
scattering.neutron.structure_factor
scattering.neutron.weighted_rdf
scattering.xray.structure_factor
scattering.xray.weighted_rdf
```

`compute_scattering` transforms the partial RDFs into partial structure factors
and combines them using composition-dependent neutron or X-ray weights. The
default is the Faber-Ziman convention; Ashcroft-Langreth is also available.
Pair columns in `structure_factor` are contributions to `S(Q) - 1`, and their
sum plus the unit baseline gives `Total`. Pair columns in `weighted_rdf` are
additive contributions to a dimensionless weighted `g(r)` whose total approaches
1; this is not the reduced PDF `G(r)`. Results are grouped in a
`ScatteringResult`, with common partial tables and separate `neutron` and
`xray` result tables.

Neutron weights use coherent scattering lengths from `periodictable`, including
user-specified isotope mixtures. X-ray weights use its Q-dependent neutral or
ionic form factors. RDF metadata normally provides composition and number
density, but both can be supplied explicitly.
If a species has only one atom and its self-RDF is unavailable, scattering
warns before substituting an ideal `g(r)=1` self-pair to complete the total.

With the `plot` extra installed:

```python
mdt.plot_structure_factor(scattering.neutron)
mdt.plot_weighted_rdf(scattering.neutron)
```

## Coordination numbers

```python
coordination = mdt.compute_coordination(
    trajectory,
    [("Na", "Cl", 3.2), ("Cl", "Na", 3.2)],
    ncore=8,
)

rad_coordination = mdt.compute_rad_coordination(
    trajectory,
    [("Na", "Cl")],
    bond_mode="directed",
    ncore=8,
)

summary = mdt.summarize_coordination(coordination)
```

`compute_coordination` counts neighbors inside each specified radial interval.
`compute_rad_coordination` instead finds the parameter-free RAD shell; directed
shells may be restricted to mutually selected neighbors with
`bond_mode="mutual"`. The default and currently supported variant is the
RAD-closed construction described by
[Higham and Henchman, J. Chem. Phys. 145, 084108 (2016)](https://doi.org/10.1063/1.4961439).
Both functions return `coordination` followed by one probability column per
definition. Each column is normalized over all center-atom/frame samples and
sums to 1 unless `max_coordination` places probability in the reported
overflow. `summarize_coordination` returns the mean, variance, and standard
deviation of each distribution.

## Bond angles

```python
angles = mdt.compute_bond_angles(
    trajectory,
    [("F", "Be", "F", 2.35, 2.35)],
    bins=180,
    ncore=8,
)
```

`compute_bond_angles` measures first-center-third angles whose two bonds meet
the supplied cutoffs. It returns `angle` in degrees and one probability-density
column per definition; each column integrates to 1 over the angle grid.

## RAD environments

Neighbor species default to all species in the trajectory.

```python
environments = mdt.compute_rad_environments(
    trajectory,
    center_species="Na",
    ncore=8,
)

contacts = environments.contacts
per_center = environments.environments
na_cl = environments.contacts_between("Na", "Cl")
speciation = environments.speciation_distribution(("Cl",))
```

RAD environments retain atom-level detail rather than only a coordination
histogram. `contacts` is a long table with one row per center-neighbor contact,
including atom indices, species, periodic minimum-image distance, and mutual
status. `environments` has one row per center, frame, and neighbor species with
coordination plus minimum, mean, and maximum distance. Speciation probabilities
are normalized over center-atom/frame samples. Occupancy and lifetime summaries
assume atom indices identify the same physical atoms throughout the trajectory.
This calculation uses the same default RAD-closed method cited above.

## Clustering

```python
distance_clusters = mdt.clustering.compute_by_distance(
    trajectory,
    species=("Na", "Cl"),
    cutoff=3.2,
    count_species="Na",
    ncore=8,
)

shared_neighbors = mdt.clustering.compute_by_shared_neighbors(
    trajectory,
    centers="Be",
    neighbors="F",
    cutoff=2.35,
    ncore=8,
)

distance_clusters.cluster_distribution
shared_neighbors.cluster_distribution
shared_neighbors.sharing_distribution
shared_neighbors.percolation_summary
```

The two clustering methods differ only in how graph edges are defined:

- **Distance clustering** connects selected atoms when
  `r_min < distance <= cutoff` and finds connected components. With no
  `count_species`, cluster size counts all selected atoms and each component is
  one sample. With `count_species`, size counts only that species and the
  probability distribution is sampled once per counted atom, which gives a
  central-atom-normalized result. The result contains `cluster_size`,
  `probability`, and metadata.
- **Shared-neighbor clustering** first finds center-neighbor contacts, then
  connects centers that share coordinating neighbors. One shared neighbor is
  corner sharing, two is edge sharing, and three or more is face sharing;
  `connected` combines all three. One calculation returns normalized cluster
  and sharing distributions, per-frame network statistics, finite-component
  distributions, and periodic percolation summaries. `connections` selects
  which categorized networks are returned, while `min_shared_neighbors`
  independently sets the percolation edge rule.

`include_isolated=False` excludes size-one components in both methods. Periodic
percolation is detected from graph connections that wrap the simulation box.

## Result caching

```python
cache = mdt.cache.AnalysisCache("analysis-cache")

rdfs = cache.get_or_compute(
    mdt.compute_rdfs,
    trajectory,
    step=0.02,
    ncore=8,
)
```

The default `fingerprint="stat"` is fast for local work. Use
`fingerprint="sha256"` when identical source files should share entries across
locations. Cache modes are `use`, `refresh`, `read_only`, and `off`. Cached
tables preserve their result type, metadata, and `DataFrame.attrs`.
Cache manifests can contain source paths and analysis parameters; keep cache
entries out of public repositories.

## Parallel execution

Functions that accept `ncore` use it as the total logical-CPU budget.
`backend="auto"` selects serial or multiprocessing execution. Workers stream
trajectory frames from disk without precaching coordinates.

## Development

```bash
python -m pytest
python -m ruff check .
python -m build
```

See [docs/architecture.md](docs/architecture.md) for internal module boundaries
and design conventions.

## License

This project is licensed under the [BSD 3-Clause License](LICENSE).
