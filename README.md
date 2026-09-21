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

Create an environment with the package and its scientific dependencies:

```bash
conda create -n mdt -c seanfayfar -c conda-forge molecular-dynamics-tools
conda activate mdt
```

### Pip from GitHub

Install the current repository directly into an existing Python environment:

```bash
python -m pip install "git+https://github.com/MIT-NRL/molecular-dynamics-tools.git"
```

### Clone the repository

Clone the source when you want a local copy or plan to contribute:

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

`load_trajectory` opens the simulation, and `compute_rdfs` calculates pair
distributions from its frames. The context manager closes the reader afterward.

```python
import molecular_dynamics_tools as mdt

with mdt.load_trajectory("simulation.xyz") as trajectory:
    rdfs = mdt.compute_rdfs(
        trajectory,
        step=0.02,
        ncore=8,
    )
```

`load_trajectory` accepts a single file, a topology/coordinate pair, or an
existing MDAnalysis `Universe`. By default it uses every frame and automatically
normalizes XYZ species order when needed.

```python
trajectory = mdt.load_trajectory(
    "simulation.xyz",
    frames=None,
    normalize_species_order="auto",
)
trajectory = mdt.load_trajectory("topology.psf", "production.dcd")
trajectory = mdt.load_trajectory(universe)
```

Use `frames` at load time to select part of a trajectory; `box` can override
missing or stored cell dimensions.

```python
trajectory = mdt.load_trajectory("simulation.xyz", frames=slice(0, 1000, 10))
trajectory = mdt.load_trajectory("plain.xyz", box=10.0)
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

`compute_rdfs` bins pair distances to estimate partial `g_ij(r)`. The selected
`pairs` are optional (all unique pairs by default); `step` sets bin width, or
omitting both `step` and `bins` uses 800 bins.

```python
rdfs = mdt.compute_rdfs(
    trajectory,
    pairs=[("Na", "Na"), ("Na", "Cl"), ("Cl", "Cl")],
    step=0.02,
    r_range=(0.0, None),
    ncore=8,
    backend="auto",
    show_progress=False,
)
```

`compute_spectral_rdfs` estimates the same `g_ij(r)` with a smooth cosine
expansion. It defaults to 60 modes; `modes="auto"` selects a cutoff per pair,
while `step` controls only the returned radial grid.

```python
spectral_rdfs = mdt.compute_spectral_rdfs(
    trajectory,
    modes="auto",
    step=0.025,
    r_range=(0.0, None),
    ncore=8,
    backend="auto",
)
```

Both RDFs are normalized by ideal pair density, so `g(r)` approaches 1 for a
spatially uniform system; the histogram also divides counts by shell volume.
The spectral method follows the cosine expansion of
[Patrone and Rosch, J. Chem. Phys. 146, 094107 (2017)](https://doi.org/10.1063/1.4977516).
The optional automatic mode selector is an MDT heuristic built on that
expansion. Results contain `r` in distance units followed by columns such as
`Na-Cl`; spectral mode selections and other settings are retained in `.attrs`.

## Scattering

`compute_scattering` transforms partial RDFs into neutron and X-ray structure
factors `S(Q)` and probe-weighted radial distributions `g(r)`.

```python
scattering = mdt.compute_scattering(
    rdfs,
    probes=("neutron", "xray"),
    convention="faber-ziman",
    q_range=(0.0, 25.0),
    q_step=0.01,
    xray_window="lorch",
)

scattering.partial_structure_factors
scattering.neutron.structure_factor
scattering.neutron.weighted_rdf
scattering.xray.structure_factor
scattering.xray.weighted_rdf
```

For the default Faber–Ziman convention, the relations follow
[Fayfar et al., *PRX Energy* 3, 013001 (2024)](https://doi.org/10.1103/PRXEnergy.3.013001).
Here $c_\alpha$ is the atomic fraction, $\rho_0$ the total number density, and
$a_\alpha^{(p)}(Q)$ the scattering amplitude for probe $p$: the coherent
scattering length $b_\alpha$ for neutrons or the form factor $f_\alpha(Q)$ for
X-rays. The symbol $\delta_{\alpha\beta}$ is the Kronecker delta.

$$
S_{\alpha\beta}(Q)-1
=4\pi\rho_0\int_0^\infty r^2[g_{\alpha\beta}(r)-1]
\frac{\sin(Qr)}{Qr}\,dr.
$$

$$
w_{\alpha\beta}^{(p)}(Q)
=\frac{(2-\delta_{\alpha\beta})c_\alpha c_\beta
a_\alpha^{(p)}(Q)a_\beta^{(p)}(Q)}
{\left[\sum_\gamma c_\gamma a_\gamma^{(p)}(Q)\right]^2},
\qquad
S^{(p)}(Q)=1+\sum_{\alpha\le\beta}w_{\alpha\beta}^{(p)}(Q)
[S_{\alpha\beta}(Q)-1].
$$

Neutron weights are independent of $Q$, so the neutron weighted RDF is a direct
combination of partial RDFs. X-ray weights vary with $Q$, so the X-ray weighted
RDF is obtained by an inverse transform:

$$
g^{(n)}(r)=\sum_{\alpha\le\beta}w_{\alpha\beta}^{(n)}g_{\alpha\beta}(r),
\qquad
g^{(x)}(r)-1=\frac{1}{2\pi^2\rho_0}
\int_{Q_{\min}}^{Q_{\max}}Q^2[S^{(x)}(Q)-1]M(Q)
\frac{\sin(Qr)}{Qr}\,dQ.
$$

$M(Q)=1$ without a window; for `xray_window="lorch"`,
$M(Q)=\sin(\pi Q/Q_{\max})/(\pi Q/Q_{\max})$.
The calculation evaluates the forward transform over the available RDF range.

The default Faber-Ziman convention can be changed to Ashcroft-Langreth.
Pair columns in `structure_factor` are contributions to `S(Q) - 1`, and their
sum plus the unit baseline gives `Total`. Pair columns in `weighted_rdf` are
additive contributions to a dimensionless weighted `g(r)` whose total approaches
1; this is not the reduced PDF `G(r)`. Results are grouped in a
`ScatteringResult`, with common partial tables and separate `neutron` and
`xray` result tables.

`plot_structure_factor` and `plot_weighted_rdf` display a probe's total and
pair contributions when the `plot` extra is installed.

```python
mdt.plot_structure_factor(scattering.neutron)
mdt.plot_weighted_rdf(scattering.neutron)
```

Neutron weights use coherent scattering lengths from `periodictable`, including
user-specified isotope mixtures. X-ray weights use its Q-dependent neutral or
ionic form factors. The shown probes, convention, and X-ray window are defaults;
the Q grid is chosen explicitly. RDF metadata supplies composition and number
density by default, but both can be supplied. If a species has only one atom
and its self-RDF is unavailable, scattering warns before substituting an ideal
`g(r)=1` self-pair to complete the total.

## Coordination numbers

`compute_coordination` counts neighbors inside each center-neighbor distance
cutoff and returns a coordination-number distribution. Definitions have the
form `(center, neighbor, r_max)` or `(center, neighbor, r_min, r_max)`.

```python
coordination = mdt.compute_coordination(
    trajectory,
    [("Na", "Cl", 3.2), ("Cl", "Na", 3.2)],
    max_coordination=None,
    ncore=8,
    backend="auto",
)
```

`compute_rad_coordination` identifies each center's RAD coordination shell
without a distance cutoff. Directed RAD-closed shells are the default; set
`bond_mode="mutual"` to retain only reciprocal contacts.

```python
rad_coordination = mdt.compute_rad_coordination(
    trajectory,
    [("Na", "Cl")],
    bond_mode="directed",
    variant="closed",
    ncore=8,
)
```

`summarize_coordination` reduces either probability distribution to its mean,
variance, and standard deviation.

```python
summary = mdt.summarize_coordination(coordination)
```

Both calculators return `coordination` followed by one probability column per
definition. Each column is normalized over center-atom/frame samples and sums
to 1 unless `max_coordination` places probability in the reported overflow.
The implemented RAD-closed construction follows
[Higham and Henchman, J. Chem. Phys. 145, 084108 (2016)](https://doi.org/10.1063/1.4961439).

## Bond angles

`compute_bond_angles` measures first-center-third angles when both bonds meet
their cutoffs. Each definition is
`(first, center, third, first_center_max, center_third_max)`.

```python
angles = mdt.compute_bond_angles(
    trajectory,
    [("F", "Be", "F", 2.35, 2.35)],
    bins=180,
    angle_range=(0.0, 180.0),
    ncore=8,
)
```

The result contains `angle` in degrees and one probability-density column per
definition; each column integrates to 1 over the angle grid.

## RAD environments

`compute_rad_environments` retains each center's RAD contacts and distances
over the trajectory, enabling species-resolved coordination and speciation.
Neighbor species default to all species in the trajectory.

```python
environments = mdt.compute_rad_environments(
    trajectory,
    center_species="Na",
    neighbor_species=None,
    bond_mode="directed",
    variant="closed",
    frames=None,
    ncore=8,
)

contacts = environments.contacts
per_center = environments.environments
na_cl = environments.contacts_between("Na", "Cl")
speciation = environments.speciation_distribution(("Cl",))
```

`contacts` is a long table with one row per center-neighbor contact,
including atom indices, species, periodic minimum-image distance, and mutual
status. `environments` has one row per center, frame, and neighbor species with
coordination plus minimum, mean, and maximum distance. Speciation probabilities
are normalized over center-atom/frame samples. Occupancy and lifetime summaries
assume atom indices identify the same physical atoms throughout the trajectory.
This calculation uses the same default RAD-closed method cited above.

## Clustering

`compute_by_distance` links selected atoms within a cutoff and measures the
sizes of connected components. `count_species` changes the size and sampling
basis to the chosen central species.

```python
distance_clusters = mdt.clustering.compute_by_distance(
    trajectory,
    species=("Na", "Cl"),
    cutoff=3.2,
    r_min=0.0,
    count_species="Na",
    include_isolated=False,
    ncore=8,
    backend="auto",
)
distance_clusters.cluster_distribution
```

`compute_by_shared_neighbors` links centers that share coordinating neighbors
and reports connected, corner-, edge-, and face-sharing networks by default.

```python
shared_neighbors = mdt.clustering.compute_by_shared_neighbors(
    trajectory,
    centers="Be",
    neighbors="F",
    cutoff=2.35,
    connections=("connected", "corner", "edge", "face"),
    min_shared_neighbors=1,
    include_isolated=False,
    ncore=8,
    backend="auto",
)

shared_neighbors.cluster_distribution
shared_neighbors.sharing_distribution
shared_neighbors.percolation_summary
```

For distance clustering, `r_min < distance <= cutoff` defines an edge. Without
`count_species`, size counts all selected atoms and each component is sampled
once; with it, size counts that species and each counted atom samples its
cluster. The result has `cluster_size`, `probability`, and metadata.

For shared-neighbor clustering, one shared neighbor means corner sharing, two
means edge sharing, and three or more means face sharing; `connected` combines
them. The result includes normalized cluster and sharing distributions,
per-frame statistics, and periodic percolation summaries. `connections`
selects returned networks; `min_shared_neighbors` independently sets the
percolation edge rule. By default, both methods omit isolated size-one
components. Percolation detects connections that wrap the periodic box.

## Result caching

`AnalysisCache.get_or_compute` reuses an analysis result when its source files
and scientific parameters match a prior run. It stores results, not trajectory
coordinates.

```python
cache = mdt.cache.AnalysisCache(
    "analysis-cache",
    fingerprint="stat",
    storage="auto",
    mode="use",
)

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

Functions that accept `ncore` use it as the total logical-CPU budget; the
default is `ncore=1`, while the examples use 8.
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
