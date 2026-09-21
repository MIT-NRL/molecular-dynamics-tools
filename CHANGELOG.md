# Changelog

This project follows semantic versioning. User-visible changes will be recorded
here as the package is developed.

## 0.2.0 - 2026-09-21

- Add atom-resolved RAD environments with neighbor distances, speciation,
  occupancy, and lifetime summaries.
- Add provenance-aware result caching with checksummed `.mdtc` archives and
  JSON or Parquet table storage; preserve non-JSON DataFrame attributes in
  Parquet round trips.
- Unify clustering under `mdt.clustering.compute_by_distance` and
  `compute_by_shared_neighbors`, including sharing categories and periodic
  percolation.
- Fix extended-XYZ species-order and column-order loading, and validate RDF
  radii against periodic box limits before neighbor queries.
- Make progress displays opt-in; add portable CI checks, a Conda recipe, and
  concise public installation and calculation documentation.
- Keep workstation-specific experiment scripts out of the repository and make
  optional tests portable.
- Report creation or reuse of normalized XYZ caches with their locations and
  sizes, and write new cache files atomically.
- Warn when scattering replaces a missing singleton self-correlation with an
  ideal `g(r)=1` placeholder.

## 0.1.0 - Initial development (not published)

- Create the standalone package scaffold.
- Define initial module boundaries and development tooling.
- License the project under the BSD 3-Clause License.
- Add a `Trajectory` abstraction backed by a public MDAnalysis `Universe`.
- Accept single files, topology/coordinate pairs, and existing Universes.
- Supplement the MDAnalysis XYZ reader with extended-XYZ lattice and origin metadata.
- Derive stable atom labels from Universe topology attributes.
- Add bounded source-frame imports for testing large trajectories.
- Add `compute_rdfs` with serial and multiprocessing execution.
- Add streaming `compute_spectral_rdfs` with global or pair-specific fixed modes.
- Add pilot-elbow automatic spectral modes with reusable pilot coefficients and diagnostics.
- Support mutually exclusive RDF `bins` and exact radial `step` controls.
- Standardize `ncore` as the total logical-CPU budget.
- Enforce that budget across freud and BLAS thread pools to prevent nested oversubscription.
- Transfer an indexed Universe once per worker without coordinate precaching.
- Add one reusable streaming frame executor for coordination, angle, and
  clustering calculations.
- Add cutoff and directed/mutual RAD coordination distributions and summaries.
- Add multi-definition bond-angle probability densities.
- Add workload-aware automatic backend selection based on bounded local timing.
- Add trajectory context-manager and explicit reader-close support.
- Add a bounded real-trajectory RDF benchmark with numerical parity checks.
- Add bounded structural backend benchmarking and original-helper regression tools.
- Store composition, pair, and mean number-density metadata on RDF results.
- Add `ScatteringComposition` with isotope mixtures and ionic X-ray form factors.
- Add Faber–Ziman and Ashcroft–Langreth partial structure factors and weights.
- Add single-process FFT neutron/X-ray structure factors and weighted radial distributions.
- Keep unweighted partial RDFs distinct from scattering-weighted PDF outputs.
