# Changelog

This project follows semantic versioning. User-visible changes will be recorded
here as the package is developed.

## 0.1.0 - Unreleased

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
- Add a bounded real-trajectory RDF benchmark with numerical parity checks.
- Store composition, pair, and mean number-density metadata on RDF results.
- Add `ScatteringComposition` with isotope mixtures and ionic X-ray form factors.
- Add Faber–Ziman and Ashcroft–Langreth partial structure factors and weights.
- Add single-process FFT neutron/X-ray structure factors and weighted reduced PDFs.
- Keep unweighted partial RDFs distinct from scattering-weighted PDF outputs.
