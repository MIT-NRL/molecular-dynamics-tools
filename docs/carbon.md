# Reusable carbon geometry

For diffraction peak fitting, Fig. 8-style size distributions and bounded
Fig. 3-style graphene calibrations, see [carbon diffraction](carbon_diffraction.md).

`analyze_graphenic_structure` analyzes one full saved frame using a bonded
graph supplied by the caller. It returns ordinary dictionaries and lists of
rows; it does not load a trajectory, select a bond cutoff, or schedule workers.
Call it for each selected frame while retaining only the results you need.

## Inputs and reuse

```python
import molecular_dynamics_tools as mdt

# positions: finite N×3 coordinates in Å
# cell: three lattice vectors as rows, in Å
# adjacency: symmetric, loop-free bonded neighbor sets/lists, indexed 0..N-1
cycles = mdt.shortest_path_ring_cycles(adjacency, maximum_ring_size=8)
result = mdt.analyze_graphenic_structure(
    positions,
    cell,
    adjacency,
    ring_cycles=cycles,
    pbc=(True, True, True),
    normal_thresholds_deg=(1.0, 2.0),
    maximum_ring_size=8,
    minimum_crystallite_rings=3,
    size_bin_width_A=1.0,
)
```

Freeze and record the caller's bond cutoff alongside these results. Atom
indices must refer to the same full-frame coordinates and graph. `pbc` accepts
a Boolean or three axis flags. Stored atom images, cell rotation, and
translation do not alter the periodic geometric measurements.

If an existing analysis already calculated bond vectors, pass
`bond_vectors=vectors`, where `vectors[i][j]` is the periodic displacement from
atom `i` to bonded atom `j`. Supplied ring cycles must be valid chordless
graph cycles within the recorded ring-size range; equivalent cycle orders are
deduplicated. Supplying cycles and vectors avoids repeating their searches.

## Method conventions

The implementation is a **periodic geometric adaptation** of
[Putman et al., Carbon 209 (2023) 117965](https://doi.org/10.1016/j.carbon.2023.03.040),
based on the full [author preprint, Section 4 and Fig. 4](https://arxiv.org/html/2212.06354v1).
It groups edge-adjacent hexagons by normal misorientation, removes junction
hexagons, and discards components smaller than three rings. The author code
was unavailable; implementation choices remain explicit in `metadata`.

- `shortest_path_ring_cycles` returns unique edge-shortest, chordless cycles.
  This repository convention differs from claiming full polypy/Franzblau
  enumeration. The default ring fraction denominator covers selected sizes
  3–8, including nonhexagonal rings.
- Six-atom least-squares planes give unsigned hexagon normals. Adjacent rings
  join only when their unsigned normal angle is **strictly below** the selected
  threshold. The default 1° and 2° populations are evaluated separately.
  Local comparison allows curvature to accumulate across a domain.
- Junction cuts simultaneously remove hexagons whose occupied shared-edge
  slots form multiple disjoint runs around the six-membered cycle. This is the
  documented interpretation of the paper's cata/peri rules. No numeric hexagon
  regularity or in-plane rotation cutoff is inferred.
- Locally winding or degenerate rings and nonmanifold shared edges are excluded
  and counted in metadata. Bond-loop lattice winding identifies periodic
  domains. A wrapping sheet is retained as a domain and receives no finite
  size, diameter, thickness, or whole-domain planarity value.

This is geometric coordination and ring analysis; it does not determine
electronic hybridization. The chosen bond graph and angular threshold affect
the resulting populations.

## Results and units

| Result key | Contents |
| --- | --- |
| `metadata` | Source, adaptation, ring convention, settings, exclusions, weighting, and periodicity conventions |
| `threshold_summaries` | One scalar row per angular threshold: ring/atom fractions, domain counts, finite-domain arithmetic mean sizes, thickness, planarity, and orientation order |
| `crystallites` | Per-domain rows with ring/atom counts, `wrap_a/b/c`, periodic flags, scoped area, and geometric diagnostics |
| `size_distributions` | All occupied finite-domain size bins, raw counts/weights, and number/area/intensity-proxy probabilities |
| `ring_counts` | Ring size to unique cycle count |
| `ring_geometry_summary` | All valid hexagons' orientation order and local planarity; adjacent-normal mean/spread; nonhexagonal fraction |

`area_A2` sums fitted ring polygon areas. Its scope is a finite fragment or one
periodic cell. Finite `sqrt_area_size_A` is the square root of that area;
`equivalent_disk_diameter_A` is `2*sqrt(area/pi)`. They are separate geometric
conventions and are **not diffraction `La(10)`**. No XRD peak fit is performed.
Missing finite metrics are `None`, including for ideal periodic graphene or
graphite; they are not zero-size crystallites.

The area-weighted unsigned-normal tensor defines orientation order
`(3*lambda_max - 1)/2`: 1 for aligned normals, with 0 the isotropic limiting
population. Its principal axis defines finite-domain thickness and RMS height.
Local ring planarity and neighboring-normal angles remain valid for wrapping
sheets. Population spread is descriptive, not an independent error estimate.

Distribution `sqrt_area_size_A` values are bin midpoints. Number probability,
area-weighted probability, and the exploratory `size**2.5` intensity proxy are
separate, normalized probability masses over finite domains. Bins with no
domains are omitted, but no occupied tails are dropped. Raw counts and weights
allow pooling across frames without averaging normalized curves. The proxy is
inspired by the paper and does not represent a fitted scattering intensity.

## Bounded work and periodic geometry

There is no all-atom distance matrix or atom thinning. Bonded-edge distances
run in batches of at most 4,096. Bounded integer LLL lattice reduction preserves
the periodic lattice, including partially periodic subspaces, before rotation
into MDAnalysis's canonical triclinic convention. This prevents an acute,
unreduced cell from exceeding the library's neighboring-image search range.
Recovered image indices are reapplied to the double-precision lattice to avoid
float32 box-closure drift.

The ring search defaults to at most 10,000,000 visited states, a frontier of
100,000 paths, and 1,000,000 rings; `max_search_states` can be set when calling
`shortest_path_ring_cycles`. Ring sizes are restricted to 3–12. Lattice
reduction has explicit iteration/coefficient bounds and rejects ill-conditioned
bases. Invalid geometry or exceeded budgets raise `GraphenicAnalysisError`
rather than returning partial counts. Memory scales with supplied graph edges,
ring identities, and retained result rows.
