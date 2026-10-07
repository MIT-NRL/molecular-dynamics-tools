"""Bounded periodic graphenic-domain diagnostics following Putman et al.

This is an explicit geometric adaptation of arXiv:2212.06354v1, Section 4,
not a reproduction of unavailable author code or an XRD La measurement.
Neighbor graphs and ring identities can be shared with existing analysis.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque

import numpy as np
from MDAnalysis.lib.distances import minimize_vectors

METHOD = "putman_hexagon_dual_graph_periodic_adaptation_v1"
DOI = "10.1016/j.carbon.2023.03.040"
SOURCE = "https://arxiv.org/html/2212.06354v1"
MAX_EDGE_BATCH = 4096
MAX_SEARCH_STATES = 10_000_000
MAX_RINGS = 1_000_000
MAX_SEARCH_FRONTIER = 100_000


class GraphenicAnalysisError(ValueError):
    """Invalid geometry or an explicitly exceeded bounded-work budget."""


def _canonical_cycle(path):
    return min(
        tuple(sequence[offset:] + sequence[:offset])
        for sequence in (list(path), list(reversed(path)))
        for offset in range(len(path))
    )


def _is_chordless(path, adjacency):
    size = len(path)
    return all(
        path[second] not in adjacency[path[first]]
        for first in range(size)
        for second in range(first + 1, size)
        if second - first not in (1, size - 1)
    )


def _reduce_periodic_basis(lattice):
    """Bounded LLL integer row operations before library triclinic distances.

    The lattice is unchanged, including partially periodic subspaces. This
    avoids MDAnalysis's neighboring-image limitation for highly skew unreduced
    cells. There is no displacement-dependent or unbounded lattice search.
    """
    basis = np.asarray(lattice, dtype=float).copy()
    transform = np.eye(len(basis), dtype=np.int64)

    def orthogonalize():
        orthogonal = basis.copy()
        coefficients = np.zeros((len(basis), len(basis)))
        lengths = np.zeros(len(basis))
        for index in range(len(basis)):
            for previous in range(index):
                coefficients[index, previous] = (
                    basis[index] @ orthogonal[previous] / lengths[previous]
                )
                orthogonal[index] -= coefficients[index, previous] * orthogonal[previous]
            lengths[index] = orthogonal[index] @ orthogonal[index]
            if lengths[index] <= 1e-24:
                raise GraphenicAnalysisError("Ill-conditioned periodic lattice basis")
        return coefficients, lengths

    index = 1
    for _iteration in range(1000):
        if index >= len(basis):
            return basis
        coefficients, lengths = orthogonalize()
        for previous in range(index - 1, -1, -1):
            multiple = int(np.rint(coefficients[index, previous]))
            if abs(multiple) > 1_000_000:
                raise GraphenicAnalysisError("Ill-conditioned periodic lattice basis")
            if multiple:
                transform[index] -= multiple * transform[previous]
                if np.max(np.abs(transform)) > 1_000_000:
                    raise GraphenicAnalysisError("Lattice-reduction coefficient budget exceeded")
                basis = transform @ lattice
                coefficients, lengths = orthogonalize()
        if lengths[index] >= (0.99 - coefficients[index, index - 1] ** 2) * lengths[index - 1]:
            index += 1
        else:
            transform[[index, index - 1]] = transform[[index - 1, index]]
            basis = transform @ lattice
            index = max(1, index - 1)
    raise GraphenicAnalysisError("Lattice-reduction iteration budget exceeded")


def minimum_image_displacements(displacements, cell, *, pbc=True):
    """Use MDAnalysis nearest images after rotating the lattice to its frame.

    Periodic lattice rows are reduced and aligned to the canonical triclinic
    convention. Nonperiodic axes get orthogonal, nonwrapping artificial box
    lengths. Image indices recovered from MDAnalysis are then applied to the
    original double-precision lattice, avoiding float32 box closure drift.
    """
    values = np.asarray(displacements, dtype=float)
    lattice_cell = np.asarray(cell, dtype=float)
    periodic = np.broadcast_to(np.asarray(pbc, dtype=bool), (3,))
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or lattice_cell.shape != (3, 3)
        or not np.isfinite(values).all()
        or not np.isfinite(lattice_cell).all()
    ):
        raise GraphenicAnalysisError("Invalid minimum-image geometry")
    if not periodic.any() or not len(values):
        return values.copy()
    lattice = lattice_cell[periodic]
    singular = np.linalg.svd(lattice, compute_uv=False)
    if singular[-1] <= 1e-12 or singular[0] / singular[-1] > 1e10:
        raise GraphenicAnalysisError("Degenerate periodic lattice basis")
    lattice = _reduce_periodic_basis(lattice)
    first = lattice[0] / np.linalg.norm(lattice[0])
    if len(lattice) >= 2:
        second = lattice[1] - (lattice[1] @ first) * first
    else:
        axis = np.eye(3)[np.argmin(abs(first))]
        second = axis - (axis @ first) * first
    second /= np.linalg.norm(second)
    third = np.cross(first, second)
    if len(lattice) == 3 and lattice[2] @ third < 0:
        third = -third
    rotation = np.column_stack((first, second, third))
    canonical = np.zeros((3, 3))
    canonical[: len(lattice)] = lattice @ rotation
    result = np.empty_like(values)
    for start in range(0, len(values), MAX_EDGE_BATCH):
        raw = values[start : start + MAX_EDGE_BATCH] @ rotation
        if len(lattice) < 3:
            artificial_length = max(1.0, 4 * np.max(np.abs(raw[:, len(lattice) :])) + 1.0)
            for axis in range(len(lattice), 3):
                canonical[axis, axis] = artificial_length
        lengths = np.linalg.norm(canonical, axis=1)
        angles = [
            math.degrees(
                math.acos(np.clip(canonical[a] @ canonical[b] / (lengths[a] * lengths[b]), -1, 1))
            )
            for a, b in ((1, 2), (0, 2), (0, 1))
        ]
        dimensions = np.asarray([*lengths, *angles])
        shortest = minimize_vectors(raw, dimensions)
        images = np.rint((raw - shortest) @ np.linalg.inv(canonical))
        images[:, len(lattice) :] = 0
        result[start : start + MAX_EDGE_BATCH] = (raw - images @ canonical) @ rotation.T
    return result


def _graph(adjacency):
    graph = [set(neighbors) for neighbors in adjacency]
    for atom, neighbors in enumerate(graph):
        if atom in neighbors or any(
            isinstance(neighbor, (bool, np.bool_))
            or not isinstance(neighbor, (int, np.integer))
            or neighbor < 0
            or neighbor >= len(graph)
            or atom not in graph[neighbor]
            for neighbor in neighbors
        ):
            raise GraphenicAnalysisError("Expected a symmetric, loop-free atom graph")
    return graph


def shortest_path_ring_cycles(
    adjacency, maximum_ring_size: int = 8, *, max_search_states: int = MAX_SEARCH_STATES
) -> tuple[tuple[int, ...], ...]:
    """Retain the repository's edge-shortest/chordless cycles with bounded BFS.

    Ring identities, not only counts, are needed by the geometric analysis.
    The criterion is intentionally labeled separately from full polypy/SP
    enumeration. Exceeding the budget raises; it never returns a partial count.
    """
    if (
        isinstance(maximum_ring_size, bool)
        or not isinstance(maximum_ring_size, int)
        or not 3 <= maximum_ring_size <= 12
        or isinstance(max_search_states, bool)
        or not isinstance(max_search_states, int)
        or max_search_states < 1
    ):
        raise GraphenicAnalysisError("Ring size must be 3..12 and work budget positive")
    graph = _graph(adjacency)
    rings = set()
    states = 0
    for start, neighbors in enumerate(graph):
        for target in sorted(neighbor for neighbor in neighbors if neighbor > start):
            queue = deque([[start]])
            shortest_size = None
            while queue:
                path = queue.popleft()
                states += 1
                if states > max_search_states:
                    raise GraphenicAnalysisError("Ring-search state budget exceeded")
                if shortest_size is not None and len(path) + 1 > shortest_size:
                    continue
                current = path[-1]
                for neighbor in sorted(graph[current]):
                    if (current == start and neighbor == target) or neighbor in path:
                        continue
                    candidate = [*path, neighbor]
                    if neighbor == target:
                        if 3 <= len(candidate) <= maximum_ring_size:
                            if shortest_size is None:
                                shortest_size = len(candidate)
                            if len(candidate) == shortest_size and _is_chordless(candidate, graph):
                                rings.add(_canonical_cycle(candidate))
                                if len(rings) > MAX_RINGS:
                                    raise GraphenicAnalysisError("Ring-count budget exceeded")
                    elif len(candidate) < maximum_ring_size:
                        queue.append(candidate)
                        if len(queue) > MAX_SEARCH_FRONTIER:
                            raise GraphenicAnalysisError("Ring-search frontier budget exceeded")
    return tuple(sorted(rings, key=lambda ring: (len(ring), ring)))


def _cycles(ring_cycles, graph, maximum_ring_size):
    result = set()
    for supplied_count, raw in enumerate(ring_cycles, start=1):
        if supplied_count > MAX_RINGS:
            raise GraphenicAnalysisError("Supplied ring-count budget exceeded")
        ring = tuple(raw)
        if (
            not 3 <= len(ring) <= maximum_ring_size
            or len(set(ring)) != len(ring)
            or any(
                isinstance(i, (bool, np.bool_))
                or not isinstance(i, (int, np.integer))
                or not 0 <= i < len(graph)
                for i in ring
            )
            or any(ring[(j + 1) % len(ring)] not in graph[i] for j, i in enumerate(ring))
            or not _is_chordless(list(ring), graph)
        ):
            raise GraphenicAnalysisError("Invalid supplied chordless ring cycle")
        result.add(_canonical_cycle(list(ring)))
        if len(result) > MAX_RINGS:
            raise GraphenicAnalysisError("Ring-count budget exceeded")
    return tuple(sorted(result, key=lambda ring: (len(ring), ring)))


def _edge_vectors(positions, cell, graph, pbc, supplied):
    vectors = {}
    batch = []

    def flush():
        if not batch:
            return
        first, second = np.asarray(batch, dtype=int).T
        displacements = minimum_image_displacements(
            positions[second] - positions[first], cell, pbc=pbc
        )
        for edge, displacement in zip(batch, displacements, strict=True):
            vectors[edge] = displacement
        batch.clear()

    for first, neighbors in enumerate(graph):
        for second in sorted(neighbor for neighbor in neighbors if neighbor > first):
            if supplied is None:
                batch.append((first, second))
                if len(batch) >= MAX_EDGE_BATCH:
                    flush()
            else:
                try:
                    displacement = np.asarray(supplied[first][second], dtype=float)
                except (IndexError, KeyError, TypeError) as error:
                    raise GraphenicAnalysisError("Supplied bond vector is missing") from error
                if displacement.shape != (3,) or not np.isfinite(displacement).all():
                    raise GraphenicAnalysisError("Invalid supplied bond vector")
                vectors[first, second] = displacement
    flush()
    if any(np.linalg.norm(v) <= 1e-10 for v in vectors.values()):
        raise GraphenicAnalysisError("Zero-length bonded edge")
    return vectors


def _step(vectors, first, second):
    return vectors[first, second] if first < second else -vectors[second, first]


def _ring_geometry(ring, positions, vectors):
    coordinates = [np.zeros(3)]
    for first, second in zip(ring[:-1], ring[1:], strict=True):
        coordinates.append(coordinates[-1] + _step(vectors, first, second))
    closure = coordinates[-1] + _step(vectors, ring[-1], ring[0])
    if np.linalg.norm(closure) > 1e-6:
        return None  # A winding cell loop is not a local hexagonal face.
    coordinates = np.asarray(coordinates)
    centered = coordinates - coordinates.mean(axis=0)
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    if singular[1] <= 1e-10:
        return None
    normal = axes[-1]
    area = float(
        abs(np.cross(coordinates, np.roll(coordinates, -1, axis=0)).sum(axis=0) @ normal) / 2
    )
    if area <= 1e-10:
        return None
    return {
        "ring": ring,
        "normal": normal,
        "area": area,
        "planarity_rms": float(np.sqrt(np.mean((centered @ normal) ** 2))),
    }


def _components(nodes, adjacency):
    unseen = set(nodes)
    result = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        queue = deque([first])
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        result.append(component)
    return result


def _junction(slots):
    """Two or more cyclic runs implement the Fig. 4 cata/peri cuts."""
    slots = set(slots)
    return sum(slot in slots and (slot - 1) % 6 not in slots for slot in range(6)) >= 2


def _orientation(normals, weights):
    if not len(normals):
        return None, None
    normals = np.asarray(normals)
    tensor = np.einsum("i,ij,ik->jk", np.asarray(weights), normals, normals)
    tensor /= sum(weights)
    values, directions = np.linalg.eigh(tensor)
    return float(np.clip((3 * values[-1] - 1) / 2, 0, 1)), directions[:, -1]


def _unwrap(rings, positions, cell, vectors, pbc):
    adjacency = defaultdict(set)
    for ring in rings:
        for index, first in enumerate(ring):
            second = ring[(index + 1) % len(ring)]
            adjacency[first].add(second)
            adjacency[second].add(first)
    first = min(adjacency)
    coordinates = {first: np.zeros(3)}
    queue = deque([first])
    wraps = np.zeros(3, dtype=bool)
    inverse = np.linalg.inv(cell) if any(pbc) else None
    while queue:
        current = queue.popleft()
        for neighbor in sorted(adjacency[current]):
            candidate = coordinates[current] + _step(vectors, current, neighbor)
            if neighbor not in coordinates:
                coordinates[neighbor] = candidate
                queue.append(neighbor)
                continue
            residual = candidate - coordinates[neighbor]
            if np.linalg.norm(residual) <= 1e-6:
                continue
            if inverse is None:
                raise GraphenicAnalysisError("Inconsistent nonperiodic bond geometry")
            winding = residual @ inverse
            integer = np.rint(winding)
            if not np.allclose(winding, integer, atol=1e-6) or any(integer[~pbc]):
                raise GraphenicAnalysisError("Bond-loop closure is not a periodic lattice vector")
            wraps |= integer != 0
    if len(coordinates) != len(adjacency):
        raise GraphenicAnalysisError("Disconnected ring component")
    return np.asarray([coordinates[i] for i in sorted(coordinates)]), wraps


def _finite_mean(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return float(np.mean(values)) if values else None


def _distribution(rows, threshold, width):
    finite = [row for row in rows if not row["periodic"]]
    buckets = defaultdict(lambda: [0, 0.0, 0.0])
    for row in finite:
        size = row["sqrt_area_size_A"]
        index = math.floor(size / width)
        buckets[index][0] += 1
        buckets[index][1] += row["area_A2"]
        buckets[index][2] += size**2.5
    total_area = sum(row["area_A2"] for row in finite)
    total_intensity = sum(row["sqrt_area_size_A"] ** 2.5 for row in finite)
    return [
        {
            "threshold_deg": threshold,
            "size_bin_lower_A": index * width,
            "size_bin_upper_A": (index + 1) * width,
            "sqrt_area_size_A": (index + 0.5) * width,
            "crystallite_count": values[0],
            "number_probability": values[0] / len(finite),
            "area_weighted_probability": values[1] / total_area,
            "intensity_proxy_weighted_probability": values[2] / total_intensity,
            "area_weight_A2": values[1],
            "intensity_proxy_weight_A2_5": values[2],
        }
        for index, values in sorted(buckets.items())
    ]


def analyze_graphenic_structure(
    positions,
    cell,
    adjacency,
    *,
    ring_cycles=None,
    bond_vectors=None,
    pbc=True,
    normal_thresholds_deg=(1.0, 2.0),
    maximum_ring_size=8,
    minimum_crystallite_rings=3,
    size_bin_width_A=1.0,
):
    """Return scalar, domain, and full-distribution rows for one saved frame.

    All geometry uses bonded-edge minimum images, including triclinic cells.
    Wrapping components are retained, but never assigned a finite diameter or
    thickness. Area is a sum of fitted ring polygon areas, not a convex hull.
    """
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    periodic = np.broadcast_to(np.asarray(pbc, dtype=bool), (3,))
    graph = _graph(adjacency)
    thresholds = tuple(float(value) for value in normal_thresholds_deg)
    if (
        positions.shape != (len(graph), 3)
        or cell.shape != (3, 3)
        or not np.isfinite(positions).all()
        or not np.isfinite(cell).all()
        or (any(periodic) and abs(np.linalg.det(cell)) < 1e-12)
        or not thresholds
        or any(not math.isfinite(value) or not 0 < value < 90 for value in thresholds)
        or len(set(thresholds)) != len(thresholds)
        or not math.isfinite(size_bin_width_A)
        or size_bin_width_A <= 0
        or isinstance(minimum_crystallite_rings, bool)
        or not isinstance(minimum_crystallite_rings, int)
        or minimum_crystallite_rings < 3
        or isinstance(maximum_ring_size, bool)
        or not isinstance(maximum_ring_size, int)
        or not 3 <= maximum_ring_size <= 12
    ):
        raise GraphenicAnalysisError("Invalid graphenic geometry or analysis settings")
    cycles = (
        shortest_path_ring_cycles(graph, maximum_ring_size)
        if ring_cycles is None
        else _cycles(ring_cycles, graph, maximum_ring_size)
    )
    vectors = _edge_vectors(positions, cell, graph, periodic, bond_vectors)
    hexagons = []
    rejected = 0
    for ring in cycles:
        if len(ring) != 6:
            continue
        geometry = _ring_geometry(ring, positions, vectors)
        if geometry is None:
            rejected += 1
        else:
            hexagons.append(geometry)
    by_edge = defaultdict(list)
    for index, item in enumerate(hexagons):
        ring = item["ring"]
        for slot, first in enumerate(ring):
            second = ring[(slot + 1) % 6]
            by_edge[min(first, second), max(first, second)].append((index, slot))
    # Nonmanifold edges cannot be interpreted as an ordinary hexagonal sheet.
    nonmanifold = {
        index for occupants in by_edge.values() if len(occupants) > 2 for index, _slot in occupants
    }
    dual = [dict() for _ in hexagons]
    neighboring_angles = []
    for occupants in by_edge.values():
        if len(occupants) != 2:
            continue
        (first, first_slot), (second, second_slot) = occupants
        if first in nonmanifold or second in nonmanifold:
            continue
        angle = math.degrees(
            math.acos(np.clip(abs(hexagons[first]["normal"] @ hexagons[second]["normal"]), 0, 1))
        )
        neighboring_angles.append(angle)
        dual[first][second] = (first_slot, angle)
        dual[second][first] = (second_slot, angle)
    eligible = set(range(len(hexagons))) - nonmanifold
    rows, summaries, distributions = [], [], []
    for threshold in thresholds:
        matched = [
            {neighbor for neighbor, (_slot, angle) in edges.items() if angle < threshold}
            for edges in dual
        ]
        junctions = {
            index
            for index in eligible
            if _junction([dual[index][neighbor][0] for neighbor in matched[index]])
        }
        retained = []
        for component in _components(eligible - junctions, matched):
            if len(component) >= minimum_crystallite_rings:
                retained.append(component)
        threshold_rows = []
        selected_atoms = set()
        accepted_rings = []
        for identifier, component in enumerate(retained, start=1):
            items = [hexagons[index] for index in component]
            atoms = {atom for item in items for atom in item["ring"]}
            selected_atoms.update(atoms)
            accepted_rings.extend(items)
            coordinates, wraps = _unwrap(
                [item["ring"] for item in items], positions, cell, vectors, periodic
            )
            is_periodic = bool(wraps.any())
            area = float(sum(item["area"] for item in items))
            order, normal = _orientation([i["normal"] for i in items], [i["area"] for i in items])
            projections = coordinates @ normal
            threshold_rows.append(
                {
                    "threshold_deg": threshold,
                    "crystallite_id": identifier,
                    "ring_count": len(component),
                    "atom_count": len(atoms),
                    "periodic": is_periodic,
                    "wrap_a": bool(wraps[0]),
                    "wrap_b": bool(wraps[1]),
                    "wrap_c": bool(wraps[2]),
                    "area_A2": area,
                    "area_scope": "periodic_cell" if is_periodic else "finite_fragment",
                    "sqrt_area_size_A": None if is_periodic else math.sqrt(area),
                    "equivalent_disk_diameter_A": None
                    if is_periodic
                    else 2 * math.sqrt(area / math.pi),
                    "thickness_A": None if is_periodic else float(np.ptp(projections)),
                    "planarity_rms_A": None if is_periodic else float(np.std(projections)),
                    "normal_nematic_order": order,
                    "mean_ring_planarity_rms_A": float(
                        np.mean([i["planarity_rms"] for i in items])
                    ),
                }
            )
        finite = [row for row in threshold_rows if not row["periodic"]]
        order, _ = _orientation(
            [i["normal"] for i in accepted_rings], [i["area"] for i in accepted_rings]
        )
        summaries.append(
            {
                "threshold_deg": threshold,
                "atom_count": len(graph),
                "total_ring_count": len(cycles),
                "hexagon_count": sum(len(ring) == 6 for ring in cycles),
                "retained_ring_count": len(accepted_rings),
                "graphenic_ring_fraction": len(accepted_rings) / len(cycles) if cycles else None,
                "graphenic_atom_fraction": len(selected_atoms) / len(graph) if graph else None,
                "crystallite_count": len(threshold_rows),
                "finite_crystallite_count": len(finite),
                "periodic_crystallite_count": len(threshold_rows) - len(finite),
                "junction_hexagon_count": len(junctions),
                "mean_sqrt_area_size_A": _finite_mean(finite, "sqrt_area_size_A"),
                "mean_equivalent_disk_diameter_A": _finite_mean(
                    finite, "equivalent_disk_diameter_A"
                ),
                "mean_thickness_A": _finite_mean(finite, "thickness_A"),
                "mean_planarity_rms_A": _finite_mean(finite, "planarity_rms_A"),
                "normal_nematic_order": order,
            }
        )
        rows.extend(threshold_rows)
        distributions.extend(_distribution(threshold_rows, threshold, size_bin_width_A))
    all_order, _ = _orientation([i["normal"] for i in hexagons], [i["area"] for i in hexagons])
    return {
        "metadata": {
            "method": METHOD,
            "doi": DOI,
            "source": SOURCE,
            "source_version": "author_preprint_v1",
            "interpretation": "Geometric adaptation; finite sizes are not XRD La(10)",
            "ring_convention": "repository_edge_shortest_chordless",
            "maximum_ring_size": maximum_ring_size,
            "normal_thresholds_deg": list(thresholds),
            "minimum_crystallite_rings": minimum_crystallite_rings,
            "junction_rule": "simultaneous_removal_of_multiple_cyclic_dual_edge_runs",
            "normal_method": "six_atom_least_squares_plane_unsigned_normal",
            "component_normal_method": "area_weighted_unsigned_normal_tensor_principal_axis",
            "normal_angle_comparison": "strictly_less_than_threshold",
            "hexagon_regularity_filter": "topology_and_plane_connectivity; no_shape_cutoff",
            "area_method": "sum_of_local_projected_hexagon_polygon_areas",
            "periodicity_method": (
                "bond_loop_lattice_winding; finite_sizes_missing_for_wrapping_domains"
            ),
            "size_bin_width_A": size_bin_width_A,
            "distribution_scope": "finite_fragments_only; occupied_bins_including_all_tails",
            "intensity_weight": "sqrt_area_size_A**2.5; exploratory_paper_proxy_not_fitted_XRD",
            "rejected_nonlocal_or_degenerate_hexagons": rejected,
            "rejected_nonmanifold_hexagons": len(nonmanifold),
            "finite_component_statistics": "arithmetic_domain_means; descriptive_not_uncertainty",
        },
        "threshold_summaries": summaries,
        "crystallites": rows,
        "size_distributions": distributions,
        "ring_counts": dict(sorted(Counter(map(len, cycles)).items())),
        "ring_geometry_summary": {
            "normal_nematic_order": all_order,
            "mean_hexagon_planarity_rms_A": float(np.mean([i["planarity_rms"] for i in hexagons]))
            if hexagons
            else None,
            "mean_adjacent_normal_misorientation_deg": float(np.mean(neighboring_angles))
            if neighboring_angles
            else None,
            "std_adjacent_normal_misorientation_deg": float(np.std(neighboring_angles))
            if neighboring_angles
            else None,
            "nonhexagonal_ring_fraction": sum(len(ring) != 6 for ring in cycles) / len(cycles)
            if cycles
            else None,
        },
    }
