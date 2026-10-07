"""Geometric references and periodic invariants for reusable carbon analysis."""

import math
from itertools import product

import numpy as np
import pytest

from molecular_dynamics_tools import carbon


def _hex_patch(centers, bond=1.42, shift=(0, 0, 0)):
    vertices, rings, mapping = [], [], {}
    basis = np.asarray([[1.5 * bond, math.sqrt(3) * bond / 2], [0, math.sqrt(3) * bond]])
    for center in centers:
        xy = np.asarray(center) @ basis
        ring = []
        for angle in np.arange(6) * math.pi / 3:
            point = np.array([xy[0] + bond * math.cos(angle), xy[1] + bond * math.sin(angle), 0])
            key = tuple(np.round(point, 10))
            if key not in mapping:
                mapping[key] = len(vertices)
                vertices.append(point + np.asarray(shift))
            ring.append(mapping[key])
        rings.append(tuple(ring))
    graph = [set() for _ in vertices]
    for ring in rings:
        for index, first in enumerate(ring):
            second = ring[(index + 1) % 6]
            graph[first].add(second)
            graph[second].add(first)
    return np.asarray(vertices), graph, rings


CORONENE = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]


def _frame(positions, graph, **kwargs):
    return carbon.analyze_graphenic_structure(positions, np.eye(3) * 30, graph, pbc=False, **kwargs)


def test_graphene_fragment_has_known_area_and_two_thresholds():
    positions, graph, rings = _hex_patch(CORONENE)
    result = _frame(positions, graph)
    assert result["ring_counts"] == {6: 7}
    hex_area = 3 * math.sqrt(3) * 1.42**2 / 2
    for summary, row in zip(result["threshold_summaries"], result["crystallites"], strict=True):
        assert summary["graphenic_ring_fraction"] == 1
        assert summary["graphenic_atom_fraction"] == 1
        assert summary["crystallite_count"] == 1
        assert summary["normal_nematic_order"] == pytest.approx(1)
        assert row["area_A2"] == pytest.approx(7 * hex_area)
        assert row["sqrt_area_size_A"] == pytest.approx(math.sqrt(7 * hex_area))
        assert row["equivalent_disk_diameter_A"] == pytest.approx(
            2 * math.sqrt(7 * hex_area / math.pi)
        )
        assert row["thickness_A"] == pytest.approx(0, abs=1e-10)
        assert row["planarity_rms_A"] == pytest.approx(0, abs=1e-10)
    assert [s["threshold_deg"] for s in result["threshold_summaries"]] == [1, 2]
    supplied = _frame(positions, graph, ring_cycles=rings)
    assert supplied == result


def test_cata_line_and_too_small_fragments_do_not_become_crystallites():
    for centers in ([(0, 0)], [(0, 0), (1, 0)], [(0, 0), (1, 0), (2, 0)]):
        positions, graph, _ = _hex_patch(centers)
        result = _frame(positions, graph)
        assert result["crystallites"] == []
        assert result["size_distributions"] == []
        assert result["threshold_summaries"][0]["mean_sqrt_area_size_A"] is None
    assert result["threshold_summaries"][0]["junction_hexagon_count"] == 1


def test_vacancy_removes_local_hexagons_without_inventing_hybridization():
    positions, graph, _ = _hex_patch(CORONENE)
    graph[0].clear()
    for neighbors in graph:
        neighbors.discard(0)
    result = _frame(positions, graph)
    assert result["ring_counts"].get(6, 0) < 7
    assert result["threshold_summaries"][0]["graphenic_atom_fraction"] < 1
    assert "sp2" not in str(result["metadata"])


def test_gentle_curvature_retains_rings_and_changes_planarity():
    positions, graph, rings = _hex_patch(CORONENE)
    radius = 200.0
    bent = positions.copy()
    bent[:, 0] = radius * np.sin(positions[:, 0] / radius)
    bent[:, 2] = radius * (1 - np.cos(positions[:, 0] / radius))
    result = _frame(bent, graph, ring_cycles=rings)
    assert result["threshold_summaries"][1]["retained_ring_count"] == 7
    assert result["crystallites"][1]["thickness_A"] > 0
    assert result["crystallites"][1]["planarity_rms_A"] > 0
    assert result["ring_geometry_summary"]["mean_adjacent_normal_misorientation_deg"] > 0
    assert result["ring_geometry_summary"]["normal_nematic_order"] < 1


def test_normal_angle_sensitivity_is_computed_as_separate_domain_populations():
    positions, graph, rings = _hex_patch(CORONENE)
    radius = 100.0
    bent = positions.copy()
    bent[:, 0] = radius * np.sin(positions[:, 0] / radius)
    bent[:, 2] = radius * (1 - np.cos(positions[:, 0] / radius))
    result = _frame(bent, graph, ring_cycles=rings)
    first, second = result["threshold_summaries"]
    assert first["threshold_deg"] == 1 and first["retained_ring_count"] == 0
    assert second["threshold_deg"] == 2 and second["retained_ring_count"] == 7


def test_disconnected_fragments_keep_full_number_and_area_weights():
    first, first_graph, _ = _hex_patch(CORONENE)
    second, second_graph, _ = _hex_patch([(0, 0), (1, 0), (0, 1)], shift=(20, 0, 0))
    offset = len(first)
    graph = first_graph + [{atom + offset for atom in neighbors} for neighbors in second_graph]
    result = _frame(np.vstack((first, second)), graph, size_bin_width_A=0.1)
    rows = [row for row in result["size_distributions"] if row["threshold_deg"] == 2]
    assert sum(row["crystallite_count"] for row in rows) == 2
    for key in (
        "number_probability",
        "area_weighted_probability",
        "intensity_proxy_weighted_probability",
    ):
        assert sum(row[key] for row in rows) == pytest.approx(1)
    occupied = [row for row in rows if row["crystallite_count"]]
    assert sorted(row["number_probability"] for row in occupied) == [0.5, 0.5]
    assert sorted(row["area_weighted_probability"] for row in occupied) == pytest.approx([0.3, 0.7])


def test_rotated_triclinic_wrapped_finite_fragment_is_invariant():
    positions, graph, rings = _hex_patch(CORONENE, shift=(18, 18, 10))
    cell = np.asarray([[25, 0, 0], [4, 26, 0], [2, 3, 27]], dtype=float)
    reference = carbon.analyze_graphenic_structure(positions, cell, graph, ring_cycles=rings)
    rng = np.random.default_rng(4)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotated_cell = cell @ rotation
    shifted = (positions + np.array([24, 12, 40])) @ rotation
    wrapped = np.mod(shifted @ np.linalg.inv(rotated_cell), 1) @ rotated_cell
    wrapped += rng.integers(-3, 4, size=(len(positions), 3)) @ rotated_cell
    transformed = carbon.analyze_graphenic_structure(
        wrapped, rotated_cell, graph, ring_cycles=rings
    )
    for before, after in zip(reference["crystallites"], transformed["crystallites"], strict=True):
        assert after["periodic"] is False
        for key in (
            "area_A2",
            "sqrt_area_size_A",
            "thickness_A",
            "planarity_rms_A",
            "normal_nematic_order",
        ):
            assert after[key] == pytest.approx(before[key], abs=1e-9)


def _periodic_graphene(layers=1):
    bond = 1.42
    lattice = np.asarray(
        [[1.5 * bond, math.sqrt(3) * bond / 2, 0], [0, math.sqrt(3) * bond, 0], [0, 0, 12]]
    )
    # Use a larger unit cell so nearest atom identities are unambiguous.
    positions, _, rings = _hex_patch([(i, j) for i in range(4) for j in range(4)])
    cell = lattice.copy()
    cell[:2] *= 4
    unique, mapping = [], {}
    remapped = []
    for ring in rings:
        row = []
        for atom in ring:
            fractional = np.mod(positions[atom] @ np.linalg.inv(cell), 1)
            fractional[np.isclose(fractional, 1, atol=1e-9)] = 0
            key = tuple(np.round(fractional, 9))
            if key not in mapping:
                mapping[key] = len(unique)
                unique.append(fractional @ cell)
            row.append(mapping[key])
        remapped.append(tuple(row))
    points = np.asarray(unique)
    graph = [set() for _ in points]
    for ring in remapped:
        for index, first in enumerate(ring):
            second = ring[(index + 1) % 6]
            graph[first].add(second)
            graph[second].add(first)
    all_graph, all_points, all_rings = [], [], []
    for layer in range(layers):
        offset = layer * len(points)
        all_graph.extend({atom + offset for atom in neighbors} for neighbors in graph)
        all_points.extend(points + np.array([0, 0, layer * 3.35]))
        all_rings.extend(tuple(atom + offset for atom in ring) for ring in remapped)
    return np.asarray(all_points), cell, all_graph, all_rings


@pytest.mark.parametrize("layers", [1, 2])
def test_periodic_graphene_and_graphite_do_not_get_a_finite_size(layers):
    positions, cell, graph, rings = _periodic_graphene(layers)
    result = carbon.analyze_graphenic_structure(positions, cell, graph, ring_cycles=rings)
    summary = result["threshold_summaries"][1]
    assert summary["periodic_crystallite_count"] == layers
    assert summary["finite_crystallite_count"] == 0
    assert summary["graphenic_ring_fraction"] == 1
    assert summary["normal_nematic_order"] == pytest.approx(1)
    assert summary["mean_sqrt_area_size_A"] is None
    assert result["size_distributions"] == []
    for row in result["crystallites"]:
        assert row["periodic"] and row["wrap_a"] and row["wrap_b"]
        assert not row["wrap_c"]
        assert row["sqrt_area_size_A"] is None
        assert row["thickness_A"] is None
        assert row["area_scope"] == "periodic_cell"


def test_existing_bond_and_ring_workloads_can_be_reused(monkeypatch):
    positions, graph, rings = _hex_patch(CORONENE)
    vectors = [
        {neighbor: positions[neighbor] - positions[index] for neighbor in neighbors}
        for index, neighbors in enumerate(graph)
    ]

    def unexpected(*_args, **_kwargs):
        raise AssertionError("Reused graph/rings/vectors must not trigger another search")

    monkeypatch.setattr(carbon, "shortest_path_ring_cycles", unexpected)
    monkeypatch.setattr(carbon, "minimum_image_displacements", unexpected)
    result = _frame(positions, graph, ring_cycles=rings, bond_vectors=vectors)
    assert result["threshold_summaries"][1]["retained_ring_count"] == 7


def test_minimum_image_batches_remain_bounded(monkeypatch):
    positions, graph, rings = _hex_patch(CORONENE)
    monkeypatch.setattr(carbon, "MAX_EDGE_BATCH", 3)
    batches = []
    original = carbon.minimize_vectors

    def tracked(values, dimensions):
        batches.append(len(values))
        return original(values, dimensions)

    monkeypatch.setattr(carbon, "minimize_vectors", tracked)
    carbon.analyze_graphenic_structure(positions, np.eye(3) * 30, graph, ring_cycles=rings)
    assert len(batches) > 1 and max(batches) <= 3


def test_ill_formed_graph_and_budget_exhaustion_fail_without_partial_statistics():
    with pytest.raises(carbon.GraphenicAnalysisError, match="symmetric"):
        carbon.shortest_path_ring_cycles([[1], []])
    positions, graph, _ = _hex_patch(CORONENE)
    with pytest.raises(carbon.GraphenicAnalysisError, match="budget"):
        carbon.shortest_path_ring_cycles(graph, max_search_states=1)
    with pytest.raises(carbon.GraphenicAnalysisError, match="settings"):
        _frame(positions, graph, normal_thresholds_deg=(0,))


def test_shortest_cycles_exclude_chords_and_count_defect_rings():
    graph = [{1, 4}, {0, 2}, {1, 3}, {2, 4}, {0, 3}]
    assert carbon.shortest_path_ring_cycles(graph) == ((0, 1, 2, 3, 4),)
    graph[0].add(2)
    graph[2].add(0)
    cycles = carbon.shortest_path_ring_cycles(graph)
    assert {len(ring) for ring in cycles} == {3, 4}
    assert 5 not in {len(ring) for ring in cycles}


@pytest.mark.parametrize(
    "periodic", [(True, True, True), (True, False, True), (False, True, False)]
)
def test_rotated_minimum_images_match_independent_neighboring_image_reference(periodic):
    rng = np.random.default_rng(19)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    cell = np.asarray([[10.0, 0.0, 0.0], [3.0, 11.0, 0.0], [-2.0, 4.0, 13.0]]) @ rotation
    periodic = np.asarray(periodic)
    offsets = np.asarray(list(product(range(-2, 3), repeat=int(sum(periodic)))))
    image_offsets = np.zeros((len(offsets), 3))
    image_offsets[:, periodic] = offsets
    values = rng.uniform(-0.45, 0.45, size=(23, 3)) @ cell
    added_images = rng.integers(-4, 5, size=(len(values), 3))
    added_images[:, ~periodic] = 0
    translated = values + added_images @ cell
    result = carbon.minimum_image_displacements(translated, cell, pbc=periodic)
    reference = []
    for value in values:
        candidates = value - image_offsets @ cell
        reference.append(candidates[np.argmin(np.linalg.norm(candidates, axis=1))])
    assert result == pytest.approx(np.asarray(reference), abs=1e-9)


@pytest.mark.parametrize("periodic", [(True, True, True), (True, True, False)])
def test_unreduced_acute_cell_is_reduced_before_library_minimum_images(periodic):
    rng = np.random.default_rng(3)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    cell = np.asarray([[10.0, 0.0, 0.0], [9.7, 1.0, 0.0], [0.0, 0.0, 10.0]]) @ rotation
    periodic = np.asarray(periodic)
    offsets = np.asarray(list(product(range(-6, 7), repeat=int(sum(periodic)))))
    images = np.zeros((len(offsets), 3))
    images[:, periodic] = offsets
    displacements = rng.uniform(-0.5, 0.5, (23, 3)) @ cell
    stored_images = rng.integers(-4, 5, (len(displacements), 3))
    stored_images[:, ~periodic] = 0
    actual = carbon.minimum_image_displacements(
        displacements + stored_images @ cell, cell, pbc=periodic
    )
    reference = []
    for value in displacements:
        candidates = value - images @ cell
        reference.append(candidates[np.argmin(np.linalg.norm(candidates, axis=1))])
    assert actual == pytest.approx(np.asarray(reference), abs=1e-9)


@pytest.mark.parametrize(
    ("slots", "junction"),
    [
        ([0, 3], True),
        ([0, 1, 3], True),
        ([0, 1, 3, 4], True),
        ([0, 1, 2], False),
        ([5, 0, 1], False),
        (list(range(6)), False),
    ],
)
def test_cyclic_slot_rule_distinguishes_cata_peri_junctions(slots, junction):
    assert carbon._junction(slots) is junction


def test_empty_ring_graph_has_missing_finite_statistics():
    result = _frame(np.asarray([[0.0, 0.0, 0.0]]), [set()])
    assert result["ring_counts"] == {}
    assert result["size_distributions"] == []
    assert result["threshold_summaries"][0]["graphenic_ring_fraction"] is None
    assert result["threshold_summaries"][0]["mean_sqrt_area_size_A"] is None


def test_reused_cycles_must_respect_recorded_ring_size_range():
    positions, graph, rings = _hex_patch(CORONENE)
    with pytest.raises(carbon.GraphenicAnalysisError, match="supplied"):
        _frame(positions, graph, ring_cycles=rings, maximum_ring_size=5)
