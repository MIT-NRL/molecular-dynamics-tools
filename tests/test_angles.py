"""Bounded shared-thread angles and independent periodic-triplet references."""

from itertools import product
from threading import Barrier, get_ident

import freud
import numpy as np
import pytest

from molecular_dynamics_tools import angles, compute_bond_angles, load_trajectory


def _reference_counts(positions, species, cell, definition, edges):
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    first_atoms = np.flatnonzero(species == definition.first)
    third_atoms = np.flatnonzero(species == definition.third)
    for center in np.flatnonzero(species == definition.center):
        # Independent explicit image enumeration for these moderate-tilt cells;
        # avoid depending on either MDT/Freud geometry or an extra test package.
        shifts = np.asarray(list(product((-1, 0, 1), repeat=3))) @ cell
        candidates = np.asarray(positions - positions[center], dtype=float)[:, None] + shifts
        norms = np.linalg.norm(candidates, axis=2)
        nearest = np.argmin(norms, axis=1)
        vectors = candidates[np.arange(len(positions)), nearest]
        distances = norms[np.arange(len(positions)), nearest]
        first = first_atoms[
            (distances[first_atoms] < definition.first_center_max) & (distances[first_atoms] > 0)
        ]
        third = third_atoms[
            (distances[third_atoms] < definition.center_third_max) & (distances[third_atoms] > 0)
        ]
        values = []
        for left in first:
            for right in third:
                if definition.first == definition.third:
                    if left == right:
                        continue
                    if np.isclose(definition.first_center_max, definition.center_third_max):
                        if left >= right:
                            continue
                cosine = np.dot(vectors[left], vectors[right]) / (
                    distances[left] * distances[right]
                )
                values.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
        counts += np.histogram(values, edges)[0]
    return counts


@pytest.mark.parametrize(
    "cell", [np.eye(3) * 8, np.array([[8, 0, 0], [1.2, 8.2, 0], [-0.7, 1.1, 7.8]])]
)
@pytest.mark.parametrize(
    "definition",
    [
        angles.AngleDefinition("C", "C", "C", 2.8, 2.8),
        angles.AngleDefinition("C", "A", "C", 2.2, 2.8),
        angles.AngleDefinition("C", "A", "B", 2.8, 2.8),
    ],
)
def test_threaded_angles_match_independent_full_triplets(cell, definition, monkeypatch):
    rng = np.random.default_rng(629)
    positions = np.asarray((rng.random((45, 3)) - 0.5) @ cell, dtype=np.float32)
    species = np.asarray(["C"] * 15 + ["A"] * 15 + ["B"] * 15)
    positions[0] = positions[20]  # Coincident outer/center atoms must be excluded.
    box = freud.Box.from_matrix(cell.T)
    edges = np.linspace(0, 180, 181)
    monkeypatch.setattr(angles, "_CENTER_BATCH_SIZE", 2)
    expected = _reference_counts(positions, species, cell, definition, edges)
    assert expected.sum() > 0
    serial = angles._angle_counts_for_frame(box, positions, species, definition, edges)
    threaded = angles._angle_counts_for_frame(box, positions, species, definition, edges, ncore=2)
    np.testing.assert_array_equal(serial, expected)
    np.testing.assert_array_equal(threaded, serial)


def test_dense_local_shell_pair_scratch_is_bounded(monkeypatch):
    rng = np.random.default_rng(42)
    positions = np.asarray(np.vstack([np.zeros(3), rng.normal(0, 0.05, (50, 3))]), dtype=np.float32)
    species = np.asarray(["A"] + ["C"] * 50)
    positions[1] = positions[0]
    cell = np.eye(3) * 4
    definition = angles.AngleDefinition("C", "A", "C", 0.3, 0.3)
    edges = np.linspace(0, 180, 181)
    monkeypatch.setattr(angles, "_PAIR_BLOCK_SIZE", 4)
    monkeypatch.setattr(angles, "_MAX_BATCH_NEIGHBORS", 7)
    result = angles._angle_counts_for_frame(
        freud.Box.from_matrix(cell.T), positions, species, definition, edges, ncore=2
    )
    np.testing.assert_array_equal(
        result, _reference_counts(positions, species, cell, definition, edges)
    )
    assert result.sum() == 49 * 48 // 2


def _trajectory(tmp_path, frames=1):
    path = tmp_path / "angles.xyz"
    path.write_text(
        ('4\nLattice="8 0 0 0 8 0 0 0 8"\nC 0 0 0\nC 1.1 0 0\nC 0 1.2 0\nC 0 0 1.3\n') * frames
    )
    return load_trajectory(path)


def test_single_frame_public_ncore_dispatches_real_shared_threads(tmp_path, monkeypatch):
    monkeypatch.setattr(angles, "_CENTER_BATCH_SIZE", 1)
    original = angles._counts_for_center_batch
    barrier, identifiers = Barrier(2), set()

    def observed(*args):
        identifiers.add(get_ident())
        barrier.wait(timeout=10)
        return original(*args)

    with _trajectory(tmp_path) as trajectory:
        serial = compute_bond_angles(trajectory, [("C", "C", "C", 1.8, 1.8)])
        monkeypatch.setattr(angles, "_counts_for_center_batch", observed)
        threaded = compute_bond_angles(trajectory, [("C", "C", "C", 1.8, 1.8)], ncore=2)
    assert len(identifiers) == 2
    assert threaded.attrs["execution"]["center_threads_per_worker"] == 2
    assert threaded.attrs["execution"]["center_backend"] == "shared_threads"
    np.testing.assert_array_equal(threaded.to_numpy(), serial.to_numpy())
    assert threaded.attrs["sample_counts"]["C-C-C"] == 12
    assert serial.attrs["execution"]["center_backend"] == "serial"


def test_multiprocessing_frame_plan_keeps_center_workers_single_threaded(tmp_path, monkeypatch):
    def execute(trajectory, frames, plan, accumulator, arguments, **kwargs):
        assert plan.worker_count == 2
        assert arguments[-1] == 1
        return [accumulator(trajectory, frames, *arguments)], {"backend": plan.backend}

    monkeypatch.setattr(angles, "execute_frame_chunks", execute)
    with _trajectory(tmp_path, frames=2) as trajectory:
        result = compute_bond_angles(
            trajectory, [("C", "C", "C", 1.8, 1.8)], ncore=2, backend="multiprocessing"
        )
    assert result.attrs["execution"]["center_threads_per_worker"] == 1
    assert result.attrs["execution"]["center_backend"] == "serial"
