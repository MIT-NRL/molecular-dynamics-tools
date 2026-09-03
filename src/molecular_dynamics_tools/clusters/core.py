"""Shared graph, connected-component, and periodic-wrapping kernels."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


class DisjointSet:
    """Union-find with path compression for per-frame cluster graphs."""

    def __init__(self, size: int):
        self.parent = np.arange(int(size), dtype=np.int32)
        self.rank = np.zeros(int(size), dtype=np.int8)

    def find(self, node: int) -> int:
        parent = self.parent
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return int(node)

    def union(self, left: int, right: int) -> None:
        left_root = self.find(int(left))
        right_root = self.find(int(right))
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
        elif self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root
        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1


def component_members(
    node_count: int, edges: Iterable[tuple[int, int]]
) -> tuple[list[list[int]], NDArray[np.int64]]:
    """Return connected-component members and each node's component index."""

    disjoint = DisjointSet(node_count)
    for left, right in edges:
        disjoint.union(left, right)
    roots = np.asarray([disjoint.find(index) for index in range(node_count)])
    _, inverse = np.unique(roots, return_inverse=True)
    components: list[list[int]] = [[] for _ in range(int(inverse.max()) + 1)] if node_count else []
    for node, component in enumerate(inverse):
        components[int(component)].append(node)
    return components, inverse.astype(np.int64, copy=False)


def component_wrap_flags(
    component: Sequence[int],
    adjacency: Mapping[int, Sequence[tuple[int, NDArray[np.int32]]]],
) -> NDArray[np.bool_]:
    """Detect noncontractible graph cycles along periodic cell axes."""

    if not component:
        return np.zeros(3, dtype=bool)
    start = int(component[0])
    assigned = {start: np.zeros(3, dtype=np.int32)}
    stack = [start]
    wraps = np.zeros(3, dtype=bool)
    while stack:
        node = stack.pop()
        for neighbor, translation in adjacency.get(node, ()):
            expected = assigned[node] + np.asarray(translation, dtype=np.int32)
            if neighbor not in assigned:
                assigned[neighbor] = expected
                stack.append(neighbor)
            else:
                wraps |= expected != assigned[neighbor]
    return wraps


def periodic_components(
    node_count: int,
    translated_edges: Iterable[tuple[int, int, NDArray[np.int32]]],
) -> list[dict[str, object]]:
    """Return component members and periodic wrapping flags for a graph."""

    edges: list[tuple[int, int]] = []
    adjacency: dict[int, list[tuple[int, NDArray[np.int32]]]] = defaultdict(list)
    for left, right, translation in translated_edges:
        shift = np.asarray(translation, dtype=np.int32)
        edges.append((int(left), int(right)))
        adjacency[int(left)].append((int(right), shift))
        adjacency[int(right)].append((int(left), -shift))
    components, _ = component_members(node_count, edges)
    result: list[dict[str, object]] = []
    for members in components:
        wraps = component_wrap_flags(members, adjacency)
        result.append(
            {
                "members": members,
                "size": len(members),
                "wrap_x": bool(wraps[0]),
                "wrap_y": bool(wraps[1]),
                "wrap_z": bool(wraps[2]),
                "wrap_any": bool(np.any(wraps)),
            }
        )
    return result


__all__ = [
    "DisjointSet",
    "component_members",
    "component_wrap_flags",
    "periodic_components",
]
