"""Deterministic mesh welding, degeneracy, components, and cleanup.

The functions here are the numeric half of the reconstruction contract: every
count and fraction is computed from the arrays with NumPy, never inferred from a
writer's claim.  Only NumPy and the standard library are imported; no heavy
meshing library is required.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DEGENERATE_AREA_M2 = 1e-12
DEFAULT_WELD_TOL = 1e-6
DEFAULT_MIN_COMPONENT_TRIANGLES = 10


class CleanupError(ValueError):
    """Raised when the mesh arrays cannot be cleaned without losing the mesh."""


def _as_vertices(vertices: Any) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(vertices, dtype=np.float64))
    if array.ndim != 2 or array.shape[1] != 3:
        raise CleanupError("vertices must have shape (N, 3)")
    if not np.isfinite(array).all():
        raise CleanupError("vertex coordinates must be finite")
    return array


def _as_faces(faces: Any, vertex_count: int) -> np.ndarray:
    array = np.asarray(faces)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise CleanupError("faces must be triangles with shape (M, 3)")
    if not np.issubdtype(array.dtype, np.integer):
        raise CleanupError("face indices must be integers")
    array = np.ascontiguousarray(array, dtype=np.int64)
    if array.min() < 0:
        raise CleanupError("face indices must be non-negative")
    if array.max() >= vertex_count:
        raise CleanupError("face index is out of range")
    return array


def weld_vertices(
    vertices: Any, tol: float = DEFAULT_WELD_TOL
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weld coincident vertices with deterministic first-occurrence ordering.

    Each coordinate is quantized with ``np.rint(coord / tol)``.  Vertices that
    share a quantized triple become one vertex: the **first occurrence in input
    order** wins, so the welded ordering (and therefore every face remap) is
    deterministic for a given input sequence.

    Returns ``(unique_vertices, remap, inverse_counts)`` where ``remap`` maps
    every original vertex index to its welded index and ``inverse_counts[i]``
    is how many input vertices were welded into ``unique_vertices[i]``.
    """
    array = _as_vertices(vertices)
    tolerance = float(tol)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise CleanupError("weld tolerance must be a positive finite number")
    if array.shape[0] == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    keys = np.rint(array / tolerance).astype(np.int64)
    _, first_index, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )
    inverse = np.asarray(inverse, dtype=np.int64).reshape(-1)
    first_index = np.asarray(first_index, dtype=np.int64).reshape(-1)

    # ``np.unique`` returns keys sorted lexicographically; re-rank by first
    # occurrence so the welded order follows the input order.
    order = np.argsort(first_index, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.shape[0], dtype=np.int64)

    unique_vertices = array[first_index[order]]
    remap = rank[inverse]
    counts = np.bincount(inverse, minlength=first_index.shape[0])[order]
    return unique_vertices, remap, np.asarray(counts, dtype=np.int64)


def degenerate_faces(vertices: Any, faces: Any) -> np.ndarray:
    """True for a triangle with a repeated index or area below 1e-12 m²."""
    array = _as_vertices(vertices)
    face_array = _as_faces(faces, array.shape[0])
    if face_array.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    repeated = (
        (face_array[:, 0] == face_array[:, 1])
        | (face_array[:, 1] == face_array[:, 2])
        | (face_array[:, 0] == face_array[:, 2])
    )
    a = array[face_array[:, 0]]
    b = array[face_array[:, 1]]
    c = array[face_array[:, 2]]
    # float64 throughout: ``np.asarray`` above already fixed the dtype.
    cross = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    return np.asarray(repeated | (area < DEGENERATE_AREA_M2), dtype=bool)


def _find(parent: list[int], node: int) -> int:
    root = node
    while parent[root] != root:
        root = parent[root]
    while parent[node] != root:
        parent[node], node = root, parent[node]
    return root


def _union(parent: list[int], sizes: list[int], left: int, right: int) -> None:
    root_left = _find(parent, left)
    root_right = _find(parent, right)
    if root_left == root_right:
        return
    if sizes[root_left] < sizes[root_right]:
        root_left, root_right = root_right, root_left
    parent[root_right] = root_left
    sizes[root_left] += sizes[root_right]


def connected_components(vertex_count: int, faces: Any) -> dict[str, Any]:
    """Union-find components over welded shared vertices/edges.

    Components with zero triangles are never counted (they do not exist in this
    formulation: only vertices referenced by a face are considered).  The
    returned ``triangle_counts`` is keyed by the compact component id used in
    ``component_ids``; unreferenced vertices map to ``-1``.
    """
    count = int(vertex_count)
    face_array = _as_faces(faces, count)
    if face_array.shape[0] == 0:
        return {
            "count": 0,
            "largest_triangles": 0,
            "largest_fraction": 0.0,
            "triangle_counts": {},
            "component_ids": np.full(count, -1, dtype=np.int64),
        }

    parent = list(range(count))
    sizes = [1] * count
    for a, b, c in face_array.tolist():
        _union(parent, sizes, a, b)
        _union(parent, sizes, b, c)
        _union(parent, sizes, a, c)

    referenced = np.unique(face_array.reshape(-1))
    roots = np.fromiter(
        (_find(parent, int(vertex)) for vertex in referenced),
        dtype=np.int64,
        count=referenced.shape[0],
    )
    _, compact = np.unique(roots, return_inverse=True)
    compact = np.asarray(compact, dtype=np.int64).reshape(-1)

    component_ids = np.full(count, -1, dtype=np.int64)
    component_ids[referenced] = compact
    triangle_components = component_ids[face_array[:, 0]]
    counts = np.bincount(triangle_components, minlength=int(compact.max()) + 1)

    total_triangles = int(face_array.shape[0])
    largest_triangles = int(counts.max()) if counts.size else 0
    return {
        "count": int(counts.shape[0]),
        "largest_triangles": largest_triangles,
        "largest_fraction": (
            float(largest_triangles) / float(total_triangles)
            if total_triangles
            else 0.0
        ),
        "triangle_counts": {
            int(index): int(value) for index, value in enumerate(counts)
        },
        "component_ids": component_ids,
    }


def clean_mesh(
    vertices: Any,
    faces: Any,
    *,
    min_component_triangles: int = DEFAULT_MIN_COMPONENT_TRIANGLES,
) -> dict[str, Any]:
    """Weld, drop degenerates and specks, drop unused vertices, and report.

    Order (fixed): weld → count/drop degenerates → drop components below
    ``min_component_triangles`` → drop unreferenced vertices → recompute final
    components.  Every fraction is relative to the documented denominator
    (degenerate fraction uses the pre-clean triangle count; the largest
    component fraction uses the final triangle count).
    """
    threshold = int(min_component_triangles)
    if threshold < 1:
        raise CleanupError("min_component_triangles must be at least 1")

    vertex_array = _as_vertices(vertices)
    face_array = _as_faces(faces, vertex_array.shape[0])
    welded_vertices, remap, _ = weld_vertices(vertex_array)
    welded_faces = remap[face_array] if face_array.shape[0] else face_array

    pre_clean_triangle_count = int(welded_faces.shape[0])
    if pre_clean_triangle_count:
        mask = degenerate_faces(welded_vertices, welded_faces)
        degenerate_removed_count = int(np.count_nonzero(mask))
        kept_faces = welded_faces[~mask]
    else:
        degenerate_removed_count = 0
        kept_faces = welded_faces
    degenerate_fraction = (
        float(degenerate_removed_count) / float(pre_clean_triangle_count)
        if pre_clean_triangle_count
        else 0.0
    )
    if kept_faces.shape[0] == 0:
        raise CleanupError("mesh has no non-degenerate triangles after welding")

    pre_drop = connected_components(welded_vertices.shape[0], kept_faces)
    triangle_component = pre_drop["component_ids"][kept_faces[:, 0]]
    triangle_counts: dict[int, int] = pre_drop["triangle_counts"]
    kept_component_ids = np.asarray(
        [
            component_id
            for component_id, triangles in triangle_counts.items()
            if triangles >= threshold
        ],
        dtype=np.int64,
    )
    removed_components = sum(
        1 for triangles in triangle_counts.values() if triangles < threshold
    )
    removed_component_triangles = sum(
        triangles for triangles in triangle_counts.values() if triangles < threshold
    )
    if kept_component_ids.size:
        keep_mask = np.isin(triangle_component, kept_component_ids)
    else:
        keep_mask = np.zeros(triangle_component.shape[0], dtype=bool)
    component_faces = kept_faces[keep_mask]
    if component_faces.shape[0] == 0:
        raise CleanupError(
            "cleanup removed every triangle: the whole mesh is below "
            f"min_component_triangles={threshold}"
        )

    referenced = np.unique(component_faces.reshape(-1))
    contiguous = np.full(welded_vertices.shape[0], -1, dtype=np.int64)
    contiguous[referenced] = np.arange(referenced.shape[0], dtype=np.int64)
    final_vertices = welded_vertices[referenced]
    final_faces = contiguous[component_faces]

    final_components = connected_components(final_vertices.shape[0], final_faces)
    bbox_min = [float(value) for value in final_vertices.min(axis=0)]
    bbox_max = [float(value) for value in final_vertices.max(axis=0)]

    stats = {
        "vertex_count": int(final_vertices.shape[0]),
        "triangle_count": int(final_faces.shape[0]),
        "bbox_min_m": bbox_min,
        "bbox_max_m": bbox_max,
        "pre_clean_triangle_count": pre_clean_triangle_count,
        "degenerate_removed_count": degenerate_removed_count,
        "degenerate_fraction": degenerate_fraction,
        "components_count": int(final_components["count"]),
        "largest_component_fraction": float(final_components["largest_fraction"]),
        "removed_components": int(removed_components),
        "removed_component_triangles": int(removed_component_triangles),
        "removed_unreferenced_vertices": int(
            welded_vertices.shape[0] - referenced.shape[0]
        ),
        "welded_vertices": int(welded_vertices.shape[0]),
        "removed_duplicate_vertices": int(
            vertex_array.shape[0] - welded_vertices.shape[0]
        ),
    }
    return {
        "vertices": final_vertices,
        "faces": final_faces,
        "stats": stats,
    }


__all__ = [
    "CleanupError",
    "DEFAULT_MIN_COMPONENT_TRIANGLES",
    "DEFAULT_WELD_TOL",
    "DEGENERATE_AREA_M2",
    "clean_mesh",
    "connected_components",
    "degenerate_faces",
    "weld_vertices",
]
