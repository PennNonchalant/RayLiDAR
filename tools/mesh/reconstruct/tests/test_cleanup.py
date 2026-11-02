"""Exact cleanup statistics on hand-built fixtures (RECON-04, D-08)."""

from __future__ import annotations

import numpy as np
import pytest

from tools.mesh.reconstruct import cleanup

TETRA_VERTICES = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
TETRA_FACES = np.array(
    [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ]
)


def _tetra(offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    vertices = TETRA_VERTICES + np.array([offset, 0.0, 0.0])
    return vertices, TETRA_FACES.copy()


def _fan_triangles(center: int, count: int, first_outer: int, radius: float = 10.0):
    """A fan of ``count`` triangles all sharing ``center`` plus outer vertices."""
    angles = 2.0 * np.pi * np.arange(count) / count
    outer = np.stack(
        [
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.ones(count),
        ],
        axis=1,
    )
    faces = [
        [center, first_outer + index, first_outer + ((index + 1) % count)]
        for index in range(count)
    ]
    return outer, np.asarray(faces, dtype=np.int64)


def test_tetrahedron_welds_and_cleans_to_exact_counts() -> None:
    result = cleanup.clean_mesh(TETRA_VERTICES, TETRA_FACES, min_component_triangles=1)

    stats = result["stats"]
    assert stats["vertex_count"] == 4
    assert stats["triangle_count"] == 4
    assert stats["welded_vertices"] == 4
    assert stats["pre_clean_triangle_count"] == 4
    assert stats["degenerate_removed_count"] == 0
    assert stats["degenerate_fraction"] == 0.0
    assert stats["components_count"] == 1
    assert stats["largest_component_fraction"] == 1.0
    assert stats["removed_components"] == 0
    assert stats["removed_unreferenced_vertices"] == 0
    assert stats["bbox_min_m"] == pytest.approx([0.0, 0.0, 0.0])
    assert stats["bbox_max_m"] == pytest.approx([1.0, 1.0, 1.0])


def test_two_disjoint_tetrahedra_have_two_components() -> None:
    left_v, left_f = _tetra(0.0)
    right_v, right_f = _tetra(100.0)
    vertices = np.vstack([left_v, right_v])
    faces = np.vstack([left_f, right_f + 4])

    result = cleanup.clean_mesh(vertices, faces, min_component_triangles=1)

    stats = result["stats"]
    assert stats["components_count"] == 2
    assert stats["largest_component_fraction"] == 0.5
    assert stats["triangle_count"] == 8


def test_repeated_index_and_zero_area_faces_are_removed_and_counted() -> None:
    outer, fan_faces = _fan_triangles(center=0, count=5, first_outer=1)
    vertices = np.vstack(
        [
            np.array([[0.0, 0.0, 0.0]]),
            outer,
            np.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0]]),
        ]
    )
    repeated_index_face = [0, 0, 1]
    collinear_face = [6, 7, 8]
    faces = np.vstack(
        [fan_faces, np.asarray([repeated_index_face, collinear_face])]
    )

    result = cleanup.clean_mesh(vertices, faces, min_component_triangles=1)

    stats = result["stats"]
    pre_clean = 7
    removed = 2
    assert stats["pre_clean_triangle_count"] == pre_clean
    assert stats["degenerate_removed_count"] == removed
    assert stats["degenerate_fraction"] == removed / pre_clean
    assert stats["triangle_count"] == 5
    assert stats["components_count"] == 1
    assert stats["largest_component_fraction"] == 1.0


def test_three_triangles_sharing_one_welded_vertex_are_one_component() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # shared centre
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 3, 4], [0, 5, 6]])

    result = cleanup.clean_mesh(vertices, faces, min_component_triangles=1)

    stats = result["stats"]
    assert stats["components_count"] == 1
    assert stats["triangle_count"] == 3
    assert stats["largest_component_fraction"] == 1.0


def test_speck_below_min_component_triangles_is_removed_as_one_component() -> None:
    # The plan text pairs a 9-triangle component with threshold 10, which would
    # remove both parts (9 < 10).  The fixture uses 12 triangles so the only
    # removable component is the 2-triangle speck, which is what the scenario
    # is meant to prove.
    outer, fan_faces = _fan_triangles(center=0, count=12, first_outer=1)
    vertices = np.vstack(
        [
            np.array([[0.0, 0.0, 0.0]]),
            outer,
            np.array(
                [
                    [100.0, 0.0, 0.0],
                    [101.0, 0.0, 0.0],
                    [100.0, 1.0, 0.0],
                    [101.0, 1.0, 0.0],
                ]
            ),
        ]
    )
    speck = outer.shape[0] + 1
    speck_faces = np.array(
        [[speck, speck + 1, speck + 2], [speck, speck + 2, speck + 3]]
    )
    faces = np.vstack([fan_faces, speck_faces])

    result = cleanup.clean_mesh(vertices, faces, min_component_triangles=10)

    stats = result["stats"]
    assert stats["pre_clean_triangle_count"] == 14
    assert stats["removed_components"] == 1
    assert stats["removed_component_triangles"] == 2
    assert stats["triangle_count"] == 12
    assert stats["components_count"] == 1
    assert stats["largest_component_fraction"] == 1.0


def test_weld_is_order_stable_for_shuffled_duplicates() -> None:
    base_vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    duplicated = np.vstack([base_vertices, base_vertices])
    # The second triangle references the duplicate copies of vertices 1, 3, 2.
    faces = np.array([[0, 1, 2], [5, 7, 6]])
    permutation = np.array([3, 0, 6, 2, 7, 1, 5, 4])
    inverse = np.argsort(permutation)
    shuffled_vertices = duplicated[permutation]
    shuffled_faces = inverse[faces]

    def canonical(vertices: np.ndarray, face_array: np.ndarray):
        welded, remap, counts = cleanup.weld_vertices(vertices)
        remapped = remap[face_array]
        coordinates = [tuple(np.round(row, 6)) for row in welded]
        topology = sorted(
            tuple(sorted((coordinates[a], coordinates[b], coordinates[c])))
            for a, b, c in remapped
        )
        return sorted(coordinates), topology, sorted(int(c) for c in counts)

    assert canonical(duplicated, faces) == canonical(shuffled_vertices, shuffled_faces)


def test_weld_vertices_reports_duplicates_and_inverse_counts() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    welded, remap, inverse_counts = cleanup.weld_vertices(vertices)

    assert welded.shape == (3, 3)
    assert remap.tolist() == [0, 1, 2, 0]
    assert inverse_counts.tolist() == [2, 1, 1]


def test_decimate_is_a_no_op_below_the_target() -> None:
    vertices, faces = cleanup.decimate(TETRA_VERTICES, TETRA_FACES, 100)

    assert vertices.shape == TETRA_VERTICES.shape
    assert faces.shape == TETRA_FACES.shape
    assert np.array_equal(vertices, TETRA_VERTICES)
    assert np.array_equal(faces, TETRA_FACES)


def test_clean_mesh_rejects_an_all_speck_mesh() -> None:
    left_v, left_f = _tetra()
    right_v, right_f = _tetra(100.0)
    vertices = np.vstack([left_v, right_v])
    faces = np.vstack([left_f, right_f + 4])

    with pytest.raises(cleanup.CleanupError, match="every triangle"):
        cleanup.clean_mesh(vertices, faces, min_component_triangles=10)
