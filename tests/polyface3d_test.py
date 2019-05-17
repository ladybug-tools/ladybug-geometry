# coding=utf-8

from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
from ladybug_geometry.geometry3d.plane import Plane
from ladybug_geometry.geometry3d.line import LineSegment3D
from ladybug_geometry.geometry3d.face import Face3D
from ladybug_geometry.geometry3d.polyface import Polyface3D

import math
import unittest
import pytest


class Polyface3DTestCase(unittest.TestCase):
    """Test for Poyface3D"""

    def test_polyface3d_init_solid(self):
        """Test the initalization of Poyface3D and basic properties of solid objects."""
        pts = [Point3D(0, 0, 0), Point3D(0, 2, 0), Point3D(2, 2, 0), Point3D(2, 0, 0),
               Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2)]
        face_indices = [(0, 1, 2, 3), (0, 4, 5, 1), (0, 3, 7, 4),
                        (2, 1, 5, 6), (2, 3, 7, 6), (4, 5, 6, 7)]
        polyface = Polyface3D(pts, face_indices)

        assert len(polyface.vertices) == 8
        assert len(polyface.face_indices) == 6
        assert len(polyface.faces) == 6
        assert len(polyface.edge_indices) == 12
        assert len(polyface.edges) == 12
        assert len(polyface.naked_edges) == 0
        assert len(polyface.non_manifold_edges) == 0
        assert len(polyface.internal_edges) == 12
        assert polyface.area == 24
        assert polyface.volume == 8
        assert polyface.is_solid

        for face in polyface.faces:
            assert face.area == 4
        assert polyface.faces[0].normal == Vector3D(0, 0, -1)
        assert polyface.faces[1].normal == Vector3D(-1, 0, 0)
        assert polyface.faces[2].normal == Vector3D(0, -1, 0)
        assert polyface.faces[3].normal == Vector3D(0, 1, 0)
        assert polyface.faces[4].normal == Vector3D(1, 0, 0)
        assert polyface.faces[5].normal == Vector3D(0, 0, 1)

    def test_polyface3d_init_open(self):
        """Test the initalization of Poyface3D and basic properties of open objects."""
        pts = [Point3D(0, 0, 0), Point3D(0, 2, 0), Point3D(2, 2, 0), Point3D(2, 0, 0),
               Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2)]
        face_indices = [(0, 1, 2, 3), (0, 4, 5, 1), (0, 3, 7, 4),
                        (2, 1, 5, 6), (2, 3, 7, 6)]
        polyface = Polyface3D(pts, face_indices)

        assert len(polyface.vertices) == 8
        assert len(polyface.face_indices) == 5
        assert len(polyface.faces) == 5
        assert len(polyface.edge_indices) == 12
        assert len(polyface.edges) == 12
        assert len(polyface.naked_edges) == 4
        assert len(polyface.non_manifold_edges) == 0
        assert len(polyface.internal_edges) == 8
        assert polyface.area == 20
        assert polyface.volume == 0
        assert polyface.is_solid is False

        for face in polyface.faces:
            assert face.area == 4

    def test_polyface3d_init_from_faces_solid(self):
        """Test the initalization of Poyface3D from_faces with a solid."""
        pts_1 = [Point3D(0, 0, 0), Point3D(0, 2, 0), Point3D(2, 2, 0), Point3D(2, 0, 0)]
        pts_2 = [Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(0, 2, 0)]
        pts_3 = [Point3D(0, 0, 0), Point3D(2, 0, 0), Point3D(2, 0, 2), Point3D(0, 0, 2)]
        pts_4 = [Point3D(2, 2, 0), Point3D(0, 2, 0), Point3D(0, 2, 2), Point3D(2, 2, 2)]
        pts_5 = [Point3D(2, 2, 0), Point3D(2, 0, 0), Point3D(2, 0, 2), Point3D(2, 2, 2)]
        pts_6 = [Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2)]
        face_1 = Face3D.from_vertices(pts_1)
        face_2 = Face3D.from_vertices(pts_2)
        face_3 = Face3D.from_vertices(pts_3)
        face_4 = Face3D.from_vertices(pts_4)
        face_5 = Face3D.from_vertices(pts_5)
        face_6 = Face3D.from_vertices(pts_6)
        polyface = Polyface3D.from_faces(
            [face_1, face_2, face_3, face_4, face_5, face_6])

        assert len(polyface.vertices) == 8
        assert len(polyface.face_indices) == 6
        assert len(polyface.faces) == 6
        assert len(polyface.edge_indices) == 12
        assert len(polyface.edges) == 12
        assert len(polyface.naked_edges) == 0
        assert len(polyface.non_manifold_edges) == 0
        assert len(polyface.internal_edges) == 12
        assert polyface.area == 24
        assert polyface.volume == 8
        assert polyface.is_solid

        for face in polyface.faces:
            assert face.area == 4
        assert polyface.faces[0].normal == Vector3D(0, 0, -1)
        assert polyface.faces[1].normal == Vector3D(-1, 0, 0)
        assert polyface.faces[2].normal == Vector3D(0, -1, 0)
        assert polyface.faces[3].normal == Vector3D(0, 1, 0)
        assert polyface.faces[4].normal == Vector3D(1, 0, 0)
        assert polyface.faces[5].normal == Vector3D(0, 0, 1)

    def test_polyface3d_init_from_faces_open(self):
        """Test the initalization of Poyface3D from_faces with an open object."""
        pts_1 = [Point3D(0, 0, 0), Point3D(0, 2, 0), Point3D(2, 2, 0), Point3D(2, 0, 0)]
        pts_2 = [Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(0, 2, 0)]
        pts_3 = [Point3D(0, 0, 0), Point3D(2, 0, 0), Point3D(2, 0, 2), Point3D(0, 0, 2)]
        pts_4 = [Point3D(2, 2, 0), Point3D(0, 2, 0), Point3D(0, 2, 2), Point3D(2, 2, 2)]
        pts_5 = [Point3D(2, 2, 0), Point3D(2, 0, 0), Point3D(2, 0, 2), Point3D(2, 2, 2)]
        face_1 = Face3D.from_vertices(pts_1)
        face_2 = Face3D.from_vertices(pts_2)
        face_3 = Face3D.from_vertices(pts_3)
        face_4 = Face3D.from_vertices(pts_4)
        face_5 = Face3D.from_vertices(pts_5)
        polyface = Polyface3D.from_faces([face_1, face_2, face_3, face_4, face_5])

        assert len(polyface.vertices) == 8
        assert len(polyface.face_indices) == 5
        assert len(polyface.faces) == 5
        assert len(polyface.edge_indices) == 12
        assert len(polyface.edges) == 12
        assert len(polyface.naked_edges) == 4
        assert len(polyface.non_manifold_edges) == 0
        assert len(polyface.internal_edges) == 8
        assert polyface.area == 20
        assert polyface.volume == 0
        assert polyface.is_solid is False

        for face in polyface.faces:
            assert face.area == 4

    def test_polyface3d_init_from_faces_tolerance(self):
        """Test the initalization of Poyface3D from_faces_tolerance."""
        pts_1 = [Point3D(0, 0, 0), Point3D(0, 2, 0), Point3D(2, 2, 0), Point3D(2, 0, 0)]
        pts_2 = [Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(0, 2, 0)]
        pts_3 = [Point3D(0, 0, 0), Point3D(2, 0, 0), Point3D(2, 0, 2), Point3D(0, 0, 2)]
        pts_4 = [Point3D(2, 2, 0), Point3D(0, 2, 0), Point3D(0, 2, 2), Point3D(2, 2, 2)]
        pts_5 = [Point3D(2, 2, 0), Point3D(2, 0, 0), Point3D(2, 0, 2), Point3D(2, 2, 2)]
        pts_6 = [Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2)]
        pts_7 = [Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2.0001)]
        face_1 = Face3D.from_vertices(pts_1)
        face_2 = Face3D.from_vertices(pts_2)
        face_3 = Face3D.from_vertices(pts_3)
        face_4 = Face3D.from_vertices(pts_4)
        face_5 = Face3D.from_vertices(pts_5)
        face_6 = Face3D.from_vertices(pts_6)
        face_7 = Face3D.from_vertices(pts_7)
        polyface_1 = Polyface3D.from_faces_tolerance(
            [face_1, face_2, face_3, face_4, face_5, face_6], 0.001)
        polyface_2 = Polyface3D.from_faces_tolerance(
            [face_1, face_2, face_3, face_4, face_5, face_7],  0.001)
        polyface_3 = Polyface3D.from_faces_tolerance(
            [face_1, face_2, face_3, face_4, face_5, face_7],  0.000001)

        assert polyface_1.is_solid
        assert polyface_2.is_solid
        assert not polyface_3.is_solid

    def test_polyface3d_init_from_box(self):
        """Test the initalization of Poyface3D from_box."""
        polyface = Polyface3D.from_box(2, 4, 2)

        assert len(polyface.vertices) == 8
        assert len(polyface.face_indices) == 6
        assert len(polyface.faces) == 6
        assert len(polyface.edge_indices) == 12
        assert len(polyface.edges) == 12
        assert len(polyface.naked_edges) == 0
        assert len(polyface.non_manifold_edges) == 0
        assert len(polyface.internal_edges) == 12
        assert polyface.area == 40
        assert polyface.volume == 16
        assert polyface.is_solid

    def test_is_solid_with_hole(self):
        """Test the is_solid property for a polyface with a hole.

        This ensures that the is_solid property still works where the Euler
        characteristic fails.
        """
        pts_1 = [Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(1, 3, 1), Point3D(0, 4, 0)]
        pts_2 = [Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(3, 1, 1), Point3D(4, 0, 0)]
        pts_3 = [Point3D(4, 4, 0), Point3D(3, 3, 1), Point3D(1, 3, 1), Point3D(0, 4, 0)]
        pts_4 = [Point3D(4, 4, 0), Point3D(3, 3, 1), Point3D(3, 1, 1), Point3D(4, 0, 0)]
        pts_5 = [Point3D(0, 0, 2), Point3D(1, 1, 1), Point3D(1, 3, 1), Point3D(0, 4, 2)]
        pts_6 = [Point3D(0, 0, 2), Point3D(1, 1, 1), Point3D(3, 1, 1), Point3D(4, 0, 2)]
        pts_7 = [Point3D(4, 4, 2), Point3D(3, 3, 1), Point3D(1, 3, 1), Point3D(0, 4, 2)]
        pts_8 = [Point3D(4, 4, 2), Point3D(3, 3, 1), Point3D(3, 1, 1), Point3D(4, 0, 2)]
        pts_9 = [Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(0, 4, 2), Point3D(0, 4, 0)]
        pts_10 = [Point3D(0, 0, 0), Point3D(4, 0, 0), Point3D(4, 0, 2), Point3D(0, 0, 2)]
        pts_11 = [Point3D(4, 4, 0), Point3D(0, 4, 0), Point3D(0, 4, 2), Point3D(4, 4, 2)]
        pts_12 = [Point3D(4, 4, 0), Point3D(4, 0, 0), Point3D(4, 0, 2), Point3D(4, 4, 2)]
        face_1 = Face3D.from_vertices(pts_1)
        face_2 = Face3D.from_vertices(pts_2)
        face_3 = Face3D.from_vertices(pts_3)
        face_4 = Face3D.from_vertices(pts_4)
        face_5 = Face3D.from_vertices(pts_5)
        face_6 = Face3D.from_vertices(pts_6)
        face_7 = Face3D.from_vertices(pts_7)
        face_8 = Face3D.from_vertices(pts_8)
        face_9 = Face3D.from_vertices(pts_9)
        face_10 = Face3D.from_vertices(pts_10)
        face_11 = Face3D.from_vertices(pts_11)
        face_12 = Face3D.from_vertices(pts_12)
        polyface = Polyface3D.from_faces([face_1, face_2, face_3, face_4, face_5,
                                          face_6, face_7, face_8, face_9, face_10,
                                          face_11, face_12])
        assert len(polyface.faces) + len(polyface.vertices) - len(polyface.edges) != 2
        assert polyface.area == pytest.approx(57.4558, rel=1e-3)
        assert polyface.volume == pytest.approx(15.333, rel=1e-3)
        assert polyface.is_solid

    def test_min_max_center(self):
        """Test the Face3D min, max and center."""
        polyface_1 = Polyface3D.from_box(2, 4, 2)
        polyface_2 = Polyface3D.from_box(math.sqrt(2), math.sqrt(2), 2, Plane(
            Vector3D(0, 0, 1), Point3D(1, 0, 0), Vector3D(1, 1, 0)))

        assert polyface_1.min == Point3D(0, 0, 0)
        assert polyface_1.max == Point3D(2, 4, 2)
        assert polyface_1.center == Point3D(1, 2, 1)
        assert polyface_1.volume == pytest.approx(16, rel=1e-3)

        assert polyface_2.min == Point3D(0, 0, 0)
        assert polyface_2.max == Point3D(2, 2, 2)
        assert polyface_2.center == Point3D(1, 1, 1)
        assert polyface_2.volume == pytest.approx(4, rel=1e-3)

    def test_duplicate(self):
        """Test the duplicate method of Face3D."""
        polyface = Polyface3D.from_box(2, 4, 2)
        new_polyface = polyface.duplicate()

        for i, pt in enumerate(polyface):
            assert pt == new_polyface[i]
        for i, fi in enumerate(polyface.face_indices):
            assert fi == new_polyface.face_indices[i]

        assert polyface.area == new_polyface.area
        assert polyface.is_solid == new_polyface.is_solid

    def test_move(self):
        """Test the Polyface3D move method."""
        polyface = Polyface3D.from_box(2, 2, 2)

        vec_1 = Vector3D(2, 2, 2)
        new_polyface = polyface.move(vec_1)
        assert new_polyface[0] == Point3D(2, 2, 2)
        assert new_polyface[1] == Point3D(2, 4, 2)
        assert new_polyface[2] == Point3D(4, 4, 2)
        assert new_polyface[3] == Point3D(4, 2, 2)

        assert polyface.area == new_polyface.area
        assert polyface.volume == new_polyface.volume

    def test_is_point_inside(self):
        """Test the is_point_inside method."""
        pts = [Point3D(0, 0, 0), Point3D(0, 2, 0), Point3D(2, 2, 0), Point3D(2, 0, 0),
               Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2)]
        face_indices = [(0, 1, 2, 3), (0, 4, 5, 1), (0, 3, 7, 4),
                        (2, 1, 5, 6), (2, 3, 7, 6), (4, 5, 6, 7)]
        polyface = Polyface3D(pts, face_indices)

        assert polyface.is_point_inside(Point3D(1, 1, 1)) is True
        assert polyface.is_point_inside(Point3D(4, 1, 1)) is False
        assert polyface.is_point_inside(Point3D(-1, 1, 1)) is False
        assert polyface.is_point_inside(Point3D(4, 4, 4)) is False


if __name__ == "__main__":
    unittest.main()
