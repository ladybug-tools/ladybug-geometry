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
        assert polyface.is_solid is False

        for face in polyface.faces:
            assert face.area == 4

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
        assert polyface.is_solid

    def test_is_solid_with_hole(self):
        """Test the is_solid property for a polyface with a hole.

        This ensures that the is_solid property still works where the Euler
        characteristic fails.
        """
        pass

    def test_min_max_center(self):
        """Test the Face3D min, max and center."""
        polyface_1 = Polyface3D.from_box(2, 4, 2)
        polyface_2 = Polyface3D.from_box(math.sqrt(2), math.sqrt(2), 2, Plane(
            Vector3D(0, 0, 1), Point3D(1, 0, 0), Vector3D(1, 1, 0)))

        assert polyface_1.min == Point3D(0, 0, 0)
        assert polyface_1.max == Point3D(2, 4, 2)
        assert polyface_1.center == Point3D(1, 2, 1)

        assert polyface_2.min == Point3D(0, 0, 0)
        assert polyface_2.max == Point3D(2, 2, 2)
        assert polyface_2.center == Point3D(1, 1, 1)

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
