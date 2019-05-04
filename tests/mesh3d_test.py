# coding=utf-8

from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
from ladybug_geometry.geometry3d.mesh import Mesh3D
from ladybug_geometry.geometry3d.plane import Plane
from ladybug_geometry.geometry2d.mesh import Mesh2D
from ladybug_geometry.geometry2d.pointvector import Point2D

import unittest
import pytest
import math


class Mesh3DTestCase(unittest.TestCase):
    """Test for Mesh3D"""

    def test_mesh3d_init(self):
        """Test the initalization of Mesh3D objects and basic properties."""
        pts = (Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2))
        mesh = Mesh3D(pts, [(0, 1, 2, 3)])
        str(mesh)  # test the string representation of the object

        assert len(mesh.vertices) == 4
        assert len(mesh.faces) == 1
        assert mesh[0] == Point3D(0, 0, 2)
        assert mesh[1] == Point3D(0, 2, 2)
        assert mesh[2] == Point3D(2, 2, 2)
        assert mesh[3] == Point3D(2, 0, 2)
        assert mesh.area == 4

        assert mesh.min == Point3D(0, 0, 2)
        assert mesh.max == Point3D(2, 2, 2)
        assert mesh.center == Point3D(1, 1, 2)

        assert len(mesh.face_areas) == 1
        assert mesh.face_areas[0] == 4
        assert len(mesh.face_centroids) == 1
        assert mesh.face_centroids[0] == Point3D(1, 1, 2)
        assert mesh._is_color_by_face is False
        assert mesh.colors is None

    def test_face_normals(self):
        """Test the Mesh3D face_normals property."""
        pts = (Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2))
        mesh = Mesh3D(pts, [(0, 1, 2, 3)])

        assert len(mesh.face_normals) == 1
        assert mesh.face_normals[0] == Vector3D(0, 0, -1)
        assert len(mesh.vertex_normals) == 4
        for vert_norm in mesh.vertex_normals:
            assert vert_norm == Vector3D(0, 0, -1)

    def test_mesh3d_incorrect(self):
        """Test the initalization of Mesh3D objects with incorrect values."""
        pts = (Point3D(0, 0), Point3D(0, 2), Point3D(2, 2), Point3D(2, 0), Point3D(4, 0))
        with pytest.raises(AssertionError):
            Mesh3D(pts, [(0, 1, 2, 3, 5)])  # too many vertices in a face
        with pytest.raises(AssertionError):
            Mesh3D(pts, [])  # we need at least one face
        with pytest.raises(AssertionError):
            Mesh3D(pts, (0, 1, 2, 3))  # incorrect input type for face
        with pytest.raises(IndexError):
            Mesh3D(pts, [(0, 1, 2, 6)])  # incorrect index used by face
        with pytest.raises(TypeError):
            Mesh3D(pts, [(0.0, 1, 2, 6)])  # incorrect use of floats for face index

    def test_mesh3d_init_two_faces(self):
        """Test the initalization of Mesh3D objects with two faces."""
        pts = (Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2),
               Point3D(2, 0, 2), Point3D(4, 0, 2))
        mesh = Mesh3D(pts, [(0, 1, 2, 3), (2, 3, 4)])

        assert len(mesh.vertices) == 5
        assert len(mesh.faces) == 2
        assert mesh[0] == Point3D(0, 0, 2)
        assert mesh[1] == Point3D(0, 2, 2)
        assert mesh[2] == Point3D(2, 2, 2)
        assert mesh[3] == Point3D(2, 0, 2)
        assert mesh[4] == Point3D(4, 0, 2)
        assert mesh.area == 6

        assert mesh.min == Point3D(0, 0, 2)
        assert mesh.max == Point3D(4, 2, 2)
        assert mesh.center == Point3D(2, 1, 2)

        assert len(mesh.face_areas) == 2
        assert mesh.face_areas[0] == 4
        assert mesh.face_areas[1] == 2
        assert len(mesh.face_centroids) == 2
        assert mesh.face_centroids[0] == Point3D(1, 1, 2)
        assert mesh.face_centroids[1].x == pytest.approx(2.67, rel=1e-2)
        assert mesh.face_centroids[1].y == pytest.approx(0.67, rel=1e-2)
        assert mesh.face_centroids[1].z == pytest.approx(2, rel=1e-2)
        assert mesh._is_color_by_face is False
        assert mesh.colors is None

    def test_mesh3d_init_from_faces(self):
        """Test the initalization of Mesh3D from_faces."""
        face_1 = (Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2))
        face_2 = (Point3D(2, 2, 2), Point3D(2, 0, 2), Point3D(4, 0, 2))
        mesh_1 = Mesh3D.from_faces([face_1, face_2])
        mesh_2 = Mesh3D.from_faces([face_1, face_2], False)

        assert len(mesh_1.vertices) == 5
        assert len(mesh_2.vertices) == 7
        assert len(mesh_1.faces) == len(mesh_2.faces) == 2
        assert mesh_1.area == mesh_2.area == 6

        assert mesh_1.min == mesh_2.min == Point3D(0, 0, 2)
        assert mesh_1.max == mesh_2.max == Point3D(4, 2, 2)
        assert mesh_1.center == mesh_2.center == Point3D(2, 1, 2)

        assert len(mesh_1.face_areas) == len(mesh_2.face_areas) == 2
        assert mesh_1.face_areas[0] == mesh_2.face_areas[0] == 4
        assert mesh_1.face_areas[1] == mesh_2.face_areas[1] == 2
        assert len(mesh_1.face_centroids) == len(mesh_2.face_centroids) == 2
        assert mesh_1.face_centroids[0] == mesh_2.face_centroids[0] == Point3D(1, 1, 2)
        assert mesh_1.face_centroids[1].x == mesh_2.face_centroids[1].x == pytest.approx(2.67, rel=1e-2)
        assert mesh_1.face_centroids[1].y == mesh_2.face_centroids[1].y == pytest.approx(0.67, rel=1e-2)
        assert mesh_1.face_centroids[1].z == mesh_2.face_centroids[1].z == pytest.approx(2, rel=1e-2)

        assert mesh_1._is_color_by_face is mesh_1._is_color_by_face is False
        assert mesh_1.colors is mesh_1.colors is None

    def test_mesh3d_from_mesh2d(self):
        """Test the initalization of Mesh3D objects from_mesh2d."""
        pts = (Point2D(0, 0), Point2D(0, 2), Point2D(2, 2), Point2D(2, 0))
        mesh_2d = Mesh2D(pts, [(0, 1, 2, 3)])
        plane = Plane(Vector3D(1, 0, 0), Point3D(0, 0, 0))

        mesh = Mesh3D.from_mesh2d(mesh_2d, plane)
        assert len(mesh.vertices) == 4
        assert len(mesh.faces) == 1
        assert mesh[0] == Point3D(0, 0, 0)
        assert mesh[1] == Point3D(0, 0, -2)
        assert mesh[2] == Point3D(0, -2, -2)
        assert mesh[3] == Point3D(0, -2, 0)
        assert mesh.area == 4
        assert mesh.min == Point3D(0, -2, -2)
        assert mesh.max == Point3D(0, 0, 0)
        assert mesh.center == Point3D(0, -1, -1)

    def test_remove_vertices(self):
        """Test the Mesh3D remove_vertices method."""
        mesh_2d = Mesh2D.from_grid(Point2D(1, 1), 8, 2, 0.25, 1)
        mesh = Mesh3D.from_mesh2d(mesh_2d)
        assert len(mesh.vertices) == 27
        assert len(mesh.faces) == 16
        assert mesh.area == 4

        pattern_1 = []
        for i in range(9):
            pattern_1.extend([True, True, False])
        mesh_1, vert_pattern = mesh.remove_vertices(pattern_1)
        assert len(mesh_1.vertices) == 18
        assert len(mesh_1.faces) == 8
        assert mesh_1.area == 2
        for face in mesh_1.faces:
            for i in face:
                mesh_1[i]  # make sure all face indices reference current vertices

    def test_remove_faces(self):
        """Test the Mesh3D remove_faces method."""
        mesh_2d = Mesh2D.from_grid(Point2D(1, 1), 8, 2, 0.25, 1)
        mesh = Mesh3D.from_mesh2d(mesh_2d)
        assert len(mesh.vertices) == 27
        assert len(mesh.faces) == 16
        assert mesh.area == 4

        pattern_1 = []
        for i in range(4):
            pattern_1.extend([True, False, False, False])
        mesh_1, vert_pattern = mesh.remove_faces(pattern_1)
        assert len(mesh_1.vertices) == 16
        assert len(mesh_1.faces) == 4
        assert mesh_1.area == 1
        for face in mesh_1.faces:
            for i in face:
                mesh_1[i]  # make sure all face indices reference current vertices

        pattern_2 = []
        for i in range(8):
            pattern_2.extend([True, False])
        mesh_2, vert_pattern = mesh.remove_faces(pattern_2)
        assert len(mesh_2.vertices) == 18
        assert len(mesh_2.faces) == 8
        assert mesh_2.area == 2
        for face in mesh_2.faces:
            for i in face:
                mesh_2[i]  # make sure all face indices reference current vertices

    def test_remove_faces_only(self):
        """Test the Mesh3D remove_faces method."""
        mesh_2d = Mesh2D.from_grid(Point2D(1, 1), 8, 2, 0.25, 1)
        mesh = Mesh3D.from_mesh2d(mesh_2d)
        assert len(mesh.vertices) == 27
        assert len(mesh.faces) == 16
        assert mesh.area == 4

        pattern_1 = []
        for i in range(4):
            pattern_1.extend([True, False, False, False])
        mesh_1 = mesh.remove_faces_only(pattern_1)
        assert len(mesh_1.vertices) == 27
        assert len(mesh_1.faces) == 4
        assert mesh_1.area == 1

        pattern_2 = []
        for i in range(8):
            pattern_2.extend([True, False])
        mesh_2 = mesh.remove_faces_only(pattern_2)
        assert len(mesh_2.vertices) == 27
        assert len(mesh_2.faces) == 8
        assert mesh_2.area == 2

    def test_move(self):
        """Test the Mesh3D move method."""
        pts = (Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2),
               Point3D(2, 0, 2), Point3D(4, 0, 2))
        mesh = Mesh3D(pts, [(0, 1, 2, 3), (2, 3, 4)])

        vec_1 = Vector3D(2, 2, -1)
        new_mesh = mesh.move(vec_1)
        assert new_mesh[0] == Point3D(2, 2, 1)
        assert new_mesh[1] == Point3D(2, 4, 1)
        assert new_mesh[2] == Point3D(4, 4, 1)
        assert new_mesh[3] == Point3D(4, 2, 1)
        assert new_mesh[4] == Point3D(6, 2, 1)

        assert mesh.area == new_mesh.area
        assert len(mesh.vertices) == len(new_mesh.vertices)
        assert len(mesh.faces) == len(new_mesh.faces)

    def test_scale(self):
        """Test the Mesh3D scale method."""
        pts = (Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2),
               Point3D(2, 0, 2), Point3D(4, 0, 2))
        mesh = Mesh3D(pts, [(0, 1, 2, 3), (2, 3, 4)])
        origin_1 = Point3D(2, 0, 2)

        new_mesh_1 = mesh.scale(2, origin_1)
        assert new_mesh_1[0] == Point3D(-2, 0, 2)
        assert new_mesh_1[1] == Point3D(-2, 4, 2)
        assert new_mesh_1[2] == Point3D(2, 4, 2)
        assert new_mesh_1[3] == Point3D(2, 0, 2)
        assert new_mesh_1[4] == Point3D(6, 0, 2)
        assert new_mesh_1.area == 24
        assert len(mesh.vertices) == len(new_mesh_1.vertices)
        assert len(mesh.faces) == len(new_mesh_1.faces)

    def test_scale_world_origin(self):
        """Test the Mesh2D scale_world_origin method."""
        pts = (Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2),
               Point3D(2, 0, 2), Point3D(4, 0, 2))
        mesh = Mesh3D(pts, [(0, 1, 2, 3), (2, 3, 4)])

        new_mesh_1 = mesh.scale_world_origin(2)
        assert new_mesh_1[0] == Point3D(0, 0, 4)
        assert new_mesh_1[1] == Point3D(0, 4, 4)
        assert new_mesh_1[2] == Point3D(4, 4, 4)
        assert new_mesh_1[3] == Point3D(4, 0, 4)
        assert new_mesh_1[4] == Point3D(8, 0, 4)
        assert new_mesh_1.area == 24
        assert len(mesh.vertices) == len(new_mesh_1.vertices)
        assert len(mesh.faces) == len(new_mesh_1.faces)


if __name__ == "__main__":
    unittest.main()
