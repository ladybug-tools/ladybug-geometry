# coding=utf-8

from ladybug_geometry.geometry2d.pointvector import Point2D, Vector2D
from ladybug_geometry.geometry2d.mesh import Mesh2D
from ladybug_geometry.geometry2d.polygon import Polygon2D

import unittest
import pytest
import math


class Mesh2DTestCase(unittest.TestCase):
    """Test for Mesh2D"""

    def test_mesh2d_init(self):
        """Test the initalization of Mesh2D objects and basic properties."""
        pts = (Point2D(0, 0), Point2D(0, 2), Point2D(2, 2), Point2D(2, 0))
        mesh = Mesh2D(pts, [(0, 1, 2, 3)])
        str(mesh)  # test the string representation of the object

        assert len(mesh.vertices) == 4
        assert len(mesh.faces) == 1
        assert mesh[0] == Point2D(0, 0)
        assert mesh[1] == Point2D(0, 2)
        assert mesh[2] == Point2D(2, 2)
        assert mesh[3] == Point2D(2, 0)
        assert mesh.area == 4

        assert mesh.min == Point2D(0, 0)
        assert mesh.max == Point2D(2, 2)
        assert mesh.center == Point2D(1, 1)
        assert mesh.centroid == Point2D(1, 1)

        assert len(mesh.face_areas) == 1
        assert mesh.face_areas[0] == 4
        assert len(mesh.face_centroids) == 1
        assert mesh.face_centroids[0] == Point2D(1, 1)

        assert mesh._is_color_by_face is False
        assert mesh.colors is None

    def test_mesh2d_incorrect(self):
        """Test the initalization of Mesh2D objects with incorrect values."""
        pts = (Point2D(0, 0), Point2D(0, 2), Point2D(2, 2), Point2D(2, 0), Point2D(4, 0))
        with pytest.raises(AssertionError):
            Mesh2D(pts, [(0, 1, 2, 3, 5)])  # too many vertices in a face
        with pytest.raises(AssertionError):
            Mesh2D(pts, [])  # we need at least one face
        with pytest.raises(AssertionError):
            Mesh2D(pts, (0, 1, 2, 3))  # incorrect input type for face
        with pytest.raises(IndexError):
            Mesh2D(pts, [(0, 1, 2, 6)])  # incorrect index used by face
        with pytest.raises(TypeError):
            Mesh2D(pts, [(0.0, 1, 2, 6)])  # incorrect use of floats for face index

    def test_mesh2d_init_concave(self):
        """Test the initalization of Mesh2D objects with a concave quad face."""
        pts = (Point2D(0, 0), Point2D(0, 2), Point2D(1, 1), Point2D(2, 0))
        mesh = Mesh2D(pts, [(0, 1, 2, 3)])

        assert len(mesh.vertices) == 4
        assert len(mesh.faces) == 1
        assert mesh[0] == Point2D(0, 0)
        assert mesh[1] == Point2D(0, 2)
        assert mesh[2] == Point2D(1, 1)
        assert mesh[3] == Point2D(2, 0)
        assert mesh.area == 2

        assert mesh.min == Point2D(0, 0)
        assert mesh.max == Point2D(2, 2)
        assert mesh.center == Point2D(1, 1)
        assert mesh.centroid.x == pytest.approx(0.667, rel=1e-2)
        assert mesh.centroid.y == pytest.approx(0.667, rel=1e-2)

        assert len(mesh.face_areas) == 1
        assert mesh.face_areas[0] == 2
        assert len(mesh.face_centroids) == 1

        mesh = Mesh2D(pts, [(3, 2, 1, 0)])
        assert len(mesh.vertices) == 4
        assert len(mesh.faces) == 1
        assert mesh.area == 2

    def test_mesh2d_init_two_faces(self):
        """Test the initalization of Mesh2D objects with two faces."""
        pts = (Point2D(0, 0), Point2D(0, 2), Point2D(2, 2), Point2D(2, 0), Point2D(4, 0))
        mesh = Mesh2D(pts, [(0, 1, 2, 3), (2, 3, 4)])

        assert len(mesh.vertices) == 5
        assert len(mesh.faces) == 2
        assert mesh[0] == Point2D(0, 0)
        assert mesh[1] == Point2D(0, 2)
        assert mesh[2] == Point2D(2, 2)
        assert mesh[3] == Point2D(2, 0)
        assert mesh[4] == Point2D(4, 0)
        assert mesh.area == 6

        assert mesh.min == Point2D(0, 0)
        assert mesh.max == Point2D(4, 2)
        assert mesh.center == Point2D(2, 1)
        assert mesh.centroid.x == pytest.approx(1.56, rel=1e-2)
        assert mesh.centroid.y == pytest.approx(0.89, rel=1e-2)

        assert len(mesh.face_areas) == 2
        assert mesh.face_areas[0] == 4
        assert mesh.face_areas[1] == 2
        assert len(mesh.face_centroids) == 2
        assert mesh.face_centroids[0] == Point2D(1, 1)
        assert mesh.face_centroids[1].x == pytest.approx(2.67, rel=1e-2)
        assert mesh.face_centroids[1].y == pytest.approx(0.67, rel=1e-2)

        assert mesh._is_color_by_face is False
        assert mesh.colors is None

    def test_mesh2d_init_from_faces(self):
        """Test the initalization of Mesh2D from_faces."""
        face_1 = (Point2D(0, 0), Point2D(0, 2), Point2D(2, 2), Point2D(2, 0))
        face_2 = (Point2D(2, 2), Point2D(2, 0), Point2D(4, 0))
        mesh_1 = Mesh2D.from_faces([face_1, face_2])
        mesh_2 = Mesh2D.from_faces([face_1, face_2], False)

        assert len(mesh_1.vertices) == 5
        assert len(mesh_2.vertices) == 7
        assert len(mesh_1.faces) == len(mesh_2.faces) == 2
        assert mesh_1.area == mesh_2.area == 6

        assert mesh_1.min == mesh_2.min == Point2D(0, 0)
        assert mesh_1.max == mesh_2.max == Point2D(4, 2)
        assert mesh_1.center == mesh_2.center == Point2D(2, 1)
        assert mesh_1.centroid.x == mesh_2.centroid.x == pytest.approx(1.56, rel=1e-2)
        assert mesh_1.centroid.y == mesh_2.centroid.y == pytest.approx(0.89, rel=1e-2)

        assert len(mesh_1.face_areas) == len(mesh_2.face_areas) == 2
        assert mesh_1.face_areas[0] == mesh_2.face_areas[0] == 4
        assert mesh_1.face_areas[1] == mesh_2.face_areas[1] == 2
        assert len(mesh_1.face_centroids) == len(mesh_2.face_centroids) == 2
        assert mesh_1.face_centroids[0] == mesh_2.face_centroids[0] == Point2D(1, 1)
        assert mesh_1.face_centroids[1].x == mesh_2.face_centroids[1].x == pytest.approx(2.67, rel=1e-2)
        assert mesh_1.face_centroids[1].y == mesh_2.face_centroids[1].y == pytest.approx(0.67, rel=1e-2)

        assert mesh_1._is_color_by_face is mesh_1._is_color_by_face is False
        assert mesh_1.colors is mesh_1.colors is None

    def test_mesh2d_init_from_polygon_triangulated(self):
        """Test the initalization of Mesh2D from_polygon_triangulated."""
        verts = (Point2D(0, 0), Point2D(0, 2), Point2D(2, 2), Point2D(4, 0))
        polygon = Polygon2D(verts)
        mesh = Mesh2D.from_polygon_triangulated(polygon)

        assert len(mesh.vertices) == 4
        assert len(mesh.faces) == 2
        assert mesh.area == 6

        assert mesh.min == Point2D(0, 0)
        assert mesh.max == Point2D(4, 2)
        assert mesh.center == Point2D(2, 1)
        assert mesh.centroid.x == pytest.approx(1.56, rel=1e-2)
        assert mesh.centroid.y == pytest.approx(0.89, rel=1e-2)

        assert len(mesh.face_areas) == 2
        assert mesh.face_areas[0] == 2
        assert mesh.face_areas[1] == 4
        assert len(mesh.face_centroids) == 2

        assert mesh._is_color_by_face is False
        assert mesh.colors is None

    def test_mesh2d_init_from_polygon_triangulated_colinear(self):
        """Test Mesh2D from_polygon_triangulated with some colinear vertices."""
        verts = (Point2D(0, 0), Point2D(0, 2), Point2D(2, 2), Point2D(4, 0), Point2D(2, 0))
        polygon = Polygon2D(verts)
        mesh = Mesh2D.from_polygon_triangulated(polygon)

        assert len(mesh.vertices) == 5
        assert len(mesh.faces) == 3
        assert mesh.area == 6

        assert mesh.min == Point2D(0, 0)
        assert mesh.max == Point2D(4, 2)
        assert mesh.center == Point2D(2, 1)
        assert mesh.centroid.x == pytest.approx(1.56, rel=1e-2)
        assert mesh.centroid.y == pytest.approx(0.89, rel=1e-2)

    def test_mesh2d_init_from_polygon_triangulated_concave(self):
        """Test Mesh2D from_polygon_triangulated with a concave polygon."""
        verts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 1), Point2D(1, 1),
                 Point2D(1, 2), Point2D(0, 2))
        polygon = Polygon2D(verts)
        mesh_1 = Mesh2D.from_polygon_triangulated(polygon)
        mesh_2 = Mesh2D.from_polygon_triangulated(polygon, False)

        assert len(mesh_1.vertices) == 6
        assert len(mesh_2.vertices) == 12
        assert len(mesh_1.faces) == len(mesh_2.faces) == 4
        assert mesh_1.area == mesh_2.area == 3

        assert mesh_1.min == mesh_2.min == Point2D(0, 0)
        assert mesh_1.max == mesh_2.max == Point2D(2, 2)
        assert mesh_1.center == mesh_2.center == Point2D(1, 1)
        assert mesh_1.centroid.x == mesh_2.centroid.x == pytest.approx(0.8333, rel=1e-2)
        assert mesh_1.centroid.y == mesh_2.centroid.y == pytest.approx(0.8333, rel=1e-2)

        assert len(mesh_1.face_areas) == len(mesh_2.face_areas) == 4
        assert len(mesh_1.face_centroids) == len(mesh_2.face_centroids) == 4

    def test_mesh2d_init_from_polygon_triangulated_incorrect(self):
        """Test the initalization of Mesh2D from_polygon_triangulated."""
        verts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 1), Point2D(1, 1),
                 Point2D(1, 2), Point2D(2, 0))
        polygon = Polygon2D(verts)
        with pytest.raises(ValueError):
            Mesh2D.from_polygon_triangulated(polygon)

    def test_mesh2d_init_from_polygon_grid(self):
        """Test the initalization of Mesh2D from_polygon_grid."""
        verts = (Point2D(0, 0), Point2D(0, 2), Point2D(2, 2), Point2D(2, 0))
        polygon = Polygon2D(verts)
        mesh = Mesh2D.from_polygon_grid(polygon, 0.5, 0.5)

        assert len(mesh.vertices) == 25
        assert len(mesh.faces) == 16
        assert mesh.area == 4

        assert mesh.min.x == pytest.approx(0.0000001, rel=1e-2)
        assert mesh.min.y == pytest.approx(0.0000001, rel=1e-2)
        assert mesh.max.x == pytest.approx(2., rel=1e-2)
        assert mesh.max.y == pytest.approx(2., rel=1e-2)
        assert mesh.center.x == pytest.approx(1, rel=1e-2)
        assert mesh.center.y == pytest.approx(1, rel=1e-2)
        assert mesh.centroid.x == pytest.approx(1, rel=1e-2)
        assert mesh.centroid.y == pytest.approx(1, rel=1e-2)

        assert len(mesh.face_areas) == 16
        assert mesh.face_areas[0] == 0.25
        assert len(mesh.face_centroids) == 16
        assert mesh._is_color_by_face is False
        assert mesh.colors is None

    def test_mesh2d_init_from_polygon_grid_concave(self):
        """Test the initalization of Mesh2D from_polygon_grid."""
        verts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 1), Point2D(1, 1),
                 Point2D(1, 2), Point2D(0, 2))
        polygon = Polygon2D(verts)
        mesh = Mesh2D.from_polygon_grid(polygon, 0.5, 0.5, False)

        assert len(mesh.vertices) == 21
        assert len(mesh.faces) == 12
        assert mesh.area == 3

        assert mesh.min.x == pytest.approx(0.0000001, rel=1e-2)
        assert mesh.min.y == pytest.approx(0.0000001, rel=1e-2)
        assert mesh.max.x == pytest.approx(2., rel=1e-2)
        assert mesh.max.y == pytest.approx(2., rel=1e-2)
        assert mesh.center.x == pytest.approx(1, rel=1e-2)
        assert mesh.center.y == pytest.approx(1, rel=1e-2)
        assert mesh.centroid.x == pytest.approx(0.83, rel=1e-2)
        assert mesh.centroid.y == pytest.approx(0.83, rel=1e-2)

        assert len(mesh.face_areas) == 12
        assert mesh.face_areas[0] == 0.25
        assert len(mesh.face_centroids) == 12
        assert mesh._is_color_by_face is False
        assert mesh.colors is None

    def test_mesh2d_init_from_grid(self):
        """Test the initalization of Mesh2D from_grid."""
        mesh = Mesh2D.from_grid(Point2D(1, 1), 8, 2, 0.25, 1)

        assert len(mesh.vertices) == 27
        assert len(mesh.faces) == 16
        assert mesh.area == 4

        assert mesh.min == Point2D(1, 1)
        assert mesh.max == Point2D(3, 3)
        assert mesh.center == Point2D(2, 2)
        assert mesh.centroid == Point2D(2, 2)

        assert len(mesh.face_areas) == 16
        assert mesh.face_areas[0] == 0.25
        assert len(mesh.face_centroids) == 16
        assert mesh._is_color_by_face is False
        assert mesh.colors is None

    def test_triangualted(self):
        """Test the Mesh2D triangualted method."""
        pts = (Point2D(0, 0), Point2D(0, 2), Point2D(2, 2), Point2D(2, 0), Point2D(4, 0))
        mesh = Mesh2D(pts, [(0, 1, 2, 3), (2, 3, 4)], ['red', 'green'])

        assert len(mesh.vertices) == 5
        assert len(mesh.faces) == 2
        assert len(mesh.colors) == 2
        assert mesh.area == 6

        tri_mesh = mesh.triangulated()
        assert len(tri_mesh.vertices) == 5
        assert len(tri_mesh.faces) == 3
        assert len(tri_mesh.colors) == 3
        assert tri_mesh.area == 6

    def test_remove_vertices(self):
        """Test the Mesh2D remove_vertices method."""
        mesh = Mesh2D.from_grid(Point2D(1, 1), 8, 2, 0.25, 1)
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
        """Test the Mesh2D remove_faces method."""
        mesh = Mesh2D.from_grid(Point2D(1, 1), 8, 2, 0.25, 1)
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
        """Test the Mesh2D remove_faces method."""
        mesh = Mesh2D.from_grid(Point2D(1, 1), 8, 2, 0.25, 1)
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
        """Test the Mesh2D move method."""
        pts = (Point2D(0, 0), Point2D(0, 2), Point2D(2, 2), Point2D(2, 0), Point2D(4, 0))
        mesh = Mesh2D(pts, [(0, 1, 2, 3), (2, 3, 4)])

        vec_1 = Vector2D(2, 2)
        new_mesh = mesh.move(vec_1)
        assert new_mesh[0] == Point2D(2, 2)
        assert new_mesh[1] == Point2D(2, 4)
        assert new_mesh[2] == Point2D(4, 4)
        assert new_mesh[3] == Point2D(4, 2)
        assert new_mesh[4] == Point2D(6, 2)

        assert mesh.area == new_mesh.area
        assert len(mesh.vertices) == len(new_mesh.vertices)
        assert len(mesh.faces) == len(new_mesh.faces)

    def test_scale(self):
        """Test the Mesh2D scale method."""
        pts = (Point2D(0, 0), Point2D(0, 2), Point2D(2, 2), Point2D(2, 0), Point2D(4, 0))
        mesh = Mesh2D(pts, [(0, 1, 2, 3), (2, 3, 4)])
        origin_1 = Point2D(2, 0)

        new_mesh_1 = mesh.scale(2, origin_1)
        assert new_mesh_1[0] == Point2D(-2, 0)
        assert new_mesh_1[1] == Point2D(-2, 4)
        assert new_mesh_1[2] == Point2D(2, 4)
        assert new_mesh_1[3] == Point2D(2, 0)
        assert new_mesh_1[4] == Point2D(6, 0)
        assert new_mesh_1.area == 24
        assert len(mesh.vertices) == len(new_mesh_1.vertices)
        assert len(mesh.faces) == len(new_mesh_1.faces)

    def test_scale_world_origin(self):
        """Test the Mesh2D scale_world_origin method."""
        pts = (Point2D(0, 0), Point2D(0, 2), Point2D(2, 2), Point2D(2, 0), Point2D(4, 0))
        mesh = Mesh2D(pts, [(0, 1, 2, 3), (2, 3, 4)])

        new_mesh_1 = mesh.scale_world_origin(2)
        assert new_mesh_1[0] == Point2D(0, 0)
        assert new_mesh_1[1] == Point2D(0, 4)
        assert new_mesh_1[2] == Point2D(4, 4)
        assert new_mesh_1[3] == Point2D(4, 0)
        assert new_mesh_1[4] == Point2D(8, 0)
        assert new_mesh_1.area == 24
        assert len(mesh.vertices) == len(new_mesh_1.vertices)
        assert len(mesh.faces) == len(new_mesh_1.faces)

    def test_rotate(self):
        """Test the Mesh2D rotate method."""
        pts = (Point2D(0, 0), Point2D(0, 2), Point2D(2, 2), Point2D(2, 0), Point2D(4, 0))
        mesh = Mesh2D(pts, [(0, 1, 2, 3), (2, 3, 4)])
        origin_1 = Point2D(1, 1)

        test_1 = mesh.rotate(math.pi, origin_1)
        assert test_1[0].x == pytest.approx(2, rel=1e-3)
        assert test_1[0].y == pytest.approx(2, rel=1e-3)
        assert test_1[2].x == pytest.approx(0, rel=1e-3)
        assert test_1[2].y == pytest.approx(0, rel=1e-3)
        assert mesh.area == test_1.area
        assert len(mesh.vertices) == len(test_1.vertices)
        assert len(mesh.faces) == len(test_1.faces)

        test_2 = mesh.rotate(math.pi/2, origin_1)
        assert test_2[0].x == pytest.approx(2, rel=1e-3)
        assert test_2[0].y == pytest.approx(0, rel=1e-3)
        assert test_2[2].x == pytest.approx(0, rel=1e-3)
        assert test_2[2].y == pytest.approx(2, rel=1e-3)
        assert mesh.area == test_2.area
        assert len(mesh.vertices) == len(test_2.vertices)
        assert len(mesh.faces) == len(test_2.faces)

    def test_reflect(self):
        """Test the Mesh2D reflect method."""
        pts = (Point2D(1, 1), Point2D(1, 2), Point2D(2, 2), Point2D(2, 1), Point2D(3, 1))
        mesh = Mesh2D(pts, [(0, 1, 2, 3), (2, 3, 4)])

        origin_1 = Point2D(1, 0)
        normal_1 = Vector2D(1, 0)
        normal_2 = Vector2D(-1, -1).normalized()

        test_1 = mesh.reflect(normal_1, origin_1)
        assert test_1[0].x == pytest.approx(1, rel=1e-3)
        assert test_1[0].y == pytest.approx(1, rel=1e-3)
        assert test_1[2].x == pytest.approx(0, rel=1e-3)
        assert test_1[2].y == pytest.approx(2, rel=1e-3)
        assert mesh.area == test_1.area
        assert len(mesh.vertices) == len(test_1.vertices)
        assert len(mesh.faces) == len(test_1.faces)

        test_1 = mesh.reflect(normal_2, Point2D(0, 0))
        assert test_1[0].x == pytest.approx(-1, rel=1e-3)
        assert test_1[0].y == pytest.approx(-1, rel=1e-3)
        assert test_1[2].x == pytest.approx(-2, rel=1e-3)
        assert test_1[2].y == pytest.approx(-2, rel=1e-3)
        assert mesh.area == test_1.area
        assert len(mesh.vertices) == len(test_1.vertices)
        assert len(mesh.faces) == len(test_1.faces)

        test_2 = mesh.reflect(normal_2, origin_1)
        assert test_2[0].x == pytest.approx(0, rel=1e-3)
        assert test_2[0].y == pytest.approx(0, rel=1e-3)
        assert test_2[2].x == pytest.approx(-1, rel=1e-3)
        assert test_2[2].y == pytest.approx(-1, rel=1e-3)
        assert mesh.area == test_2.area
        assert len(mesh.vertices) == len(test_2.vertices)
        assert len(mesh.faces) == len(test_2.faces)


if __name__ == "__main__":
    unittest.main()
