# coding=utf-8

from ladybug_geometry.geometry2d.pointvector import Point2D
from ladybug_geometry.geometry2d.mesh import Mesh2D
from ladybug_geometry.geometry2d.polygon import Polygon2D

import unittest
import pytest


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
            Mesh2D(pts, (0, 1, 2, 3, 5))  # incorrect input type for face

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


if __name__ == "__main__":
    unittest.main()
