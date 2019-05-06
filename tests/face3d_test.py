# coding=utf-8

from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D, \
    Point3DImmutable
from ladybug_geometry.geometry3d.plane import Plane
from ladybug_geometry.geometry3d.line import LineSegment3D, LineSegment3DImmutable
from ladybug_geometry.geometry3d.face import Face3D


import unittest
import pytest


class Face3DTestCase(unittest.TestCase):
    """Test for Face3D"""

    def test_face3d_init(self):
        """Test the initalization of Face3D objects and basic properties."""
        pts = (Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2))
        plane = Plane(Vector3D(0, 0, 1), Point3D(0, 0, 2))
        face = Face3D(pts, plane)
        str(face)  # test the string representation of the face

        assert isinstance(face.plane, Plane)
        assert face.plane.n == Vector3D(0, 0, 1)
        assert face.plane.n == face.normal
        assert face.plane.o == Point3D(0, 0, 2)

        assert isinstance(face.vertices, tuple)
        assert len(face.vertices) == 4
        assert len(face) == 4
        for point in face:
            assert isinstance(point, Point3DImmutable)

        assert isinstance(face.boundary_segments, tuple)
        assert len(face.boundary_segments) == 4
        for seg in face.boundary_segments:
            assert isinstance(seg, LineSegment3DImmutable)
            assert seg.length == 2
        assert face.has_holes is False
        assert face.hole_segments is None

        assert face.area == 4
        assert face.perimeter == 8
        assert face.is_clockwise is True
        assert face.is_convex is True
        assert face.is_self_intersecting is False
        assert face.vertices[0] == face[0]

    def test_face3d_init_from_vertices(self):
        """Test the initalization of Face3D objects from_vertices."""
        pts = (Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2))
        face = Face3D.from_vertices(pts)

        assert isinstance(face.plane, Plane)
        assert face.plane.n == Vector3D(0, 0, -1)
        assert face.plane.n == face.normal
        assert face.plane.o == Point3D(0, 0, 2)

        assert isinstance(face.vertices, tuple)
        assert len(face.vertices) == 4
        assert len(face) == 4
        for point in face:
            assert isinstance(point, Point3DImmutable)

        assert isinstance(face.boundary_segments, tuple)
        assert len(face.boundary_segments) == 4
        for seg in face.boundary_segments:
            assert isinstance(seg, LineSegment3DImmutable)
            assert seg.length == 2
        assert face.has_holes is False
        assert face.hole_segments is None

        assert face.area == 4
        assert face.perimeter == 8
        assert face.is_clockwise is False
        assert face.is_convex is True
        assert face.is_self_intersecting is False
        assert face.vertices[0] == face[0]

    def test_face3d_init_from_extrusion(self):
        """Test the initalization of Face3D from_extrusion."""
        line_seg = LineSegment3D(Point3D(0, 0, 0), Vector3D(2, 0, 0))
        extru_vec = Vector3D(0, 0, 2)
        face = Face3D.from_extrusion(line_seg, extru_vec)

        assert isinstance(face.plane, Plane)
        assert face.plane.is_coplanar(Plane(Vector3D(0, 1, 0), Point3D(0, 0, 0)))
        assert face.plane.n == face.normal
        assert face.plane.o == Point3D(0, 0, 0)

        assert isinstance(face.vertices, tuple)
        assert len(face.vertices) == 4
        assert len(face) == 4
        for point in face:
            assert isinstance(point, Point3DImmutable)

        assert isinstance(face.boundary_segments, tuple)
        assert len(face.boundary_segments) == 4
        for seg in face.boundary_segments:
            assert isinstance(seg, LineSegment3DImmutable)
            assert seg.length == 2
        assert face.has_holes is False
        assert face.hole_segments is None

        assert face.area == 4
        assert face.perimeter == 8
        assert face.is_clockwise is True
        assert face.is_convex is True
        assert face.is_self_intersecting is False

    def test_face3d_init_from_rectangle(self):
        """Test the initalization of Face3D from_rectangle."""
        plane = Plane(Vector3D(0, 0, 1), Point3D(2, 2, 2))
        face = Face3D.from_rectangle(2, 2, plane)

        assert isinstance(face.plane, Plane)
        assert face.plane.n == Vector3D(0, 0, 1)
        assert face.plane.n == face.normal
        assert face.plane.o == Point3D(2, 2, 2)

        assert isinstance(face.vertices, tuple)
        assert len(face.vertices) == 4
        for point in face.vertices:
            assert isinstance(point, Point3DImmutable)

        assert isinstance(face.boundary_segments, tuple)
        assert len(face.boundary_segments) == 4
        for seg in face.boundary_segments:
            assert isinstance(seg, LineSegment3DImmutable)
            assert seg.length == 2
        assert face.has_holes is False
        assert face.hole_segments is None

        assert face.area == 4
        assert face.perimeter == 8
        assert face.is_clockwise is True
        assert face.is_convex is True
        assert face.is_self_intersecting is False

    def test_face3d_init_from_regular_polygon(self):
        """Test the initalization of Face3D from_regular_polygon."""
        plane = Plane(Vector3D(0, 0, 1), Point3D(2, 2, 2))
        face = Face3D.from_regular_polygon(8, 2, plane)

        assert isinstance(face.plane, Plane)
        assert face.plane.n == Vector3D(0, 0, 1)
        assert face.plane.n == face.normal
        assert face.plane.o == Point3D(2, 2, 2)

        assert isinstance(face.vertices, tuple)
        assert len(face.vertices) == 8
        for point in face.vertices:
            assert isinstance(point, Point3DImmutable)
        assert isinstance(face.boundary_segments, tuple)
        assert len(face.boundary_segments) == 8
        for seg in face.boundary_segments:
            assert isinstance(seg, LineSegment3DImmutable)
            assert seg.length == pytest.approx(1.5307337, rel=1e-3)
        assert face.has_holes is False
        assert face.hole_segments is None

        assert face.area == pytest.approx(11.3137084, rel=1e-3)
        assert face.perimeter == pytest.approx(1.5307337 * 8, rel=1e-3)
        assert face.is_clockwise is False
        assert face.is_convex is True
        assert face.is_self_intersecting is False

        polygon = Face3D.from_regular_polygon(3)
        assert len(polygon.vertices) == 3
        polygon = Face3D.from_regular_polygon(20)
        assert len(polygon.vertices) == 20
        with pytest.raises(AssertionError):
            polygon = Face3D.from_regular_polygon(2)

    def test_face3d_init_from_shape_with_hole(self):
        """Test the initalization of Face3D from_shape_with_holes with one hole."""
        bound_pts = [Point3D(0, 0), Point3D(4, 0), Point3D(4, 4), Point3D(0, 4)]
        hole_pts = [Point3D(1, 1), Point3D(3, 1), Point3D(3, 3), Point3D(1, 3)]
        face = Face3D.from_shape_with_holes(bound_pts, [hole_pts])

        assert face.plane.n == Vector3D(0, 0, 1)
        assert face.plane.n == face.normal
        assert face.plane.o == Point3D(0, 0, 0)

        assert isinstance(face.vertices, tuple)
        assert len(face.vertices) == 10
        for point in face.vertices:
            assert isinstance(point, Point3DImmutable)

        assert isinstance(face.boundary_segments, tuple)
        assert len(face.boundary_segments) == 4
        for seg in face.boundary_segments:
            assert isinstance(seg, LineSegment3DImmutable)
        assert face.has_holes is True
        assert isinstance(face.hole_segments, tuple)
        assert len(face.hole_segments) == 1
        assert len(face.hole_segments[0]) == 4
        for seg in face.hole_segments[0]:
            assert isinstance(seg, LineSegment3DImmutable)

        assert face.area == 12
        assert face.perimeter == pytest.approx(24, rel=1e-3)
        assert face.is_clockwise is False
        assert face.is_convex is False
        assert face.is_self_intersecting is False

    def test_face3d_init_from_shape_with_holes(self):
        """Test the initalization of Face3D from_shape_with_holes."""
        bound_pts = [Point3D(0, 0), Point3D(4, 0), Point3D(4, 4), Point3D(0, 4)]
        hole_pts_1 = [Point3D(1, 1), Point3D(1.5, 1), Point3D(1.5, 1.5), Point3D(1, 1.5)]
        hole_pts_2 = [Point3D(2, 2), Point3D(3, 2), Point3D(3, 3), Point3D(2, 3)]
        face = Face3D.from_shape_with_holes(bound_pts, [hole_pts_1, hole_pts_2])

        assert face.plane.n == Vector3D(0, 0, 1)
        assert face.plane.n == face.normal
        assert face.plane.o == Point3D(0, 0, 0)

        assert isinstance(face.vertices, tuple)
        assert len(face.vertices) == 16
        for point in face.vertices:
            assert isinstance(point, Point3DImmutable)

        assert isinstance(face.boundary_segments, tuple)
        assert len(face.boundary_segments) == 4
        for seg in face.boundary_segments:
            assert isinstance(seg, LineSegment3DImmutable)
        assert face.has_holes is True
        assert isinstance(face.hole_segments, tuple)
        assert len(face.hole_segments) == 2

        assert face.area == 16 - 1.25
        assert face.perimeter == pytest.approx(22, rel=1e-3)
        assert face.is_clockwise is False
        assert face.is_convex is False
        assert face.is_self_intersecting is False

    def test_clockwise(self):
        """Test the clockwise property."""
        plane_1 = Plane(Vector3D(0, 0, 1))
        plane_2 = Plane(Vector3D(0, 0, -1))
        pts_1 = (Point3D(0, 0), Point3D(2, 0), Point3D(2, 2), Point3D(0, 2))
        pts_2 = (Point3D(0, 0), Point3D(0, 2), Point3D(2, 2), Point3D(2, 0))
        face_1 = Face3D(pts_1, plane_1)
        face_2 = Face3D(pts_2, plane_1)
        face_3 = Face3D(pts_1, plane_2)
        face_4 = Face3D(pts_2, plane_2)

        assert face_1.is_clockwise is face_1.polygon2d.is_clockwise is False
        assert face_2.is_clockwise is face_2.polygon2d.is_clockwise is True
        assert face_3.is_clockwise is face_3.polygon2d.is_clockwise is True
        assert face_4.is_clockwise is face_4.polygon2d.is_clockwise is False

        assert face_1.area == face_2.area == face_3.area == face_4.area == 4

    def test_is_convex(self):
        """Test the convex property."""
        plane_1 = Plane(Vector3D(0, 0, 1))
        plane_2 = Plane(Vector3D(0, 0, -1))
        pts_1 = (Point3D(0, 0), Point3D(2, 0), Point3D(2, 2), Point3D(0, 2))
        pts_2 = (Point3D(0, 0), Point3D(2, 0), Point3D(2, 1), Point3D(1, 1),
                 Point3D(1, 2), Point3D(0, 2))
        face_1 = Face3D(pts_1, plane_1)
        face_2 = Face3D(pts_2, plane_1)
        face_3 = Face3D(pts_1, plane_2)
        face_4 = Face3D(pts_2, plane_2)

        assert face_1.is_convex is True
        assert face_2.is_convex is False
        assert face_3.is_convex is True
        assert face_4.is_convex is False

    def test_is_self_intersecting(self):
        """Test the is_self_intersecting property."""
        plane_1 = Plane(Vector3D(0, 0, 1))
        plane_2 = Plane(Vector3D(0, 0, -1))
        pts_1 = (Point3D(0, 0), Point3D(2, 0), Point3D(2, 2), Point3D(0, 2))
        pts_2 = (Point3D(0, 0), Point3D(0, 2), Point3D(2, 0), Point3D(2, 2))
        face_1 = Face3D(pts_1, plane_1)
        face_2 = Face3D(pts_2, plane_1)
        face_3 = Face3D(pts_1, plane_2)
        face_4 = Face3D(pts_2, plane_2)

        assert face_1.is_self_intersecting is False
        assert face_2.is_self_intersecting is True
        assert face_3.is_self_intersecting is False
        assert face_4.is_self_intersecting is True

    def test_min_max_center(self):
        """Test the Face3D min, max and center."""
        pts_1 = (Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(2, 2, 2), Point3D(0, 2, 2))
        pts_2 = (Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(0, 0, 2))
        plane_1 = Plane(Vector3D(0, 0, 1), Point3D(0, 0, 2))
        plane_2 = Plane(Vector3D(0, 1, 0))
        face_1 = Face3D(pts_1, plane_1)
        face_2 = Face3D(pts_2, plane_2)

        assert face_1.min == Point3D(0, 0, 2)
        assert face_1.max == Point3D(2, 2, 2)
        assert face_1.center == Point3D(1, 1, 2)

        assert face_2.min == Point3D(0, 0, 0)
        assert face_2.max == Point3D(2, 0, 2)
        assert face_2.center == Point3D(1, 0, 1)

    def test_duplicate(self):
        """Test the duplicate method of Face3D."""
        pts = (Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(2, 2, 2), Point3D(0, 2, 2))
        plane_1 = Plane(Vector3D(0, 0, 1), Point3D(0, 0, 2))
        face = Face3D(pts, plane_1)
        new_face = face.duplicate()

        for i, pt in enumerate(new_face):
            assert pt == pts[i]

        assert face.area == new_face.area
        assert face.perimeter == new_face.perimeter
        assert face.is_clockwise == new_face.is_clockwise
        assert face.is_convex == new_face.is_convex
        assert face.is_self_intersecting == new_face.is_self_intersecting

    def test_triangulated_mesh_and_centroid(self):
        """Test the duplicate method of Face3D."""
        pts_1 = (Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(2, 2, 2), Point3D(0, 2, 2))
        plane_1 = Plane(Vector3D(0, 0, 1), Point3D(0, 0, 2))
        face_1 = Face3D(pts_1, plane_1)

        assert len(face_1.triangulated_mesh3d.vertices) == 4
        assert len(face_1.triangulated_mesh3d.faces) == 2
        assert face_1.triangulated_mesh3d.area == 4

        assert face_1.triangulated_mesh3d.min == Point3D(0, 0, 2)
        assert face_1.triangulated_mesh3d.max == Point3D(2, 2, 2)
        assert face_1.triangulated_mesh3d.center == Point3D(1, 1, 2)
        assert face_1.centroid == Point3D(1, 1, 2)

        pts_2 = (Point3D(0, 0), Point3D(0, 2), Point3D(2, 2),
                 Point3D(4, 0), Point3D(2, -2))
        plane_2 = Plane(Vector3D(0, 0, 1))
        face_2 = Face3D(pts_2, plane_2)

        assert len(face_2.triangulated_mesh3d.vertices) == 5
        assert len(face_2.triangulated_mesh3d.faces) == 3
        assert face_2.triangulated_mesh3d.area == 10

        assert face_2.triangulated_mesh3d.min == Point3D(0, -2, 0)
        assert face_2.triangulated_mesh3d.max == Point3D(4, 2, 0)
        assert face_2.triangulated_mesh3d.center == Point3D(2, 0, 0)
        assert face_2.centroid.x == pytest.approx(1.73, rel=1e-2)
        assert face_2.centroid.y == pytest.approx(0.2667, rel=1e-2)
        assert face_2.centroid.z == 0

    def test_validate_planarity(self):
        """Test the validate_planarity method of Face3D."""
        pts_1 = (Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(2, 2, 2), Point3D(0, 2, 2))
        pts_2 = (Point3D(0, 0, 0), Point3D(2, 0, 2), Point3D(2, 2, 2), Point3D(0, 2, 2))
        pts_3 = (Point3D(0, 0, 2.0001), Point3D(2, 0, 2), Point3D(2, 2, 2), Point3D(0, 2, 2))
        plane_1 = Plane(Vector3D(0, 0, 1), Point3D(0, 0, 2))
        face_1 = Face3D(pts_1, plane_1)
        face_2 = Face3D(pts_2, plane_1)
        face_3 = Face3D(pts_3, plane_1)

        assert face_1.validate_planarity(0.001) is True
        assert face_2.validate_planarity(0.001, False) is False
        with pytest.raises(Exception):
            face_2.validate_planarity(0.0001)
        assert face_3.validate_planarity(0.001) is True
        assert face_3.validate_planarity(0.000001, False) is False
        with pytest.raises(Exception):
            face_3.validate_planarity(0.000001)

    def test_flip(self):
        """Test the flip method of Face3D."""
        pts_1 = (Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(2, 2, 2), Point3D(0, 2, 2))
        plane_1 = Plane(Vector3D(0, 0, 1), Point3D(0, 0, 2))
        face_1 = Face3D(pts_1, plane_1)
        face_2 = face_1.flip()

        assert face_1.normal == face_2.normal.reversed()
        assert face_1.is_clockwise is False
        assert face_2.is_clockwise is True
        for i, pt in enumerate(face_1.vertices):
            assert pt == face_2[i]

    def test_move(self):
        """Test the Polygon2D move method."""
        pts_1 = (Point3D(0, 0, 0), Point3D(2, 0, 0), Point3D(2, 2, 0), Point3D(0, 2, 0))
        plane_1 = Plane(Vector3D(0, 0, 1), Point3D(0, 0, 0))
        face_1 = Face3D(pts_1, plane_1)

        vec_1 = Vector3D(2, 2, 2)
        new_face = face_1.move(vec_1)
        assert new_face[0] == Point3D(2, 2, 2)
        assert new_face[1] == Point3D(4, 2, 2)
        assert new_face[2] == Point3D(4, 4, 2)
        assert new_face[3] == Point3D(2, 4, 2)
        assert new_face.plane.o == face_1.plane.o.move(vec_1)
        assert new_face.plane.n == face_1.plane.n

        assert face_1.area == new_face.area
        assert face_1.perimeter == new_face.perimeter
        assert face_1.is_clockwise is new_face.is_clockwise
        assert face_1.is_convex is new_face.is_convex
        assert face_1.is_self_intersecting is new_face.is_self_intersecting


if __name__ == "__main__":
    unittest.main()
