# coding=utf-8

from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D, \
    Point3DImmutable, Vector3DImmutable
from ladybug_geometry.geometry3d.plane import Plane
from ladybug_geometry.geometry3d.line import LineSegment3D, LineSegment3DImmutable
from ladybug_geometry.geometry3d.face import Face3D


import unittest
import pytest
import math


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

        assert face.area == 4
        assert face.perimeter == 8
        assert face.is_clockwise is True
        assert face.is_convex is True
        assert face.is_self_intersecting is False


if __name__ == "__main__":
    unittest.main()
