# coding=utf-8
import pytest

from ladybug_geometry.geometry2d.pointvector import Vector2D
from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
from ladybug_geometry.geometry3d.plane import Plane
from ladybug_geometry.geometry3d.line import LineSegment3D
from ladybug_geometry.geometry3d.ray import Ray3D
from ladybug_geometry.geometry3d.face import Face3D

import math
import json


def test_face3d_init():
    """Test the initialization of Face3D objects and basic properties."""
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
        assert isinstance(point, Point3D)

    assert isinstance(face.boundary_segments, tuple)
    assert len(face.boundary_segments) == 4
    for seg in face.boundary_segments:
        assert isinstance(seg, LineSegment3D)
        assert seg.length == 2
    assert face.has_holes is False
    assert face.hole_segments is None

    assert face.area == 4
    assert face.perimeter == 8
    assert face.is_clockwise is False
    assert face.is_convex is True
    assert face.is_self_intersecting is False
    assert face.vertices[0] == face[0]


def test_equality():
    """Test the equality of Face3D objects."""
    pts = (Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2))
    pts_2 = (Point3D(0.1, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2))
    plane = Plane(Vector3D(0, 0, 1), Point3D(0, 0, 2))
    face = Face3D(pts, plane)
    face_dup = face.duplicate()
    face_alt = Face3D(pts_2, plane)

    assert face is face
    assert face is not face_dup
    assert face == face_dup
    assert hash(face) == hash(face_dup)
    assert face != face_alt
    assert hash(face) != hash(face_alt)


def test_face3d_to_from_dict():
    """Test the to/from dict of Face3D objects."""
    pts = (Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2))
    plane = Plane(Vector3D(0, 0, 1), Point3D(0, 0, 2))
    face = Face3D(pts, plane)
    face_dict = face.to_dict()
    new_face = Face3D.from_dict(face_dict)
    assert isinstance(new_face, Face3D)
    assert new_face.to_dict() == face_dict

    bound_pts = [Point3D(0, 0), Point3D(4, 0), Point3D(4, 4), Point3D(0, 4)]
    hole_pts = [Point3D(1, 1), Point3D(3, 1), Point3D(3, 3), Point3D(1, 3)]
    face = Face3D(bound_pts, None, [hole_pts])
    face_dict = face.to_dict()
    new_face = Face3D.from_dict(face_dict)
    assert isinstance(new_face, Face3D)
    assert new_face.to_dict() == face_dict


def test_face3d_init_from_vertices():
    """Test the initialization of Face3D objects without a plane."""
    pts = (Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2))
    face = Face3D(pts)

    assert isinstance(face.plane, Plane)
    assert face.plane.n == Vector3D(0, 0, -1)
    assert face.plane.n == face.normal
    assert face.plane.o == Point3D(0, 0, 2)

    assert isinstance(face.vertices, tuple)
    assert len(face.vertices) == 4
    assert len(face) == 4
    for point in face:
        assert isinstance(point, Point3D)

    assert isinstance(face.boundary_segments, tuple)
    assert len(face.boundary_segments) == 4
    for seg in face.boundary_segments:
        assert isinstance(seg, LineSegment3D)
        assert seg.length == 2
    assert face.has_holes is False
    assert face.hole_segments is None

    assert face.area == 4
    assert face.perimeter == 8
    assert face.is_clockwise is False
    assert face.is_convex is True
    assert face.is_self_intersecting is False
    assert face.vertices[0] == face[0]


def test_face3d_init_from_vertices_colinear():
    """Test the initialization of Face3D objects with colinear vertices."""
    pts = (Point3D(0, 0, 2), Point3D(0, 1, 2), Point3D(0, 2, 2), Point3D(2, 2, 2),
           Point3D(2, 0, 2))
    face = Face3D(pts)

    assert not face.normal.is_zero(0.000001)
    assert face.plane.n == Vector3D(0, 0, -1)
    assert face.plane.n == face.normal
    assert face.plane.o == Point3D(0, 0, 2)


def test_face3d_init_from_extrusion():
    """Test the initialization of Face3D from_extrusion."""
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
        assert isinstance(point, Point3D)

    assert isinstance(face.boundary_segments, tuple)
    assert len(face.boundary_segments) == 4
    for seg in face.boundary_segments:
        assert isinstance(seg, LineSegment3D)
        assert seg.length == 2
    assert face.has_holes is False
    assert face.hole_segments is None

    assert face.area == 4
    assert face.perimeter == 8
    assert face.is_clockwise is False
    assert face.is_convex is True
    assert face.is_self_intersecting is False


def test_face3d_init_from_rectangle():
    """Test the initialization of Face3D from_rectangle."""
    plane = Plane(Vector3D(0, 0, 1), Point3D(2, 2, 2))
    face = Face3D.from_rectangle(2, 2, plane)

    assert isinstance(face.plane, Plane)
    assert face.plane.n == Vector3D(0, 0, 1)
    assert face.plane.n == face.normal
    assert face.plane.o == Point3D(2, 2, 2)

    assert isinstance(face.vertices, tuple)
    assert len(face.vertices) == 4
    for point in face.vertices:
        assert isinstance(point, Point3D)

    assert isinstance(face.boundary_segments, tuple)
    assert len(face.boundary_segments) == 4
    for seg in face.boundary_segments:
        assert isinstance(seg, LineSegment3D)
        assert seg.length == 2
    assert face.has_holes is False
    assert face.hole_segments is None

    assert face.area == 4
    assert face.perimeter == 8
    assert face.is_clockwise is False
    assert face.is_convex is True
    assert face.is_self_intersecting is False


def test_face3d_init_from_regular_polygon():
    """Test the initialization of Face3D from_regular_polygon."""
    plane = Plane(Vector3D(0, 0, 1), Point3D(2, 2, 2))
    face = Face3D.from_regular_polygon(8, 2, plane)

    assert isinstance(face.plane, Plane)
    assert face.plane.n == Vector3D(0, 0, 1)
    assert face.plane.n == face.normal
    assert face.plane.o == Point3D(2, 2, 2)

    assert isinstance(face.vertices, tuple)
    assert len(face.vertices) == 8
    for point in face.vertices:
        assert isinstance(point, Point3D)
    assert isinstance(face.boundary_segments, tuple)
    assert len(face.boundary_segments) == 8
    for seg in face.boundary_segments:
        assert isinstance(seg, LineSegment3D)
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


def test_face3d_init_from_shape_with_hole():
    """Test the initialization of Face3D from_shape_with_holes with one hole."""
    bound_pts = [Point3D(0, 0), Point3D(4, 0), Point3D(4, 4), Point3D(0, 4)]
    hole_pts = [Point3D(1, 1), Point3D(3, 1), Point3D(3, 3), Point3D(1, 3)]
    face = Face3D(bound_pts, None, [hole_pts])

    assert face.plane.n == Vector3D(0, 0, 1)
    assert face.plane.n == face.normal
    assert face.plane.o == Point3D(0, 0, 0)

    assert isinstance(face.vertices, tuple)
    assert len(face.vertices) == 10
    for point in face.vertices:
        assert isinstance(point, Point3D)

    assert isinstance(face.boundary_segments, tuple)
    assert len(face.boundary_segments) == 4
    for seg in face.boundary_segments:
        assert isinstance(seg, LineSegment3D)
    assert face.has_holes is True
    assert isinstance(face.hole_segments, tuple)
    assert len(face.hole_segments) == 1
    assert len(face.hole_segments[0]) == 4
    for seg in face.hole_segments[0]:
        assert isinstance(seg, LineSegment3D)

    assert face.area == 12
    assert face.perimeter == pytest.approx(24, rel=1e-3)
    assert face.is_clockwise is False
    assert face.is_convex is False
    assert face.is_self_intersecting is False


def test_face3d_init_from_shape_with_holes():
    """Test the initialization of Face3D from_shape_with_holes."""
    bound_pts = [Point3D(0, 0), Point3D(4, 0), Point3D(4, 4), Point3D(0, 4)]
    hole_pts_1 = [Point3D(1, 1), Point3D(1.5, 1), Point3D(1.5, 1.5), Point3D(1, 1.5)]
    hole_pts_2 = [Point3D(2, 2), Point3D(3, 2), Point3D(3, 3), Point3D(2, 3)]
    face = Face3D(bound_pts, None, [hole_pts_1, hole_pts_2])

    assert face.plane.n == Vector3D(0, 0, 1)
    assert face.plane.n == face.normal
    assert face.plane.o == Point3D(0, 0, 0)

    assert isinstance(face.vertices, tuple)
    assert len(face.vertices) == 16
    for point in face.vertices:
        assert isinstance(point, Point3D)

    assert isinstance(face.boundary_segments, tuple)
    assert len(face.boundary_segments) == 4
    for seg in face.boundary_segments:
        assert isinstance(seg, LineSegment3D)
    assert face.has_holes is True
    assert isinstance(face.hole_segments, tuple)
    assert len(face.hole_segments) == 2

    assert face.area == 16 - 1.25
    assert face.perimeter == pytest.approx(22, rel=1e-3)
    assert face.is_clockwise is False
    assert face.is_convex is False
    assert face.is_self_intersecting is False


def test_face3d_init_from_punched_geometry():
    """Test the initialization of Face3D from_shape_with_holes."""
    bound_pts = [Point3D(0, 0), Point3D(4, 0), Point3D(4, 4), Point3D(0, 4)]
    hole_pts_1 = [Point3D(1, 1), Point3D(1.5, 1), Point3D(1.5, 1.5), Point3D(1, 1.5)]
    hole_pts_2 = [Point3D(2, 2), Point3D(3, 2), Point3D(3, 3), Point3D(2, 3)]
    face_1 = Face3D(bound_pts)
    face_2 = Face3D(bound_pts, None, [hole_pts_1])
    sub_face = Face3D(hole_pts_2)

    face = Face3D.from_punched_geometry(face_1, [sub_face])
    assert isinstance(face.vertices, tuple)
    assert len(face.vertices) == 10
    assert face.has_holes is True
    assert isinstance(face.hole_segments, tuple)
    assert len(face.hole_segments) == 1
    assert face.area == 15
    assert face.perimeter == pytest.approx(20, rel=1e-3)
    assert face.is_clockwise is False
    assert face.is_convex is False
    assert face.is_self_intersecting is False

    face = Face3D.from_punched_geometry(face_2, [sub_face])
    assert isinstance(face.vertices, tuple)
    assert len(face.vertices) == 16
    assert face.has_holes is True
    assert isinstance(face.hole_segments, tuple)
    assert len(face.hole_segments) == 2
    assert face.area == 16 - 1.25
    assert face.perimeter == pytest.approx(22, rel=1e-3)
    assert face.is_clockwise is False
    assert face.is_convex is False
    assert face.is_self_intersecting is False


def test_face3d_init_h_shape():
    """Test the initialization of Face3D that is H-shaped."""
    geo_file = './tests/json/h_shaped_face.json'
    with open(geo_file, 'r') as fp:
        geo_dict = json.load(fp)
    floor_geo = Face3D.from_dict(geo_dict)

    assert floor_geo.normal.z == pytest.approx(-1, rel=1e-3)


def test_is_geometrically_equivalent():
    """Test the is_geometrically_equivalent method."""
    plane_1 = Plane(Vector3D(0, 0, 1))
    plane_2 = Plane(Vector3D(0, 0, -1))
    pts_1 = (Point3D(0, 0), Point3D(2, 0), Point3D(2, 2), Point3D(0, 2))
    pts_2 = (Point3D(0, 0), Point3D(0, 2), Point3D(2, 2), Point3D(2, 0))
    pts_3 = (Point3D(0, 0), Point3D(2, 0), Point3D(2.1, 2.1), Point3D(0, 2))
    pts_4 = (Point3D(1, 0), Point3D(0, 1), Point3D(1, 2), Point3D(2, 1))
    pts_5 = (Point3D(0, 0), Point3D(2, 2), Point3D(2, 0), Point3D(0, 2))
    pts_6 = (Point3D(2, 0), Point3D(2, 2), Point3D(0, 2), Point3D(0, 0))
    face_1 = Face3D(pts_1, plane_1)
    face_2 = Face3D(pts_2, plane_1)
    face_3 = Face3D(pts_1, plane_2)
    face_4 = Face3D(pts_2, plane_2)
    face_5 = Face3D(pts_3, plane_1)
    face_6 = Face3D(pts_4, plane_1)
    face_7 = Face3D(pts_5, plane_1)
    face_8 = Face3D(pts_6, plane_1)

    assert face_1.is_geometrically_equivalent(face_2, 0.0001) is True
    assert face_1.is_geometrically_equivalent(face_3, 0.0001) is True
    assert face_1.is_geometrically_equivalent(face_4, 0.0001) is True
    assert face_1.is_geometrically_equivalent(face_5, 0.0001) is False
    assert face_1.is_geometrically_equivalent(face_6, 0.0001) is False
    assert face_1.is_geometrically_equivalent(face_7, 0.0001) is False
    assert face_1.is_geometrically_equivalent(face_8, 0.0001) is True


def test_is_centered_adjacent():
    """Test the is_centered_adjacent method."""
    plane_1 = Plane(Vector3D(0, 0, 1))
    plane_2 = Plane(Vector3D(0, 0, -1))
    pts_1 = (Point3D(0, 0), Point3D(2, 0), Point3D(2, 2), Point3D(0, 2))
    pts_2 = (Point3D(0, 0), Point3D(0, 2), Point3D(2, 2), Point3D(2, 0))
    pts_3 = (Point3D(0, 0), Point3D(2, 0), Point3D(2.1, 2.1), Point3D(0, 2))
    pts_4 = (Point3D(1, 0), Point3D(0, 1), Point3D(1, 2), Point3D(2, 1))
    pts_5 = (Point3D(2, 0), Point3D(2, 2), Point3D(0, 2), Point3D(0, 0))
    face_1 = Face3D(pts_1, plane_1)
    face_2 = Face3D(pts_2, plane_1)
    face_3 = Face3D(pts_1, plane_2)
    face_4 = Face3D(pts_2, plane_2)
    face_5 = Face3D(pts_3, plane_1)
    face_6 = Face3D(pts_4, plane_1)
    face_7 = Face3D(pts_5, plane_1)

    assert face_1.is_centered_adjacent(face_2, 0.0001) is True
    assert face_1.is_centered_adjacent(face_3, 0.0001) is True
    assert face_1.is_centered_adjacent(face_4, 0.0001) is True
    assert face_1.is_centered_adjacent(face_5, 0.0001) is False
    assert face_1.is_centered_adjacent(face_6, 0.0001) is False
    assert face_1.is_centered_adjacent(face_7, 0.0001) is True


def test_is_sub_face():
    """Test the is_sub_face method."""
    bound_pts = [Point3D(0, 0), Point3D(4, 0), Point3D(4, 4), Point3D(0, 4)]
    sub_pts_1 = [Point3D(1, 1), Point3D(1.5, 1), Point3D(1.5, 1.5), Point3D(1, 1.5)]
    sub_pts_2 = [Point3D(2, 2), Point3D(3, 2), Point3D(3, 3), Point3D(2, 3)]
    sub_pts_3 = [Point3D(2, 2), Point3D(6, 2), Point3D(6, 6), Point3D(2, 6)]
    sub_pts_4 = [Point3D(5, 5), Point3D(6, 5), Point3D(6, 6), Point3D(5, 6)]
    sub_pts_5 = [Point3D(), Point3D(2), Point3D(2, 0, 2), Point3D(0, 0, 2)]
    plane_1 = Plane(Vector3D(0, 0, 1))
    plane_2 = Plane(Vector3D(0, 0, -1))
    plane_3 = Plane(Vector3D(0, 1, 0))
    face = Face3D(bound_pts, plane_1)
    sub_face_1 = Face3D(sub_pts_1, plane_1)
    sub_face_2 = Face3D(sub_pts_2, plane_2)
    sub_face_3 = Face3D(sub_pts_3, plane_1)
    sub_face_4 = Face3D(sub_pts_4, plane_1)
    sub_face_5 = Face3D(sub_pts_5, plane_3)

    assert face.is_sub_face(sub_face_1, 0.0001, 0.0001) is True
    assert face.is_sub_face(sub_face_2, 0.0001, 0.0001) is True
    assert face.is_sub_face(sub_face_3, 0.0001, 0.0001) is False
    assert face.is_sub_face(sub_face_4, 0.0001, 0.0001) is False
    assert face.is_sub_face(sub_face_5, 0.0001, 0.0001) is False


def test_is_point_on_face():
    """Test the is_point_on_face method."""
    bound_pts = [Point3D(0, 0), Point3D(4, 0), Point3D(4, 4), Point3D(0, 4)]
    sub_pt_1 = Point3D(1, 1)
    sub_pt_2 = Point3D(3, 2)
    sub_pt_3 = Point3D(6, 2)
    sub_pt_4 = Point3D(6, 6)
    sub_pt_5 = Point3D(2, 0, 2)
    plane_1 = Plane(Vector3D(0, 0, 1))
    face = Face3D(bound_pts, plane_1)

    assert face.is_point_on_face(sub_pt_1, 0.0001) is True
    assert face.is_point_on_face(sub_pt_2, 0.0001) is True
    assert face.is_point_on_face(sub_pt_3, 0.0001) is False
    assert face.is_point_on_face(sub_pt_4, 0.0001) is False
    assert face.is_point_on_face(sub_pt_5, 0.0001) is False


def test_clockwise():
    """Test the clockwise property."""
    plane_1 = Plane(Vector3D(0, 0, 1))
    plane_2 = Plane(Vector3D(0, 0, -1))
    pts_1 = (Point3D(0, 0), Point3D(2, 0), Point3D(2, 2), Point3D(0, 2))
    pts_2 = (Point3D(0, 0), Point3D(0, 2), Point3D(2, 2), Point3D(2, 0))
    face_1 = Face3D(pts_1, plane_1, enforce_right_hand=True)
    face_2 = Face3D(pts_2, plane_1, enforce_right_hand=True)
    face_3 = Face3D(pts_1, plane_2, enforce_right_hand=True)
    face_4 = Face3D(pts_2, plane_2, enforce_right_hand=True)

    assert face_1.is_clockwise is face_1.polygon2d.is_clockwise is False
    assert face_2.is_clockwise is face_2.polygon2d.is_clockwise is False
    assert face_3.is_clockwise is face_3.polygon2d.is_clockwise is False
    assert face_4.is_clockwise is face_4.polygon2d.is_clockwise is False

    face_1 = Face3D(pts_1, plane_1, enforce_right_hand=False)
    face_2 = Face3D(pts_2, plane_1, enforce_right_hand=False)
    face_3 = Face3D(pts_1, plane_2, enforce_right_hand=False)
    face_4 = Face3D(pts_2, plane_2, enforce_right_hand=False)

    assert face_1.is_clockwise is face_1.polygon2d.is_clockwise is False
    assert face_2.is_clockwise is face_2.polygon2d.is_clockwise is True
    assert face_3.is_clockwise is face_3.polygon2d.is_clockwise is True
    assert face_4.is_clockwise is face_4.polygon2d.is_clockwise is False

    assert face_1.area == face_2.area == face_3.area == face_4.area == 4


def test_is_convex():
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


def test_is_self_intersecting():
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


def test_is_valid():
    """Test the is_valid property."""
    plane_1 = Plane(Vector3D(0, 0, 1))
    pts_1 = (Point3D(0, 0), Point3D(2, 0), Point3D(2, 2))
    pts_2 = (Point3D(0, 0), Point3D(2, 0), Point3D(2, 0))
    face_1 = Face3D(pts_1, plane_1)
    face_2 = Face3D(pts_2, plane_1)

    assert face_1.is_valid is True
    assert face_2.is_valid is False


def test_min_max_center():
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


def test_upper_left_counter_clockwise_vertices():
    """Test the upper_left_counter_clockwise_vertices property."""
    plane_1 = Plane(Vector3D(0, 1, 0))
    plane_2 = Plane(Vector3D(0, -1, 0))
    pts_1 = (Point3D(0, 0, 0), Point3D(2, 0, 0), Point3D(2, 0, 2), Point3D(0, 0, 2))
    pts_2 = (Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(2, 0, 0))
    face_1 = Face3D(pts_1, plane_1, enforce_right_hand=False)
    face_2 = Face3D(pts_2, plane_1, enforce_right_hand=False)
    face_3 = Face3D(pts_1, plane_2, enforce_right_hand=False)
    face_4 = Face3D(pts_2, plane_2, enforce_right_hand=False)

    up_cclock_1 = face_1.upper_left_counter_clockwise_vertices
    assert up_cclock_1[0] == Point3D(2, 0, 2)
    assert face_1.is_clockwise is True
    assert Face3D(up_cclock_1, enforce_right_hand=False).is_clockwise is False
    up_cclock_2 = face_2.upper_left_counter_clockwise_vertices
    assert up_cclock_2[0] == Point3D(2, 0, 2)
    assert face_2.is_clockwise is False
    assert not Face3D(up_cclock_2, enforce_right_hand=False).is_clockwise
    assert up_cclock_1 == up_cclock_2

    up_cclock_3 = face_3.upper_left_counter_clockwise_vertices
    assert up_cclock_3[0] == Point3D(0, 0, 2)
    assert face_3.is_clockwise is False
    assert Face3D(up_cclock_3, enforce_right_hand=False).is_clockwise is False
    up_cclock_4 = face_4.upper_left_counter_clockwise_vertices
    assert up_cclock_4[0] == Point3D(0, 0, 2)
    assert face_4.is_clockwise is True
    assert Face3D(up_cclock_4, enforce_right_hand=False).is_clockwise is False


def test_upper_left_counter_clockwise_vertices_triangle():
    """Test the upper_left_counter_clockwise_vertices property with triangles."""
    plane_1 = Plane(Vector3D(0, 1, 0))
    pts_1 = (Point3D(0, 0, 0), Point3D(2, 0, -1), Point3D(0, 0, 2))
    pts_2 = (Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(2, 0, -1))
    face_1 = Face3D(pts_1, plane_1, enforce_right_hand=False)
    face_2 = Face3D(pts_2, plane_1, enforce_right_hand=False)

    up_cclock_1 = face_1.upper_left_counter_clockwise_vertices
    assert up_cclock_1[0] == Point3D(0, 0, 2)
    assert face_1.is_clockwise is True
    assert Face3D(up_cclock_1, enforce_right_hand=False).is_clockwise is False
    up_cclock_2 = face_2.upper_left_counter_clockwise_vertices
    assert up_cclock_2[0] == Point3D(0, 0, 2)
    assert face_2.is_clockwise is False
    assert not Face3D(up_cclock_2, enforce_right_hand=False).is_clockwise
    assert up_cclock_1 == up_cclock_2


def test_duplicate():
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


def test_remove_colinear_vertices():
    """Test the remove_colinear_vertices method of Face3D."""
    pts_1 = (Point3D(0, 0), Point3D(2, 0), Point3D(2, 2), Point3D(0, 2))
    pts_2 = (Point3D(0, 0), Point3D(1, 0), Point3D(2, 0), Point3D(2, 2),
             Point3D(0, 2))
    plane_1 = Plane(Vector3D(0, 0, 1))
    face_1 = Face3D(pts_1, plane_1)
    face_2 = Face3D(pts_2, plane_1)

    assert len(face_1.remove_colinear_vertices(0.0001).vertices) == 4
    assert len(face_2.remove_colinear_vertices(0.0001).vertices) == 4


def test_remove_colinear_vertices_custom():
    """Test the remove_colinear_vertices method with some custom geometry."""
    geo_file = './tests/json/dup_vert_face.json'
    with open(geo_file, 'r') as fp:
        geo_dict = json.load(fp)
    face_geo = Face3D.from_dict(geo_dict)

    assert -1e-3 < face_geo.normal.z < 1e-3
    assert len(face_geo.vertices) == 17

    assert len(face_geo.remove_colinear_vertices(0.0001).vertices) == 16


def test_triangulated_mesh_and_centroid():
    """Test the triangulation properties of Face3D."""
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


def test_triangulated_mesh_with_holes():
    """Test the triangulation properties of a Face3D with holes."""
    geo_file = './tests/json/face_with_holes.json'
    with open(geo_file, 'r') as fp:
        geo_dict = json.load(fp)
    face_geo = Face3D.from_dict(geo_dict)

    assert len(face_geo.triangulated_mesh3d.vertices) == 12
    assert len(face_geo.triangulated_mesh3d.faces) == 14
    assert face_geo.triangulated_mesh3d.area == pytest.approx(face_geo.area, rel=1e-3)


def test_triangulated_mesh_with_many_holes():
    """Test the triangulation properties of a Face3D with many holes."""
    geo_file = './tests/json/faces_with_many_holes.json'
    with open(geo_file, 'r') as fp:
        geo_dict = json.load(fp)
    for geo in geo_dict:
        face_geo = Face3D.from_dict(geo)
        assert face_geo.triangulated_mesh3d.area == \
            pytest.approx(face_geo.area, rel=1e-3)


def test_check_planar():
    """Test the check_planar method of Face3D."""
    pts_1 = (Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(2, 2, 2), Point3D(0, 2, 2))
    pts_2 = (Point3D(0, 0, 0), Point3D(2, 0, 2), Point3D(2, 2, 2), Point3D(0, 2, 2))
    pts_3 = (Point3D(0, 0, 2.0001), Point3D(2, 0, 2), Point3D(2, 2, 2), Point3D(0, 2, 2))
    plane_1 = Plane(Vector3D(0, 0, 1), Point3D(0, 0, 2))
    face_1 = Face3D(pts_1, plane_1)
    face_2 = Face3D(pts_2, plane_1)
    face_3 = Face3D(pts_3, plane_1)

    assert face_1.check_planar(0.001) is True
    assert face_2.check_planar(0.001, False) is False
    with pytest.raises(Exception):
        face_2.check_planar(0.0001)
    assert face_3.check_planar(0.001) is True
    assert face_3.check_planar(0.000001, False) is False
    with pytest.raises(Exception):
        face_3.check_planar(0.000001)


def test_flip():
    """Test the flip method of Face3D."""
    pts_1 = (Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(2, 2, 2), Point3D(0, 2, 2))
    plane_1 = Plane(Vector3D(0, 0, 1), Point3D(0, 0, 2))
    face_1 = Face3D(pts_1, plane_1)
    face_2 = face_1.flip()

    assert face_1.normal == face_2.normal.reverse()
    assert face_1.is_clockwise is False
    assert face_2.is_clockwise is False
    for i, pt in enumerate(reversed(face_1.vertices)):
        assert pt == face_2[i]


def test_move():
    """Test the Face3D move method."""
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
    assert new_face.normal == face_1.normal


def test_scale():
    """Test the Face3D scale method."""
    pts_1 = (Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(2, 2, 2), Point3D(0, 2, 2))
    plane_1 = Plane(Vector3D(0, 0, 1), Point3D(0, 0, 2))
    face_1 = Face3D(pts_1, plane_1)
    pts_2 = (Point3D(1, 1), Point3D(2, 1), Point3D(2, 2), Point3D(1, 2))
    plane_2 = Plane(Vector3D(0, 0, 1))
    face_2 = Face3D(pts_2, plane_2)
    origin_1 = Point3D(2, 0)
    origin_2 = Point3D(1, 1)

    new_face_1 = face_1.scale(2, origin_1)
    assert new_face_1[0] == Point3D(-2, 0, 4)
    assert new_face_1[1] == Point3D(2, 0, 4)
    assert new_face_1[2] == Point3D(2, 4, 4)
    assert new_face_1[3] == Point3D(-2, 4, 4)
    assert new_face_1.area == face_1.area * 2 ** 2
    assert new_face_1.perimeter == face_1.perimeter * 2
    assert new_face_1.is_clockwise is face_1.is_clockwise
    assert new_face_1.is_convex is face_1.is_convex
    assert new_face_1.is_self_intersecting is face_1.is_self_intersecting
    assert new_face_1.normal == face_1.normal

    new_face_2 = face_2.scale(2, origin_2)
    assert new_face_2[0] == Point3D(1, 1)
    assert new_face_2[1] == Point3D(3, 1)
    assert new_face_2[2] == Point3D(3, 3)
    assert new_face_2[3] == Point3D(1, 3)
    assert new_face_2.area == face_2.area * 2 ** 2
    assert new_face_2.perimeter == face_2.perimeter * 2
    assert new_face_2.is_clockwise is face_2.is_clockwise
    assert new_face_2.is_convex is face_2.is_convex
    assert new_face_2.is_self_intersecting is face_2.is_self_intersecting
    assert new_face_2.normal == face_2.normal


def test_scale_world_origin():
    """Test the Face3D scale method with None origin."""
    pts = (Point3D(1, 1, 2), Point3D(2, 1, 2), Point3D(2, 2, 2), Point3D(1, 2, 2))
    plane = Plane(Vector3D(0, 0, 1), Point3D(0, 0, 2))
    face = Face3D(pts, plane)

    new_face = face.scale(2)
    assert new_face[0] == Point3D(2, 2, 4)
    assert new_face[1] == Point3D(4, 2, 4)
    assert new_face[2] == Point3D(4, 4, 4)
    assert new_face[3] == Point3D(2, 4, 4)
    assert new_face.area == face.area * 2 ** 2
    assert new_face.perimeter == face.perimeter * 2
    assert new_face.is_clockwise is face.is_clockwise is False
    assert new_face.is_convex is face.is_convex
    assert new_face.is_self_intersecting is face.is_self_intersecting
    assert new_face.normal == face.normal

    face = Face3D(pts, plane, enforce_right_hand=False)
    new_face = face.scale(2)
    assert new_face.is_clockwise is False
    new_face = face.scale(-2)
    assert new_face.is_clockwise is False


def test_rotate():
    """Test the Face3D rotate method."""
    pts = (Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(2, 2, 2), Point3D(0, 2, 2))
    plane = Plane(Vector3D(0, 0, 1), Point3D(0, 0, 2))
    face = Face3D(pts, plane)
    origin = Point3D(0, 0, 0)
    axis = Vector3D(1, 0, 0)

    test_1 = face.rotate(axis, math.pi, origin)
    assert test_1[0].x == pytest.approx(0, rel=1e-3)
    assert test_1[0].y == pytest.approx(0, rel=1e-3)
    assert test_1[0].z == pytest.approx(-2, rel=1e-3)
    assert test_1[2].x == pytest.approx(2, rel=1e-3)
    assert test_1[2].y == pytest.approx(-2, rel=1e-3)
    assert test_1[2].z == pytest.approx(-2, rel=1e-3)
    assert face.area == test_1.area
    assert len(face.vertices) == len(test_1.vertices)

    test_2 = face.rotate(axis, math.pi / 2, origin)
    assert test_2[0].x == pytest.approx(0, rel=1e-3)
    assert test_2[0].y == pytest.approx(-2, rel=1e-3)
    assert test_2[0].z == pytest.approx(0, rel=1e-3)
    assert test_2[2].x == pytest.approx(2, rel=1e-3)
    assert test_2[2].y == pytest.approx(-2, rel=1e-3)
    assert test_2[2].z == pytest.approx(2, rel=1e-3)
    assert face.area == test_2.area
    assert len(face.vertices) == len(test_2.vertices)

    face = Face3D(pts, plane, enforce_right_hand=False)
    test_1 = face.rotate(axis, math.pi, origin)
    assert test_1.is_clockwise is False
    test_2 = face.rotate(axis, math.pi/2, origin)
    assert test_2.is_clockwise is False


def test_rotate_xy():
    """Test the Face3D rotate_xy method."""
    pts = (Point3D(1, 1, 2), Point3D(2, 1, 2), Point3D(2, 2, 2), Point3D(1, 2, 2))
    plane = Plane(Vector3D(0, 0, 1), Point3D(0, 0, 2))
    face = Face3D(pts, plane)
    origin_1 = Point3D(1, 1, 0)

    test_1 = face.rotate_xy(math.pi, origin_1)
    assert test_1[0].x == pytest.approx(1, rel=1e-3)
    assert test_1[0].y == pytest.approx(1, rel=1e-3)
    assert test_1[0].z == pytest.approx(2, rel=1e-3)
    assert test_1[2].x == pytest.approx(0, rel=1e-3)
    assert test_1[2].y == pytest.approx(0, rel=1e-3)
    assert test_1[2].z == pytest.approx(2, rel=1e-3)

    test_2 = face.rotate_xy(math.pi/2, origin_1)
    assert test_2[0].x == pytest.approx(1, rel=1e-3)
    assert test_2[0].y == pytest.approx(1, rel=1e-3)
    assert test_1[0].z == pytest.approx(2, rel=1e-3)
    assert test_2[2].x == pytest.approx(0, rel=1e-3)
    assert test_2[2].y == pytest.approx(2, rel=1e-3)
    assert test_1[2].z == pytest.approx(2, rel=1e-3)

    face = Face3D(pts, plane, enforce_right_hand=False)
    test_1 = face.rotate_xy(math.pi, origin_1)
    assert test_1.is_clockwise is False
    test_2 = face.rotate_xy(math.pi/2, origin_1)
    assert test_2.is_clockwise is False


def test_reflect():
    """Test the Face3D reflect method."""
    pts = (Point3D(1, 1, 2), Point3D(2, 1, 2), Point3D(2, 2, 2), Point3D(1, 2, 2))
    plane = Plane(Vector3D(0, 0, 1), Point3D(0, 0, 2))
    face = Face3D(pts, plane)

    origin_1 = Point3D(1, 0, 2)
    normal_1 = Vector3D(1, 0, 0)
    normal_2 = Vector3D(-1, -1, 0).normalize()

    test_1 = face.reflect(normal_1, origin_1)
    assert test_1[-1].x == pytest.approx(1, rel=1e-3)
    assert test_1[-1].y == pytest.approx(1, rel=1e-3)
    assert test_1[-1].z == pytest.approx(2, rel=1e-3)
    assert test_1[1].x == pytest.approx(0, rel=1e-3)
    assert test_1[1].y == pytest.approx(2, rel=1e-3)
    assert test_1[1].z == pytest.approx(2, rel=1e-3)

    test_1 = face.reflect(normal_2, Point3D(0, 0, 2))
    assert test_1.is_clockwise is False
    assert test_1[-1].x == pytest.approx(-1, rel=1e-3)
    assert test_1[-1].y == pytest.approx(-1, rel=1e-3)
    assert test_1[-1].z == pytest.approx(2, rel=1e-3)
    assert test_1[1].x == pytest.approx(-2, rel=1e-3)
    assert test_1[1].y == pytest.approx(-2, rel=1e-3)
    assert test_1[1].z == pytest.approx(2, rel=1e-3)

    test_2 = face.reflect(normal_2, origin_1)
    assert test_2.is_clockwise is False
    assert test_2[-1].x == pytest.approx(0, rel=1e-3)
    assert test_2[-1].y == pytest.approx(0, rel=1e-3)
    assert test_2[-1].z == pytest.approx(2, rel=1e-3)
    assert test_2[1].x == pytest.approx(-1, rel=1e-3)
    assert test_2[1].y == pytest.approx(-1, rel=1e-3)
    assert test_2[1].z == pytest.approx(2, rel=1e-3)

    face = Face3D(pts, plane, enforce_right_hand=False)
    test_1 = face.reflect(normal_1, origin_1)
    assert test_1.is_clockwise is False
    test_1 = face.reflect(normal_2, Point3D(0, 0, 2))
    assert test_1.is_clockwise is False
    test_2 = face.reflect(normal_2, origin_1)
    assert test_2.is_clockwise is False


def test_intersect_line_ray():
    """Test the Face3D intersect_line_ray method."""
    pts = (Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(2, 1, 2), Point3D(1, 1, 2),
           Point3D(1, 2, 2), Point3D(0, 2, 2))
    plane = Plane(Vector3D(0, 0, 1), Point3D(0, 0, 2))
    face = Face3D(pts, plane)
    ray_1 = Ray3D(Point3D(0.5, 0.5, 0), Vector3D(0, 0, 1))
    ray_2 = Ray3D(Point3D(0.5, 0.5, 0), Vector3D(0, 0, -1))
    ray_3 = Ray3D(Point3D(1.5, 1.5, 0), Vector3D(0, 0, 1))
    ray_4 = Ray3D(Point3D(-1, -1, 0), Vector3D(0, 0, 1))
    line_1 = LineSegment3D(Point3D(0.5, 0.5, 0), Vector3D(0, 0, 3))
    line_2 = LineSegment3D(Point3D(0.5, 0.5, 0), Vector3D(0, 0, 1))

    assert face.intersect_line_ray(ray_1) == Point3D(0.5, 0.5, 2)
    assert face.intersect_line_ray(ray_2) is None
    assert face.intersect_line_ray(ray_3) is None
    assert face.intersect_line_ray(ray_4) is None
    assert face.intersect_line_ray(line_1) == Point3D(0.5, 0.5, 2)
    assert face.intersect_line_ray(line_2) is None


def test_intersect_plane():
    """Test the Face3D intersect_plane method."""
    pts = (Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(2, 1, 2), Point3D(1, 1, 2),
           Point3D(1, 2, 2), Point3D(0, 2, 2))
    plane = Plane(Vector3D(0, 0, 1), Point3D(0, 0, 2))
    face = Face3D(pts, plane)

    plane_1 = Plane(Vector3D(0, 1, 0), Point3D(0.5, 0.5, 0))
    plane_2 = Plane(Vector3D(1, 0, 0), Point3D(0.5, 0.5, 0))
    plane_3 = Plane(Vector3D(0, 1, 0), Point3D(0.5, 1.5, 0))
    plane_4 = Plane(Vector3D(0, 1, 0), Point3D(0, 3, 0))
    plane_5 = Plane(Vector3D(1, 1, 0), Point3D(0, 2.5, 0))

    assert len(face.intersect_plane(plane_1)) == 1
    assert face.intersect_plane(plane_1)[0].p1 == Point3D(2, 0.5, 2)
    assert face.intersect_plane(plane_1)[0].p2 == Point3D(0, 0.5, 2)
    assert len(face.intersect_plane(plane_2)) == 1
    assert face.intersect_plane(plane_2)[0].p1 == Point3D(0.5, 0, 2)
    assert face.intersect_plane(plane_2)[0].p2 == Point3D(0.5, 2, 2)
    assert len(face.intersect_plane(plane_3)) == 1
    assert face.intersect_plane(plane_3)[0].p1 == Point3D(1, 1.5, 2)
    assert face.intersect_plane(plane_3)[0].p2 == Point3D(0, 1.5, 2)
    assert face.intersect_plane(plane_4) is None
    assert len(face.intersect_plane(plane_5)) == 2


def test_project_point():
    """Test the Face3D project_point method."""
    pts = (Point3D(0, 0), Point3D(2, 0), Point3D(2, 1), Point3D(1, 1),
           Point3D(1, 2), Point3D(0, 2))
    plane = Plane(Vector3D(0, 0, 1))
    face = Face3D(pts, plane)

    pt_1 = Point3D(0.5, 0.5, 2)
    pt_2 = Point3D(0.5, 0.5, -2)
    pt_3 = Point3D(1.5, 1.5, 2)
    pt_4 = Point3D(-1, -1, 2)

    assert face.project_point(pt_1) == Point3D(0.5, 0.5, 0)
    assert face.project_point(pt_2) == Point3D(0.5, 0.5, 0)
    assert face.project_point(pt_3) is None
    assert face.project_point(pt_4) is None


def test_mesh_grid():
    """Test the Face3D mesh_grid method."""
    pts = (Point3D(0, 0), Point3D(0, 2), Point3D(2, 2), Point3D(2, 0))
    plane = Plane(Vector3D(0, 0, 1))
    face = Face3D(pts, plane)
    mesh = face.mesh_grid(0.5)

    assert len(mesh.vertices) == 25
    assert len(mesh.faces) == 16
    assert mesh.area == 4

    assert mesh.min.x == 0
    assert mesh.min.y == 0
    assert mesh.min.z == 0
    assert mesh.max.x == 2
    assert mesh.max.y == 2
    assert mesh.max.z == 0
    assert mesh.center.x == 1
    assert mesh.center.y == 1
    assert len(mesh.face_areas) == 16
    assert mesh.face_areas[0] == 0.25
    assert len(mesh.face_centroids) == 16

    mesh_2 = face.mesh_grid(0.5, 0.5, 1, False)
    mesh_3 = face.mesh_grid(0.5, 0.5, 1, True)

    assert mesh_2.min.z == pytest.approx(1, rel=1e-2)
    assert mesh_2.max.z == pytest.approx(1, rel=1e-2)
    assert mesh_3.min.z == pytest.approx(-1, rel=1e-2)
    assert mesh_3.max.z == pytest.approx(-1, rel=1e-2)


def test_countour_by_number():
    """Test the countour_by_number method."""
    pts_1 = [Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(2, 0, 0)]
    pts_2 = [Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(4, 0, 0)]
    plane = Plane(Vector3D(0, 1, 0))
    face_1 = Face3D(pts_1, plane)
    face_2 = Face3D(pts_2, plane)

    contours = face_1.countour_by_number(4, Vector2D(0, 1), False, 0.01)
    assert len(contours) == 4
    assert contours[0].p2.z == pytest.approx(2, rel=1e-3)
    assert contours[-1].p2.z == pytest.approx(0.5, rel=1e-3)

    contours = face_1.countour_by_number(4, Vector2D(1), False, 0.01)
    assert len(contours) == 4
    assert contours[-1].p2.x == pytest.approx(1.5, rel=1e-3)

    contours = face_1.countour_by_number(4, Vector2D(1), True, 0.01)
    assert len(contours) == 4
    assert contours[-1].p2.x == pytest.approx(0.5, rel=1e-3)

    contours = face_2.countour_by_number(4, Vector2D(0, 1), False, 0.01)
    assert len(contours) == 4
    contours = face_2.countour_by_number(8, Vector2D(1), False, 0.01)
    assert len(contours) == 8


def test_countour_by_distance_between():
    """Test the countour_by_distance_between method."""
    pts_1 = [Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(2, 0, 0)]
    pts_2 = [Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(4, 0, 0)]
    plane = Plane(Vector3D(0, 1, 0))
    face_1 = Face3D(pts_1, plane)
    face_2 = Face3D(pts_2, plane)

    contours = face_1.countour_by_distance_between(0.5, Vector2D(0, 1), False, 0.01)
    assert len(contours) == 4
    assert contours[0].p2.z == pytest.approx(2, rel=1e-3)
    assert contours[-1].p2.z == pytest.approx(0.5, rel=1e-3)

    contours = face_1.countour_by_distance_between(0.5, Vector2D(1), False, 0.01)
    assert len(contours) == 4
    assert contours[-1].p2.x == pytest.approx(1.5, rel=1e-3)

    contours = face_1.countour_by_distance_between(0.5, Vector2D(1), True, 0.01)
    assert len(contours) == 4
    assert contours[-1].p2.x == pytest.approx(0.5, rel=1e-3)

    contours = face_2.countour_by_distance_between(0.5, Vector2D(0, 1), False, 0.01)
    assert len(contours) == 4
    contours = face_2.countour_by_distance_between(0.5, Vector2D(1), False, 0.01)
    assert len(contours) == 8


def test_countour_fins_by_number():
    """Test the countour_fins_by_number method."""
    pts_1 = [Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(2, 0, 0)]
    plane = Plane(Vector3D(0, 1, 0))
    face_1 = Face3D(pts_1, plane)

    fins = face_1.countour_fins_by_number(4, 0.5, 0.5, 0, Vector2D(0, 1), False, 0.01)
    assert len(fins) == 4

    fins = face_1.countour_fins_by_number(4, 0.5, 0.5, 0, Vector2D(1), False, 0.01)
    assert len(fins) == 4

    fins = face_1.countour_fins_by_number(
        4, 0.5, 0.5, math.pi/4, Vector2D(0, 1), False, 0.01)
    assert len(fins) == 4


def test_countour_fins_by_distance_between():
    """Test the countour_fins_by_distance_between method."""
    pts_1 = [Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(2, 0, 0)]
    pts_2 = [Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(4, 0, 0)]
    plane = Plane(Vector3D(0, 1, 0))
    face_1 = Face3D(pts_1, plane)
    face_2 = Face3D(pts_2, plane)

    fins = face_1.countour_fins_by_distance_between(
        0.5, 0.5, 0.5, 0, Vector2D(0, 1), False, 0.01)
    assert len(fins) == 4

    fins = face_1.countour_fins_by_distance_between(
        0.25, 0.5, 0.5,0, Vector2D(1), False, 0.01)
    assert len(fins) == 8

    fins = face_2.countour_fins_by_distance_between(
        0.5, 0.5, 0.5, 0, Vector2D(1), False, 0.01)
    assert len(fins) == 8


def test_extract_rectangle():
    """Test the Face3D extract_rectangle method."""
    pts_1 = [Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(2, 0, 0)]
    pts_2 = [pt for pt in reversed(pts_1)]
    pts_3 = [Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(4, 0, 0)]
    pts_4 = [Point3D(-2, 0, 0), Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(4, 0, 0)]
    pts_5 = [Point3D(0, 0, 0), Point3D(-2, 0, 2), Point3D(4, 0, 2), Point3D(2, 0, 0)]
    pts_6 = [Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(1, 0, 3), Point3D(2, 0, 2),
             Point3D(2, 0, 0)]
    pts_7 = [Point3D(-2, 0, 0), Point3D(0, 0, 2), Point3D(2, 0, 2),
             Point3D(-1, 0, 0)]
    plane = Plane(Vector3D(0, 1, 0))
    face_1 = Face3D(pts_1, plane)
    face_2 = Face3D(pts_2, plane)
    face_3 = Face3D(pts_3, plane)
    face_4 = Face3D(pts_4, plane)
    face_5 = Face3D(pts_5, plane)
    face_6 = Face3D(pts_6, plane)
    face_7 = Face3D(pts_7, plane)

    f1_result = face_1.extract_rectangle(0.0001)
    assert f1_result[0] == LineSegment3D.from_end_points(Point3D(2, 0, 0), Point3D(0, 0, 0))
    assert f1_result[1] == LineSegment3D.from_end_points(Point3D(2, 0, 2), Point3D(0, 0, 2))
    assert len(f1_result[2]) == 0
    f2_result = face_2.extract_rectangle(0.0001)
    assert f2_result[0] == LineSegment3D.from_end_points(Point3D(2, 0, 0), Point3D(0, 0, 0))
    assert f2_result[1] == LineSegment3D.from_end_points(Point3D(2, 0, 2), Point3D(0, 0, 2))
    assert len(f2_result[2]) == 0
    f3_result = face_3.extract_rectangle(0.0001)
    assert f3_result[0] == LineSegment3D.from_end_points(Point3D(2, 0, 0), Point3D(0, 0, 0))
    assert f3_result[1] == LineSegment3D.from_end_points(Point3D(2, 0, 2), Point3D(0, 0, 2))
    assert len(f3_result[2]) == 1
    f4_result = face_4.extract_rectangle(0.0001)
    assert f4_result[0] == LineSegment3D.from_end_points(Point3D(2, 0, 0), Point3D(0, 0, 0))
    assert f4_result[1] == LineSegment3D.from_end_points(Point3D(2, 0, 2), Point3D(0, 0, 2))
    assert len(f4_result[2]) == 2
    f5_result = face_5.extract_rectangle(0.0001)
    assert f5_result[0] == LineSegment3D.from_end_points(Point3D(2, 0, 0), Point3D(0, 0, 0))
    assert f5_result[1] == LineSegment3D.from_end_points(Point3D(2, 0, 2), Point3D(0, 0, 2))
    assert len(f5_result[2]) == 2
    f6_result = face_6.extract_rectangle(0.0001)
    assert f6_result[0] == LineSegment3D.from_end_points(Point3D(2, 0, 0), Point3D(0, 0, 0))
    assert f6_result[1] == LineSegment3D.from_end_points(Point3D(2, 0, 2), Point3D(0, 0, 2))
    assert len(f6_result[2]) == 1
    f7_result = face_7.extract_rectangle(0.0001)
    assert f7_result is None


def test_extract_rectangle_complex():
    """Test the Face3D extract_rectangle method with a more complex shape."""
    pts_1 = (Point3D(-1, -1, 0), Point3D(-12, -1, 0), Point3D(-12, -1, 2),
             Point3D(-10, -1, 3), Point3D(-1, -1, 3))
    plane = Plane(Vector3D(0, 1, 0))
    face_1 = Face3D(pts_1, plane)
    f1_result = face_1.extract_rectangle(0.0001)

    assert f1_result[0] == LineSegment3D.from_end_points(Point3D(-1, -1, 0), Point3D(-10, -1, 0))
    assert f1_result[1] == LineSegment3D.from_end_points(Point3D(-1, -1, 3), Point3D(-10, -1, 3))
    assert len(f1_result[2]) == 1
    assert len(f1_result[2][0].vertices) == 4


def test_sub_faces_by_ratio_gridded():
    """Test the Face3D sub_faces_by_ratio_gridded method."""
    pts_1 = (Point3D(0, 0, 0), Point3D(12, 0, 0), Point3D(12, 0, 12), Point3D(0, 0, 6))
    face_1 = Face3D(pts_1)

    sub_faces = face_1.sub_faces_by_ratio_gridded(0.4, 2, 2)
    assert len(sub_faces) == 24
    assert sum([face.area for face in sub_faces]) == pytest.approx(face_1.area * 0.4, rel=1e-3)

    sub_faces = face_1.sub_faces_by_ratio_gridded(0.4, 12, 12)
    assert len(sub_faces) == 1
    assert sum([face.area for face in sub_faces]) == pytest.approx(face_1.area * 0.4, rel=1e-3)


def test_sub_faces_by_ratio_rectangle():
    """Test the Face3D sub_faces_by_ratio_rectangle method."""
    pts_1 = (Point3D(0, 0, 0), Point3D(12, 0, 0), Point3D(12, 0, 2), Point3D(0, 0, 3))
    plane_1 = Plane(Vector3D(0, 1, 0))
    plane_2 = Plane(Vector3D(0, -1, 0))
    face_1 = Face3D(pts_1, plane_1)
    face_2 = Face3D(pts_1, plane_2)

    sub_faces = face_1.sub_faces_by_ratio_rectangle(0.4, 0.0001)
    assert len(sub_faces) == 2
    areas = [srf.area for srf in sub_faces]
    assert sum(areas) == pytest.approx(face_1.area * 0.4, rel=1e-3)
    for face in sub_faces:
        assert face.is_clockwise is False

    sub_faces = face_2.sub_faces_by_ratio_rectangle(0.4, 0.0001)
    for face in sub_faces:
        assert face.is_clockwise is False


def test_sub_faces_by_ratio_sub_rectangle():
    """Test the Face3D sub_faces_by_ratio_sub_rectangle method."""
    pts_1 = (Point3D(0, 0, 0), Point3D(12, 0, 0), Point3D(12, 0, 2), Point3D(0, 0, 3))
    plane = Plane(Vector3D(0, 1, 0))
    face_1 = Face3D(pts_1, plane)
    window_height = 2
    sill_height = 1
    horiz_separation = 3
    vert_separation = 0

    sub_faces = face_1.sub_faces_by_ratio_sub_rectangle(
        0.4, window_height, sill_height, horiz_separation, vert_separation, 0.0001)
    assert len(sub_faces) == 5
    areas = [srf.area for srf in sub_faces]
    assert sum(areas) == pytest.approx(face_1.area * 0.4, rel=1e-3)

    sub_faces_2 = face_1.sub_faces_by_ratio_sub_rectangle(
        0.99, window_height, sill_height, horiz_separation, vert_separation, 0.0001)
    assert len(sub_faces_2) == 2
    areas = [srf.area for srf in sub_faces_2]
    assert sum(areas) == pytest.approx(face_1.area * 0.99, rel=1e-3)


def test_sub_faces_by_ratio():
    """Test the sub_faces_by_ratio method."""
    pts_1 = (Point3D(0, 0), Point3D(0, 2), Point3D(2, 2), Point3D(2, 0))
    pts_2 = (Point3D(0, 0), Point3D(2, 0), Point3D(2, 1), Point3D(1, 1),
             Point3D(1, 2), Point3D(0, 2))
    plane = Plane(Vector3D(0, 0, 1))
    face_1 = Face3D(pts_1, plane)
    face_2 = Face3D(pts_2, plane)

    sub_faces_1 = face_1.sub_faces_by_ratio(0.5)
    assert len(sub_faces_1) == 1
    assert sub_faces_1[0].area == pytest.approx(face_1.area * 0.5, rel=1e-3)

    sub_faces_2 = face_2.sub_faces_by_ratio(0.5)
    assert len(sub_faces_2) == 4
    areas = [srf.area for srf in sub_faces_2]
    assert sum(areas) == pytest.approx(face_2.area * 0.5, rel=1e-3)


def test_sub_faces_by_dimension_rectangle():
    """Test the sub_faces_by_dimension_rectangle method."""
    pts_1 = (Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(2, 0, 0))
    pts_2 = (Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(4, 0, 2), Point3D(4, 0, 0))
    plane = Plane(Vector3D(0, 0, 1))
    face_1 = Face3D(pts_1)
    face_2 = Face3D(pts_2)
    sub_face_height = 1.0
    sill_height = 1.0
    rect_height = 2.0
    div_dist = 1.0

    sub_faces_1 = face_1.sub_faces_by_dimension_rectangle(
        sub_face_height, 3.0, sill_height, div_dist, 0.1)
    assert len(sub_faces_1) == 1
    segs_1 = sub_faces_1[0].boundary_segments
    assert segs_1[1].length == sub_face_height

    sub_faces_2 = face_2.sub_faces_by_dimension_rectangle(
        sub_face_height, 1.0, sill_height, div_dist, 0.1)
    assert len(sub_faces_2) == 3
    segs_2 = sub_faces_2[0].boundary_segments
    assert segs_2[0].length == 1.0
    assert segs_2[1].length == sub_face_height
