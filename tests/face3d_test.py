# coding=utf-8
import pytest

from ladybug_geometry.geometry2d import Vector2D, Polygon2D
from ladybug_geometry.geometry3d import Point3D, Vector3D, Ray3D, LineSegment3D, \
    Plane, Face3D

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
    assert round(face.altitude, 3) == round(math.pi / 2, 3)
    assert round(face.azimuth, 3) == 0
    assert face.is_clockwise is False
    assert face.is_convex is True
    assert face.is_self_intersecting is False
    assert face.vertices[0] == face[0]


def test_face3d_pole_of_inaccessibility():
    """Test the Face3D.pole_of_inaccessibility method."""
    pts = (Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2))
    plane = Plane(Vector3D(0, 0, 1), Point3D(0, 0, 2))
    face = Face3D(pts, plane)

    pole = face.pole_of_inaccessibility(0.01)
    assert isinstance(pole, Point3D)
    assert pole.x == pytest.approx(1.0, rel=1e-3)
    assert pole.y == pytest.approx(1.0, rel=1e-3)
    assert pole.z == pytest.approx(2.0, rel=1e-3)

    deg_pts = (Point3D(0, 0, 2), Point3D(0, 0.0001, 2),
               Point3D(2, 0.0001, 2), Point3D(2, 0, 2))
    plane = Plane(Vector3D(0, 0, 1), Point3D(0, 0, 2))
    deg_face = Face3D(deg_pts, plane)
    deg_pole = deg_face.pole_of_inaccessibility(0.01)
    assert isinstance(deg_pole, Point3D)


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


def test_face3d_no_holes():
    """Test the initialization of Face3D without holes."""
    pts = (Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2))
    plane = Plane(Vector3D(0, 0, 1), Point3D(0, 0, 2))
    face = Face3D(pts, plane, None)
    assert not face.has_holes
    face = Face3D(pts, plane, [])
    assert not face.has_holes


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


def test_face3d_shape_with_many_holes():
    """Test Face3D from_shape_with_holes with a Face3D above the vertex threshold."""
    geo_file = './tests/json/face_with_200_holes.json'
    with open(geo_file, 'r') as fp:
        geo_dict = json.load(fp)
    face_geo = Face3D.from_dict(geo_dict)

    v_count = len(face_geo.boundary)  # count the vertices for hole merging method
    for h in face_geo.holes:
        v_count += len(h)
    assert v_count > face_geo.HOLE_VERTEX_THRESHOLD
    assert len(face_geo.vertices) > v_count + len(face_geo.holes)

    assert face_geo.boundary[0] == face_geo.vertices[0]
    assert face_geo.area == pytest.approx(62.416884, rel=1e-4)


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


def test_face3d_split_through_holes():
    """Test the Face3D split_through_holes method."""
    bound_pts = [Point3D(0, 0), Point3D(4, 0), Point3D(4, 4), Point3D(0, 4)]
    hole_pts = [Point3D(1, 1), Point3D(2, 1), Point3D(2, 2), Point3D(1, 2)]
    face = Face3D(bound_pts, None, [hole_pts])

    face_1, face_2 = face.split_through_holes()
    assert len(face_1.vertices) + len(face_2.vertices) == 12

    bound_pts = [Point3D(0, 0), Point3D(4, 0), Point3D(4, 4), Point3D(0, 4)]
    hole_pts = [Point3D(2, 2), Point3D(3, 2), Point3D(3, 3), Point3D(2, 3)]
    face = Face3D(bound_pts, None, [hole_pts])

    face_1, face_2 = face.split_through_holes()
    assert len(face_1.vertices) + len(face_2.vertices) == 12

    bound_pts = [Point3D(0, 0), Point3D(4, 0), Point3D(4, 4), Point3D(0, 4)]
    hole_pts_1 = [Point3D(1, 1), Point3D(1.5, 1), Point3D(1.5, 1.5), Point3D(1, 1.5)]
    hole_pts_2 = [Point3D(2, 2), Point3D(3, 2), Point3D(3, 3), Point3D(2, 3)]
    face = Face3D(bound_pts, None, [hole_pts_1, hole_pts_2])

    face_1, face_2 = face.split_through_holes()
    assert len(face_1.vertices) + len(face_2.vertices) == 18


def test_face3d_split_through_holes_detailed():
    """Test the Face3D split_through_holes method with a detailed Face3D."""
    geo_file = './tests/json/tri_hole_face.json'
    with open(geo_file, 'r') as fp:
        geo_dict = json.load(fp)
    face_1 = Face3D.from_dict(geo_dict)
    s_faces = face_1.split_through_holes()
    assert 2 < len(s_faces) < 20


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


def test_corner_vertices():
    """Test the various properties for corner vertices."""
    plane_1 = Plane(Vector3D(0, 1, 0))
    plane_2 = Plane(Vector3D(0, -1, 0))
    pts_1 = (Point3D(0, 0, 0), Point3D(2, 0, 0), Point3D(2, 0, 2), Point3D(0, 0, 2))
    pts_2 = (Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(2, 0, 0))
    face_1 = Face3D(pts_1, plane_1)
    face_2 = Face3D(pts_2, plane_1)
    face_3 = Face3D(pts_1, plane_2)

    assert face_1.upper_left_corner.is_equivalent(Point3D(2, 0, 2), 1e-6)
    assert face_1.lower_left_corner.is_equivalent(Point3D(2, 0, 0), 1e-6)
    assert face_1.upper_right_corner.is_equivalent(Point3D(0, 0, 2), 1e-6)
    assert face_1.lower_right_corner.is_equivalent(Point3D(0, 0, 0), 1e-6)

    assert face_2.upper_left_corner.is_equivalent(Point3D(2, 0, 2), 1e-6)
    assert face_2.lower_left_corner.is_equivalent(Point3D(2, 0, 0), 1e-6)
    assert face_2.upper_right_corner.is_equivalent(Point3D(0, 0, 2), 1e-6)
    assert face_2.lower_right_corner.is_equivalent(Point3D(0, 0, 0), 1e-6)

    assert face_3.upper_left_corner.is_equivalent(Point3D(0, 0, 2), 1e-6)
    assert face_3.lower_left_corner.is_equivalent(Point3D(0, 0, 0), 1e-6)
    assert face_3.upper_right_corner.is_equivalent(Point3D(2, 0, 2), 1e-6)
    assert face_3.lower_right_corner.is_equivalent(Point3D(2, 0, 0), 1e-6)


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
    pts_3 = (Point3D(0, 0), Point3D(2, 0), Point3D(2, 2),
             Point3D(0, 2), Point3D(0, 2))
    pts_4 = (Point3D(24.21, 23.22, 8.00), Point3D(24.21, 23.22, 11.60),
             Point3D(24.61, 23.22, 11.60), Point3D(27.85, 23.22, 11.60),
             Point3D(30.28, 23.22, 11.60), Point3D(30.28, 23.22, 0.00),
             Point3D(27.85, 23.22, 0.00), Point3D(27.85, 23.22, 4.00),
             Point3D(27.85, 23.22, 8.00), Point3D(27.85, 23.22, 8.00),
             Point3D(24.61, 23.22, 8.00), Point3D(24.61, 23.22, 8.00))
    plane_1 = Plane(Vector3D(0, 0, 1))
    face_1 = Face3D(pts_1, plane_1)
    face_2 = Face3D(pts_2, plane_1)
    face_3 = Face3D(pts_3, plane_1)
    face_4 = Face3D(pts_4)

    assert len(face_1.remove_colinear_vertices(0.0001).vertices) == 4
    assert len(face_2.remove_colinear_vertices(0.0001).vertices) == 4
    assert len(face_3.remove_colinear_vertices(0.0001).vertices) == 4
    assert len(face_4.remove_colinear_vertices(0.0001).vertices) == 6


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


def test_triangulated_mesh_edge_case():
    """Test triangulation properties of a Face3D triggering a zero division."""
    geo_file = './tests/json/edge_case_face.json'
    with open(geo_file, 'r') as fp:
        geo_dict = json.load(fp)
    face_geo = Face3D.from_dict(geo_dict)
    assert face_geo.triangulated_mesh3d.area != 0


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


def test_normal_with_slightly_nonplanar():
    """Test that a slightly non-planar shape still has a relatively close normal."""
    geo_file = './tests/json/slight_nonplanar_face.json'
    with open(geo_file, 'r') as fp:
        geo_dict = json.load(fp)
    face_geo = Face3D.from_dict(geo_dict)

    assert 0.99 < face_geo.normal.z < 1.01
    assert not face_geo.check_planar(0.000001, False)
    with pytest.raises(Exception):
        face_geo.check_planar(0.000001)


def test_normal_with_colinear_vertices():
    """Test that shapes with colinear vertices have a relatively close normal."""
    geo_file = './tests/json/faces_colinear_verts.json'
    with open(geo_file, 'r') as fp:
        geo_dict = json.load(fp)
    face_geos = [Face3D.from_dict(geo) for geo in geo_dict]

    assert -0.01 < face_geos[0].normal.z < 0.01
    assert -0.01 < face_geos[1].normal.z < 0.01


def test_normal_with_jagged_vertices():
    """Test that shapes with colinear vertices have a relatively close normal."""
    geo_file = './tests/json/face_jagged.json'
    with open(geo_file, 'r') as fp:
        geo_dict = json.load(fp)
    face_geo = Face3D.from_dict(geo_dict)

    assert 0.99 < face_geo.normal.z < 1.01
    assert not face_geo.check_planar(0.000001, False)
    face_geo.remove_colinear_vertices(0.01)  # correct plane ensures removal of verts


def test_jagged_l_face():
    """Test that shapes with colinear vertices have a relatively close normal."""
    geo_file = './tests/json/jagged_l.json'
    with open(geo_file, 'r') as fp:
        geo_dict = json.load(fp)
    face_geo = Face3D.from_dict(geo_dict)

    assert -0.99 > face_geo.normal.z > -1.01


def test_face_with_tough_normal():
    """Test that shapes with perfect symmetry to undermine normal calc."""
    geo_file = './tests/json/face_with_tough_normal.json'
    with open(geo_file, 'r') as fp:
        geo_dict = json.load(fp)
    face_geo = Face3D.from_dict(geo_dict)

    assert 0.99 < face_geo.normal.z < 1.01
    face_geo.remove_colinear_vertices(0.01)  # correct plane ensures removal of verts


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
    test_2 = face.rotate(axis, math.pi / 2, origin)
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

    test_2 = face.rotate_xy(math.pi / 2, origin_1)
    assert test_2[0].x == pytest.approx(1, rel=1e-3)
    assert test_2[0].y == pytest.approx(1, rel=1e-3)
    assert test_1[0].z == pytest.approx(2, rel=1e-3)
    assert test_2[2].x == pytest.approx(0, rel=1e-3)
    assert test_2[2].y == pytest.approx(2, rel=1e-3)
    assert test_1[2].z == pytest.approx(2, rel=1e-3)

    face = Face3D(pts, plane, enforce_right_hand=False)
    test_1 = face.rotate_xy(math.pi, origin_1)
    assert test_1.is_clockwise is False
    test_2 = face.rotate_xy(math.pi / 2, origin_1)
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


def test_contour_by_number():
    """Test the contour_by_number method."""
    pts_1 = [Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(2, 0, 0)]
    pts_2 = [Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(4, 0, 0)]
    plane = Plane(Vector3D(0, 1, 0))
    face_1 = Face3D(pts_1, plane)
    face_2 = Face3D(pts_2, plane)

    contours = face_1.contour_by_number(4, Vector2D(0, 1), False, 0.01)
    assert len(contours) == 4
    assert contours[0].p2.z == pytest.approx(2, rel=1e-3)
    assert contours[-1].p2.z == pytest.approx(0.5, rel=1e-3)

    contours = face_1.contour_by_number(4, Vector2D(1), False, 0.01)
    assert len(contours) == 4
    assert contours[-1].p2.x == pytest.approx(1.5, rel=1e-3)

    contours = face_1.contour_by_number(4, Vector2D(1), True, 0.01)
    assert len(contours) == 4
    assert contours[-1].p2.x == pytest.approx(0.5, rel=1e-3)

    contours = face_2.contour_by_number(4, Vector2D(0, 1), False, 0.01)
    assert len(contours) == 4
    contours = face_2.contour_by_number(8, Vector2D(1), False, 0.01)
    assert len(contours) == 8


def test_contour_by_distance_between():
    """Test the contour_by_distance_between method."""
    pts_1 = [Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(2, 0, 0)]
    pts_2 = [Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(4, 0, 0)]
    plane = Plane(Vector3D(0, 1, 0))
    face_1 = Face3D(pts_1, plane)
    face_2 = Face3D(pts_2, plane)

    contours = face_1.contour_by_distance_between(0.5, Vector2D(0, 1), False, 0.01)
    assert len(contours) == 4
    assert contours[0].p2.z == pytest.approx(2, rel=1e-3)
    assert contours[-1].p2.z == pytest.approx(0.5, rel=1e-3)

    contours = face_1.contour_by_distance_between(0.5, Vector2D(1), False, 0.01)
    assert len(contours) == 4
    assert contours[-1].p2.x == pytest.approx(1.5, rel=1e-3)

    contours = face_1.contour_by_distance_between(0.5, Vector2D(1), True, 0.01)
    assert len(contours) == 4
    assert contours[-1].p2.x == pytest.approx(0.5, rel=1e-3)

    contours = face_2.contour_by_distance_between(0.5, Vector2D(0, 1), False, 0.01)
    assert len(contours) == 4
    contours = face_2.contour_by_distance_between(0.5, Vector2D(1), False, 0.01)
    assert len(contours) == 8


def test_contour_fins_by_number():
    """Test the contour_fins_by_number method."""
    pts_1 = [Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(2, 0, 0)]
    plane = Plane(Vector3D(0, 1, 0))
    face_1 = Face3D(pts_1, plane)

    fins = face_1.contour_fins_by_number(4, 0.5, 0.5, 0, Vector2D(0, 1), False, 0.01)
    assert len(fins) == 4

    fins = face_1.contour_fins_by_number(4, 0.5, 0.5, 0, Vector2D(1), False, 0.01)
    assert len(fins) == 4

    fins = face_1.contour_fins_by_number(
        4, 0.5, 0.5, math.pi / 4, Vector2D(0, 1), False, 0.01)
    assert len(fins) == 4


def test_contour_fins_by_distance_between():
    """Test the contour_fins_by_distance_between method."""
    pts_1 = [Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(2, 0, 0)]
    pts_2 = [Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(4, 0, 0)]
    plane = Plane(Vector3D(0, 1, 0))
    face_1 = Face3D(pts_1, plane)
    face_2 = Face3D(pts_2, plane)

    fins = face_1.contour_fins_by_distance_between(
        0.5, 0.5, 0.5, 0, Vector2D(0, 1), False, 0.01)
    assert len(fins) == 4

    fins = face_1.contour_fins_by_distance_between(
        0.25, 0.5, 0.5, 0, Vector2D(1), False, 0.01)
    assert len(fins) == 8

    fins = face_2.contour_fins_by_distance_between(
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
    assert f1_result[0] == LineSegment3D.from_end_points(
        Point3D(2, 0, 0), Point3D(0, 0, 0))
    assert f1_result[1] == LineSegment3D.from_end_points(
        Point3D(2, 0, 2), Point3D(0, 0, 2))
    assert len(f1_result[2]) == 0
    f2_result = face_2.extract_rectangle(0.0001)
    assert f2_result[0] == LineSegment3D.from_end_points(
        Point3D(2, 0, 0), Point3D(0, 0, 0))
    assert f2_result[1] == LineSegment3D.from_end_points(
        Point3D(2, 0, 2), Point3D(0, 0, 2))
    assert len(f2_result[2]) == 0
    f3_result = face_3.extract_rectangle(0.0001)
    assert f3_result[0] == LineSegment3D.from_end_points(
        Point3D(2, 0, 0), Point3D(0, 0, 0))
    assert f3_result[1] == LineSegment3D.from_end_points(
        Point3D(2, 0, 2), Point3D(0, 0, 2))
    assert len(f3_result[2]) == 1
    f4_result = face_4.extract_rectangle(0.0001)
    assert f4_result[0] == LineSegment3D.from_end_points(
        Point3D(2, 0, 0), Point3D(0, 0, 0))
    assert f4_result[1] == LineSegment3D.from_end_points(
        Point3D(2, 0, 2), Point3D(0, 0, 2))
    assert len(f4_result[2]) == 2
    f5_result = face_5.extract_rectangle(0.0001)
    assert f5_result[0] == LineSegment3D.from_end_points(
        Point3D(2, 0, 0), Point3D(0, 0, 0))
    assert f5_result[1] == LineSegment3D.from_end_points(
        Point3D(2, 0, 2), Point3D(0, 0, 2))
    assert len(f5_result[2]) == 2
    f6_result = face_6.extract_rectangle(0.0001)
    assert f6_result[0] == LineSegment3D.from_end_points(
        Point3D(2, 0, 0), Point3D(0, 0, 0))
    assert f6_result[1] == LineSegment3D.from_end_points(
        Point3D(2, 0, 2), Point3D(0, 0, 2))
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

    assert f1_result[0] == LineSegment3D.from_end_points(
        Point3D(-1, -1, 0), Point3D(-10, -1, 0))
    assert f1_result[1] == LineSegment3D.from_end_points(
        Point3D(-1, -1, 3), Point3D(-10, -1, 3))
    assert len(f1_result[2]) == 1
    assert len(f1_result[2][0].vertices) == 4


def test_sub_faces_by_ratio_gridded():
    """Test the Face3D sub_faces_by_ratio_gridded method."""
    pts_1 = (Point3D(0, 0, 0), Point3D(12, 0, 0), Point3D(12, 0, 12), Point3D(0, 0, 6))
    face_1 = Face3D(pts_1)

    sub_faces = face_1.sub_faces_by_ratio_gridded(0.4, 2, 2)
    assert len(sub_faces) == 24
    assert sum([face.area for face in sub_faces]) == \
        pytest.approx(face_1.area * 0.4, rel=1e-3)

    sub_faces = face_1.sub_faces_by_ratio_gridded(0.4, 12, 12)
    assert len(sub_faces) == 1
    assert sum([face.area for face in sub_faces]) == \
        pytest.approx(face_1.area * 0.4, rel=1e-3)


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
    for sf in sub_faces:
        assert face_1.is_sub_face(sf, 0.01, 1)

    sub_faces_2 = face_1.sub_faces_by_ratio_sub_rectangle(
        0.95, window_height, sill_height, horiz_separation, vert_separation, 0.0001)
    assert len(sub_faces_2) == 5
    areas = [srf.area for srf in sub_faces_2]
    assert sum(areas) == pytest.approx(face_1.area * 0.95, rel=1e-3)
    for sf in sub_faces_2:
        assert face_1.is_sub_face(sf, 0.01, 1)


def test_sub_faces_by_ratio_sub_rectangle_non_rect():
    """Test the sub_faces_by_ratio_sub_rectangle method with a non-rectangular face."""
    pts_1 = (Point3D(0, 0, 0), Point3D(5, 0, 0), Point3D(5, 0, 6), Point3D(0, 0, 5))
    face_1 = Face3D(pts_1)

    sub_faces = face_1.sub_faces_by_ratio_sub_rectangle(0.25, 2, 0.8, 3, 0, 0.01)
    assert len(sub_faces) == 3
    areas = [srf.area for srf in sub_faces]
    assert sum(areas) == pytest.approx(face_1.area * 0.25, rel=1e-3)
    for sf in sub_faces:
        assert face_1.is_sub_face(sf, 0.01, 1)

    pts_2 = [
        Point3D(34.068566268151841, 18.694085210192, 3.9999999999999987),
        Point3D(34.068566268151862, 18.694085210192011, 7.0),
        Point3D(34.442641391689776, 21.198643189820409, 7.0000000000000018),
        Point3D(34.442641391689783, 21.198643189820412, 7.4999999999999973),
        Point3D(33.710274088688173, 16.295199316291654, 7.5),
        Point3D(33.710274088688173, 16.295199316291651, 4.0)
    ]
    face_2 = Face3D(pts_2)

    sub_faces = face_2.sub_faces_by_ratio_sub_rectangle(0.25, 2, 0.8, 3, 0, 0.01)
    assert len(sub_faces) == 4
    areas = [srf.area for srf in sub_faces]
    assert all(a > 0.001 for a in areas)
    assert sum(areas) == pytest.approx(face_2.area * 0.25, rel=1e-3)
    for sf in sub_faces:
        assert face_2.is_sub_face(sf, 0.01, 1)


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
    face_1 = Face3D(pts_1)
    face_2 = Face3D(pts_2)
    sub_face_height = 1.0
    sill_height = 1.0
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


def test_sub_faces_by_ratio_sub_rectangle_tol_issue():
    """Test the Face3D sub_faces_by_ratio_sub_rectangle method."""
    geo_file = './tests/json/face_tolerance_by_ratio.json'
    with open(geo_file, 'r') as fp:
        geo_dict = json.load(fp)
    face_1 = Face3D.from_dict(geo_dict)
    window_height = 10
    sill_height = 3
    horiz_separation = 9
    vert_separation = 0

    sub_faces = face_1.sub_faces_by_ratio_sub_rectangle(
        0.4, window_height, sill_height, horiz_separation, vert_separation, 0.0001)
    assert len(sub_faces) == 9
    areas = [srf.area for srf in sub_faces]
    assert sum(areas) == pytest.approx(face_1.area * 0.4, rel=1e-3)
    for sf in sub_faces:
        assert face_1.is_sub_face(sf, 0.01, 1)


def test_coplanar_union():
    """Test the coplanar_union method."""
    b_pts1 = (Point3D(-14.79, -36.61, 0.00), Point3D(6.68, -36.61, 0.00),
              Point3D(6.68, -10.37, 0.00), Point3D(-14.79, -10.37, 0.00))
    h_pts1 = (
        (Point3D(-9.71, -22.66, 0.00), Point3D(-9.71, -16.19, 0.00),
         Point3D(-5.01, -16.19, 0.00), Point3D(-5.01, -22.66, 0.00)),
        (Point3D(0.31, -35.20, 0.00), Point3D(0.31, -29.89, 0.00),
         Point3D(5.15, -29.89, 0.00), Point3D(5.15, -35.20, 0.00))
    )
    b_pts2 = (Point3D(-7.02, -32.34, 0.00), Point3D(13.23, -32.34, 0.00),
              Point3D(13.23, -18.57, 0.00), Point3D(-7.02, -18.57, 0.00))
    face1 = Face3D(b_pts1, holes=h_pts1)
    face2 = Face3D(b_pts2)

    face_union = Face3D.coplanar_union(face1, face2, 0.01, 1)

    assert isinstance(face_union, Face3D)
    assert len(face_union.holes) == 2


def test_coplanar_split():
    """Test the coplanar_split method."""
    b_pts1 = (Point3D(-14.79, -36.61, 0.00), Point3D(6.68, -36.61, 0.00),
              Point3D(6.68, -10.37, 0.00), Point3D(-14.79, -10.37, 0.00))
    h_pts1 = (
        (Point3D(-9.71, -22.66, 0.00), Point3D(-9.71, -16.19, 0.00),
         Point3D(-5.01, -16.19, 0.00), Point3D(-5.01, -22.66, 0.00)),
        (Point3D(0.31, -35.20, 0.00), Point3D(0.31, -29.89, 0.00),
         Point3D(5.15, -29.89, 0.00), Point3D(5.15, -35.20, 0.00))
    )
    b_pts2 = (Point3D(-7.02, -32.34, 0.00), Point3D(13.23, -32.34, 0.00),
              Point3D(13.23, -18.57, 0.00), Point3D(-7.02, -18.57, 0.00))
    face1 = Face3D(b_pts1, holes=h_pts1)
    face2 = Face3D(b_pts2)

    split1, split2 = Face3D.coplanar_split(face1, face2, 0.01, 1)

    assert len(split1) == 2
    assert len(split2) == 4


def test_group_by_coplanar_overlap():
    """Test the group_by_coplanar_overlap method."""
    bound_pts1 = [Point3D(0, 0), Point3D(4, 0), Point3D(4, 4), Point3D(0, 4)]
    bound_pts2 = [Point3D(2, 2), Point3D(6, 2), Point3D(6, 6), Point3D(2, 6)]
    bound_pts3 = [Point3D(6, 6), Point3D(7, 6), Point3D(7, 7), Point3D(6, 7)]
    face1 = Face3D(bound_pts1)
    face2 = Face3D(bound_pts2)
    face3 = Face3D(bound_pts3)

    all_faces = [face1, face2, face3]

    grouped_faces = Face3D.group_by_coplanar_overlap(all_faces, 0.01)
    assert len(grouped_faces) == 2
    assert len(grouped_faces[0]) == 2
    assert len(grouped_faces[1]) == 1

    grouped_faces = Face3D.group_by_coplanar_overlap(list(reversed(all_faces)), 0.01)
    assert len(grouped_faces) == 2
    assert len(grouped_faces[0]) == 1
    assert len(grouped_faces[1]) == 2


def test_join_coplanar_faces():
    """Test the join_coplanar_faces method."""
    geo_file = './tests/json/polygons_for_joined_boundary.json'
    with open(geo_file, 'r') as fp:
        geo_dict = json.load(fp)
    polygons = [Polygon2D.from_dict(p) for p in geo_dict]
    faces = [Face3D([Point3D(p.x, p.y, 0) for p in poly]) for poly in polygons]

    joined_faces = Face3D.join_coplanar_faces(faces, 0.01)
    assert len(joined_faces) == 1
    assert joined_faces[0].has_holes
    assert len(joined_faces[0].holes) == 5


def test_extract_all_from_stl():
    file_path = 'tests/stl/cube_binary.stl'
    faces = Face3D.extract_all_from_stl(file_path)
    assert len(faces) == 12
    assert all((isinstance(f, Face3D) for f in faces))
    assert all((len(f) == 3 for f in faces))
