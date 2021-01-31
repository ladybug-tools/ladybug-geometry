# coding=utf-8
import pytest

from ladybug_geometry.geometry2d.polyline import Polyline2D
from ladybug_geometry.geometry2d.pointvector import Point2D
from ladybug_geometry.geometry3d.polyline import Polyline3D
from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
from ladybug_geometry.geometry3d.line import LineSegment3D
from ladybug_geometry.geometry3d.plane import Plane

import math


def test_polyline3d_init():
    """Test the initialization of Polyline3D objects and basic properties."""
    pts = (Point3D(0, 0), Point3D(2, 0), Point3D(2, 2), Point3D(0, 2))
    pline = Polyline3D(pts)

    str(pline)  # test the string representation of the polyline

    assert isinstance(pline.vertices, tuple)
    assert len(pline.vertices) == 4
    assert len(pline) == 4
    for point in pline:
        assert isinstance(point, Point3D)

    assert isinstance(pline.segments, tuple)
    assert len(pline.segments) == 3
    for seg in pline.segments:
        assert isinstance(seg, LineSegment3D)
        assert seg.length == 2

    assert pline.p1 == pts[0]
    assert pline.p2 == pts[-1]

    assert pline.length == 6
    assert pline.vertices[0] == pline[0]

    p_array = pline.to_array()
    assert isinstance(p_array, tuple)
    assert len(p_array) == 4
    for arr in p_array:
        assert isinstance(p_array, tuple)
        assert len(arr) == 3
    pline_2 = Polyline3D.from_array(p_array)
    assert pline == pline_2


def test_equality():
    """Test the equality of Polyline3D objects."""
    pts = (Point3D(0, 0), Point3D(2, 0), Point3D(2, 2), Point3D(0, 2))
    pts_2 = (Point3D(0, 0), Point3D(2, 0), Point3D(2, 2), Point3D(0.1, 2))
    pline = Polyline3D(pts)
    pline_dup = pline.duplicate()
    pline_alt = Polyline3D(pts_2)

    assert pline is pline
    assert pline is not pline_dup
    assert pline == pline_dup
    assert hash(pline) == hash(pline_dup)
    assert pline != pline_alt
    assert hash(pline) != hash(pline_alt)


def test_to_from_polygon():
    """Test the from_polygon method."""
    pts_1 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    pline2d = Polyline2D(pts_1, interpolated=True)

    pline = Polyline3D.from_polyline2d(pline2d)
    assert isinstance(pline, Polyline3D)
    assert len(pline) == 4
    assert pline.interpolated

    new_pline2d = pline.to_polyline2d()
    assert isinstance(new_pline2d, Polyline2D)
    assert len(new_pline2d) == 4
    assert new_pline2d.interpolated


def test_polyline3d_to_from_dict():
    """Test the to/from dict of Polyline3D objects."""
    pts = (Point3D(0, 0), Point3D(2, 0), Point3D(2, 2), Point3D(0, 2))
    pline = Polyline3D(pts)
    pline_dict = pline.to_dict()
    new_pline = Polyline3D.from_dict(pline_dict)
    assert isinstance(new_pline, Polyline3D)
    assert new_pline.to_dict() == pline_dict


def test_min_max_center():
    """Test the Polyline3D min, max and center."""
    pts = (Point3D(0, 0, 3), Point3D(2, 0, 3), Point3D(2, 2, 3), Point3D(0, 2, 3))
    pline = Polyline3D(pts)

    assert pline.min == Point3D(0, 0, 3)
    assert pline.max == Point3D(2, 2, 3)
    assert pline.center == Point3D(1, 1, 3)


def test_remove_colinear_vertices():
    """Test the remove_colinear_vertices method of Polyline3D."""
    pts_1 = (Point3D(0, 0, 3), Point3D(2, 0, 3), Point3D(2, 2, 3), Point3D(0, 2, 3))
    pts_2 = (Point3D(0, 0, 3), Point3D(1, 0, 3), Point3D(2, 0, 3), Point3D(2, 2, 3),
             Point3D(0, 2, 3))
    pline_1 = Polyline3D(pts_1)
    pline_2 = Polyline3D(pts_2)

    assert len(pline_1.remove_colinear_vertices(0.0001).vertices) == 4
    assert len(pline_2.remove_colinear_vertices(0.0001).vertices) == 4


def test_pline3d_duplicate():
    """Test the duplicate method of Polyline3D."""
    pts = (Point3D(0, 0), Point3D(2, 0), Point3D(2, 2), Point3D(0, 2))
    pline = Polyline3D(pts)
    new_pline = pline.duplicate()

    for i, pt in enumerate(new_pline):
        assert pt == pts[i]

    assert pline.length == new_pline.length
    assert pline.vertices == new_pline.vertices


def test_reverse():
    """Test the reverse method."""
    pts_1 = (Point3D(0, 0), Point3D(2, 0), Point3D(2, 2), Point3D(0, 2))
    pline = Polyline3D(pts_1)
    new_pline = pline.reverse()

    assert pline.length == new_pline.length
    assert pline.vertices == tuple(reversed(new_pline.vertices))


def test_move():
    """Test the Polyline3D move method."""
    pts = (Point3D(0, 0), Point3D(2, 0), Point3D(2, 2), Point3D(0, 2))
    pline = Polyline3D(pts)

    vec_1 = Vector3D(2, 2, 2)
    new_pline = pline.move(vec_1)
    assert new_pline[0] == Point3D(2, 2, 2)
    assert new_pline[1] == Point3D(4, 2, 2)
    assert new_pline[2] == Point3D(4, 4, 2)
    assert new_pline[3] == Point3D(2, 4, 2)
    assert pline.length == new_pline.length


def test_scale():
    """Test the Polyline3D scale method."""
    pts_1 = (Point3D(0, 0), Point3D(2, 0), Point3D(2, 2), Point3D(0, 2))
    pline_1 = Polyline3D(pts_1)
    pts_2 = (Point3D(1, 1), Point3D(2, 1), Point3D(2, 2), Point3D(1, 2))
    pline_2 = Polyline3D(pts_2)
    origin_1 = Point3D(2, 0)
    origin_2 = Point3D(1, 1)

    new_pline_1 = pline_1.scale(2, origin_1)
    assert new_pline_1[0] == Point3D(-2, 0)
    assert new_pline_1[1] == Point3D(2, 0)
    assert new_pline_1[2] == Point3D(2, 4)
    assert new_pline_1[3] == Point3D(-2, 4)
    assert new_pline_1.length == pline_1.length * 2

    new_pline_2 = pline_2.scale(2, origin_2)
    assert new_pline_2[0] == Point3D(1, 1)
    assert new_pline_2[1] == Point3D(3, 1)
    assert new_pline_2[2] == Point3D(3, 3)
    assert new_pline_2[3] == Point3D(1, 3)
    assert new_pline_2.length == pline_2.length * 2


def test_scale_world_origin():
    """Test the Polyline3D scale method with None origin."""
    pts = (Point3D(1, 1, 2), Point3D(2, 1, 2), Point3D(2, 2, 2), Point3D(1, 2, 2))
    pline = Polyline3D(pts)

    new_pline = pline.scale(2)
    assert new_pline[0] == Point3D(2, 2, 4)
    assert new_pline[1] == Point3D(4, 2, 4)
    assert new_pline[2] == Point3D(4, 4, 4)
    assert new_pline[3] == Point3D(2, 4, 4)
    assert new_pline.length == pline.length * 2


def test_rotate():
    """Test the Polyline3D rotate method."""
    pts = (Point3D(1, 1, 2), Point3D(2, 1, 2), Point3D(2, 2, 2), Point3D(1, 2, 2))
    pline = Polyline3D(pts)
    origin_1 = Point3D(1, 1)

    test_1 = pline.rotate(Vector3D(0, 0, 1), math.pi, origin_1)
    assert test_1[0].x == pytest.approx(1, rel=1e-3)
    assert test_1[0].y == pytest.approx(1, rel=1e-3)
    assert test_1[2].x == pytest.approx(0, rel=1e-3)
    assert test_1[2].y == pytest.approx(0, rel=1e-3)
    assert pline.length == pytest.approx(test_1.length, rel=1e-3)

    test_2 = pline.rotate(Vector3D(0, 0, 1), math.pi/2, origin_1)
    assert test_2[0].x == pytest.approx(1, rel=1e-3)
    assert test_2[0].y == pytest.approx(1, rel=1e-3)
    assert test_2[2].x == pytest.approx(0, rel=1e-3)
    assert test_2[2].y == pytest.approx(2, rel=1e-3)


def test_rotate_xy():
    """Test the Polyline3D rotate_xy method."""
    pts = (Point3D(1, 1, 2), Point3D(2, 1, 2), Point3D(2, 2, 2), Point3D(1, 2, 2))
    pline = Polyline3D(pts)
    origin_1 = Point3D(1, 1)

    test_1 = pline.rotate_xy(math.pi, origin_1)
    assert test_1[0].x == pytest.approx(1, rel=1e-3)
    assert test_1[0].y == pytest.approx(1, rel=1e-3)
    assert test_1[2].x == pytest.approx(0, rel=1e-3)
    assert test_1[2].y == pytest.approx(0, rel=1e-3)
    assert pline.length == pytest.approx(test_1.length, rel=1e-3)

    test_2 = pline.rotate_xy(math.pi/2, origin_1)
    assert test_2[0].x == pytest.approx(1, rel=1e-3)
    assert test_2[0].y == pytest.approx(1, rel=1e-3)
    assert test_2[2].x == pytest.approx(0, rel=1e-3)
    assert test_2[2].y == pytest.approx(2, rel=1e-3)


def test_reflect():
    """Test the Polyline3D reflect method."""
    pts = (Point3D(1, 1, 2), Point3D(2, 1, 2), Point3D(2, 2, 2), Point3D(1, 2, 2))
    pline = Polyline3D(pts)

    origin_1 = Point3D(1, 0)
    normal_1 = Vector3D(1, 0)
    normal_2 = Vector3D(-1, -1).normalize()

    test_1 = pline.reflect(normal_1, origin_1)
    assert test_1[0].x == pytest.approx(1, rel=1e-3)
    assert test_1[0].y == pytest.approx(1, rel=1e-3)
    assert test_1[2].x == pytest.approx(0, rel=1e-3)
    assert test_1[2].y == pytest.approx(2, rel=1e-3)
    assert pline.length == pytest.approx(test_1.length, rel=1e-3)


def test_intersect_plane():
    """Test the Polyline3D intersect_plane method."""
    pts = (Point3D(0, 0, 1), Point3D(2, 0, -1), Point3D(2, 2, 1), Point3D(0, 2, -1))
    pline = Polyline3D(pts)

    plane_1 = Plane(n=Vector3D(0, 0, 1), o=Point3D(0, 0, 0))
    assert len(pline.intersect_plane(plane_1)) == 3

    plane_2 = Plane(n=Vector3D(0, 1, 0), o=Point3D(1, 1, 0))
    assert len(pline.intersect_plane(plane_2)) == 1

    plane_3 = Plane(n=Vector3D(0, 0, 1), o=Point3D(0, 0, 3))
    plane_4 = Plane(n=Vector3D(0, 0, 1), o=Point3D(0, 0, -2))
    assert len(pline.intersect_plane(plane_3)) == 0
    assert len(pline.intersect_plane(plane_4)) == 0


def test_split_with_plane():
    """Test the Polyline3D split_with_plane method."""
    pts = (Point3D(0, 0, 1), Point3D(2, 0, -1), Point3D(2, 2, 1), Point3D(0, 2, -1))
    pline = Polyline3D(pts)

    plane_1 = Plane(n=Vector3D(0, 0, 1), o=Point3D(0, 0, 0))
    splits = pline.split_with_plane(plane_1)
    assert len(splits) == 4
    assert isinstance(splits[0], LineSegment3D)
    assert isinstance(splits[1], Polyline3D)

    plane_2 = Plane(n=Vector3D(0, 1, 0), o=Point3D(1, 1, 0))
    assert len(pline.split_with_plane(plane_2)) == 2

    plane_3 = Plane(n=Vector3D(0, 0, 1), o=Point3D(0, 0, 3))
    plane_4 = Plane(n=Vector3D(0, 0, 1), o=Point3D(0, 0, -2))
    assert len(pline.split_with_plane(plane_3)) == 1
    assert len(pline.split_with_plane(plane_4)) == 1


def test_join_segments():
    """Test the join_segments method."""
    pts = (Point3D(0, 0), Point3D(2, 0), Point3D(2, 2), Point3D(0, 2))
    l_segs = (LineSegment3D.from_end_points(pts[0], pts[1]),
            LineSegment3D.from_end_points(pts[1], pts[2]),
            LineSegment3D.from_end_points(pts[2], pts[3]),
            LineSegment3D.from_end_points(pts[3], pts[0]))
    p_lines = Polyline3D.join_segments(l_segs, 0.01)
    assert len(p_lines) == 1
    assert isinstance(p_lines[0], Polyline3D)
    assert len(p_lines[0]) == 5
    assert p_lines[0].is_closed(0.01)

    l_segs = (LineSegment3D.from_end_points(pts[0], pts[1]),
            LineSegment3D.from_end_points(pts[2], pts[3]),
            LineSegment3D.from_end_points(pts[1], pts[2]),
            LineSegment3D.from_end_points(pts[3], pts[0]))
    p_lines = Polyline3D.join_segments(l_segs, 0.01)
    assert len(p_lines) == 1
    assert isinstance(p_lines[0], Polyline3D)
    assert len(p_lines[0]) == 5
    assert p_lines[0].is_closed(0.01)

    l_segs = (LineSegment3D.from_end_points(pts[0], pts[1]),
            LineSegment3D.from_end_points(pts[1], pts[2]),
            LineSegment3D.from_end_points(pts[0], pts[3]),
            LineSegment3D.from_end_points(pts[3], pts[2]))
    p_lines = Polyline3D.join_segments(l_segs, 0.01)
    assert len(p_lines) == 1
    assert isinstance(p_lines[0], Polyline3D)
    assert len(p_lines[0]) == 5
    assert p_lines[0].is_closed(0.01)


def test_join_segments_multiple_pline():
    """Test the join_segments method with multiple polylines."""
    pts = (Point3D(0, 0), Point3D(2, 0), Point3D(2, 2), Point3D(0, 2))
    extra_pts = (Point3D(3, 3), Point3D(4, 3), Point3D(4, 4), Point3D(3, 4))
    l_segs = (LineSegment3D.from_end_points(extra_pts[0], extra_pts[1]),
              LineSegment3D.from_end_points(pts[0], pts[1]),
              LineSegment3D.from_end_points(pts[1], pts[2]),
              LineSegment3D.from_end_points(pts[0], pts[3]),
              LineSegment3D.from_end_points(pts[3], pts[2]))
    p_lines = Polyline3D.join_segments(l_segs, 0.01)
    assert len(p_lines) == 2

    l_segs = (LineSegment3D.from_end_points(extra_pts[0], extra_pts[1]),
              LineSegment3D.from_end_points(pts[0], pts[1]),
              LineSegment3D.from_end_points(pts[1], pts[2]),
              LineSegment3D.from_end_points(pts[0], pts[3]),
              LineSegment3D.from_end_points(pts[3], pts[2]),
              LineSegment3D.from_end_points(extra_pts[2], extra_pts[3]),
              LineSegment3D.from_end_points(extra_pts[1], extra_pts[2]),
              LineSegment3D.from_end_points(extra_pts[0], extra_pts[3]))
    p_lines = Polyline3D.join_segments(l_segs, 0.01)
    assert len(p_lines) == 2
    for p_line in p_lines:
        assert isinstance(p_line, Polyline3D)
        assert len(p_line) == 5
        assert p_line.is_closed(0.01)


def test_join_segments_disconnected():
    """Test the join_segments method with diconnected polylines."""
    pts = (Point3D(0, 0), Point3D(2, 0), Point3D(2, 2), Point3D(0, 2))
    extra_pts = (Point3D(3, 3), Point3D(4, 3), Point3D(4, 4), Point3D(3, 4))

    l_segs = (LineSegment3D.from_end_points(extra_pts[0], extra_pts[1]),
              LineSegment3D.from_end_points(pts[0], pts[1]),
              LineSegment3D.from_end_points(pts[2], pts[3]),
              LineSegment3D.from_end_points(extra_pts[3], extra_pts[2]))
    p_lines = Polyline3D.join_segments(l_segs, 0.01)
    assert len(p_lines) == 4
