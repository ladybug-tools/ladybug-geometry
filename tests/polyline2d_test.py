# coding=utf-8
import pytest

from ladybug_geometry.geometry2d.polyline import Polyline2D
from ladybug_geometry.geometry2d.polygon import Polygon2D
from ladybug_geometry.geometry2d.pointvector import Point2D, Vector2D
from ladybug_geometry.geometry2d.line import LineSegment2D
from ladybug_geometry.geometry2d.ray import Ray2D

import math


def test_polyline2d_init():
    """Test the initialization of Polyline2D objects and basic properties."""
    pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    pline = Polyline2D(pts)

    str(pline)  # test the string representation of the polyline

    assert isinstance(pline.vertices, tuple)
    assert len(pline.vertices) == 4
    assert len(pline) == 4
    for point in pline:
        assert isinstance(point, Point2D)

    assert isinstance(pline.segments, tuple)
    assert len(pline.segments) == 3
    for seg in pline.segments:
        assert isinstance(seg, LineSegment2D)
        assert seg.length == 2

    assert pline.p1 == pts[0]
    assert pline.p2 == pts[-1]

    assert pline.length == 6
    assert pline.is_self_intersecting is False

    assert pline.vertices[0] == pline[0]

    p_array = pline.to_array()
    assert isinstance(p_array, tuple)
    assert len(p_array) == 4
    for arr in p_array:
        assert isinstance(p_array, tuple)
        assert len(arr) == 2
    pline_2 = Polyline2D.from_array(p_array)
    assert pline == pline_2


def test_equality():
    """Test the equality of Polyline2D objects."""
    pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    pts_2 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0.1, 2))
    pline = Polyline2D(pts)
    pline_dup = pline.duplicate()
    pline_alt = Polyline2D(pts_2)

    assert pline is pline
    assert pline is not pline_dup
    assert pline == pline_dup
    assert hash(pline) == hash(pline_dup)
    assert pline != pline_alt
    assert hash(pline) != hash(pline_alt)


def test_to_polygon():
    """Test the to_polygon method."""
    pts_1 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    pts_2 = pts_1 + (Point2D(0, 0),)

    pline_1 = Polyline2D(pts_1)
    pline_2 = Polyline2D(pts_2)

    polygon_1 = pline_1.to_polygon(0.01)
    polygon_2 = pline_2.to_polygon(0.01)

    assert isinstance(polygon_1, Polygon2D)
    assert isinstance(polygon_2, Polygon2D)
    assert len(polygon_1) == 4
    assert len(polygon_2) == 4


def test_from_polygon():
    """Test the from_polygon method."""
    pts_1 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    pgon = Polygon2D(pts_1)

    pline = Polyline2D.from_polygon(pgon)
    assert isinstance(pline, Polyline2D)
    assert len(pline) == 5


def test_polyline2d_to_from_dict():
    """Test the to/from dict of Polyline2D objects."""
    pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    pline = Polyline2D(pts)
    pline_dict = pline.to_dict()
    new_pline = Polyline2D.from_dict(pline_dict)
    assert isinstance(new_pline, Polyline2D)
    assert new_pline.to_dict() == pline_dict


def test_is_self_intersecting():
    """Test the is_self_intersecting property."""
    pts_1 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    pline_1 = Polyline2D(pts_1)
    pts_2 = (Point2D(0, 0), Point2D(0, 2), Point2D(2, 0), Point2D(2, 2),
             Point2D(0.5, 0.5))
    pline_2 = Polyline2D(pts_2)

    assert not pline_1.is_self_intersecting
    assert pline_2.is_self_intersecting


def test_min_max_center():
    """Test the Polyline2D min, max and center."""
    pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    pline = Polyline2D(pts)

    assert pline.min == Point2D(0, 0)
    assert pline.max == Point2D(2, 2)
    assert pline.center == Point2D(1, 1)


def test_remove_colinear_vertices():
    """Test the remove_colinear_vertices method of Polyline2D."""
    pts_1 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    pts_2 = (Point2D(0, 0), Point2D(1, 0), Point2D(2, 0), Point2D(2, 2),
             Point2D(0, 2))
    pline_1 = Polyline2D(pts_1)
    pline_2 = Polyline2D(pts_2)

    assert len(pline_1.remove_colinear_vertices(0.0001).vertices) == 4
    assert len(pline_2.remove_colinear_vertices(0.0001).vertices) == 4


def test_pline2d_duplicate():
    """Test the duplicate method of Polyline2D."""
    pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    pline = Polyline2D(pts)
    new_pline = pline.duplicate()

    for i, pt in enumerate(new_pline):
        assert pt == pts[i]

    assert pline.length == new_pline.length
    assert pline.vertices == new_pline.vertices
    assert pline.is_self_intersecting == new_pline.is_self_intersecting


def test_reverse():
    """Test the reverse method."""
    pts_1 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    pline = Polyline2D(pts_1)
    new_pline = pline.reverse()

    assert pline.length == new_pline.length
    assert pline.vertices == tuple(reversed(new_pline.vertices))
    assert pline.is_self_intersecting == new_pline.is_self_intersecting


def test_move():
    """Test the Polyline2D move method."""
    pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    pline = Polyline2D(pts)

    vec_1 = Vector2D(2, 2)
    new_pline = pline.move(vec_1)
    assert new_pline[0] == Point2D(2, 2)
    assert new_pline[1] == Point2D(4, 2)
    assert new_pline[2] == Point2D(4, 4)
    assert new_pline[3] == Point2D(2, 4)

    assert pline.length == new_pline.length
    assert pline.is_self_intersecting is new_pline.is_self_intersecting


def test_scale():
    """Test the Polyline2D scale method."""
    pts_1 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    pline_1 = Polyline2D(pts_1)
    pts_2 = (Point2D(1, 1), Point2D(2, 1), Point2D(2, 2), Point2D(1, 2))
    pline_2 = Polyline2D(pts_2)
    origin_1 = Point2D(2, 0)
    origin_2 = Point2D(1, 1)

    new_pline_1 = pline_1.scale(2, origin_1)
    assert new_pline_1[0] == Point2D(-2, 0)
    assert new_pline_1[1] == Point2D(2, 0)
    assert new_pline_1[2] == Point2D(2, 4)
    assert new_pline_1[3] == Point2D(-2, 4)
    assert new_pline_1.length == pline_1.length * 2

    new_pline_2 = pline_2.scale(2, origin_2)
    assert new_pline_2[0] == Point2D(1, 1)
    assert new_pline_2[1] == Point2D(3, 1)
    assert new_pline_2[2] == Point2D(3, 3)
    assert new_pline_2[3] == Point2D(1, 3)
    assert new_pline_2.length == pline_2.length * 2


def test_scale_world_origin():
    """Test the Polyline2D scale method with None origin."""
    pts = (Point2D(1, 1), Point2D(2, 1), Point2D(2, 2), Point2D(1, 2))
    pline = Polyline2D(pts)

    new_pline = pline.scale(2)
    assert new_pline[0] == Point2D(2, 2)
    assert new_pline[1] == Point2D(4, 2)
    assert new_pline[2] == Point2D(4, 4)
    assert new_pline[3] == Point2D(2, 4)
    assert new_pline.length == pline.length * 2


def test_rotate():
    """Test the Polyline2D rotate method."""
    pts = (Point2D(1, 1), Point2D(2, 1), Point2D(2, 2), Point2D(1, 2))
    pline = Polyline2D(pts)
    origin_1 = Point2D(1, 1)

    test_1 = pline.rotate(math.pi, origin_1)
    assert test_1[0].x == pytest.approx(1, rel=1e-3)
    assert test_1[0].y == pytest.approx(1, rel=1e-3)
    assert test_1[2].x == pytest.approx(0, rel=1e-3)
    assert test_1[2].y == pytest.approx(0, rel=1e-3)
    assert pline.length == pytest.approx(test_1.length, rel=1e-3)

    test_2 = pline.rotate(math.pi/2, origin_1)
    assert test_2[0].x == pytest.approx(1, rel=1e-3)
    assert test_2[0].y == pytest.approx(1, rel=1e-3)
    assert test_2[2].x == pytest.approx(0, rel=1e-3)
    assert test_2[2].y == pytest.approx(2, rel=1e-3)


def test_reflect():
    """Test the Polyline2D reflect method."""
    pts = (Point2D(1, 1), Point2D(2, 1), Point2D(2, 2), Point2D(1, 2))
    pline = Polyline2D(pts)

    origin_1 = Point2D(1, 0)
    normal_1 = Vector2D(1, 0)
    normal_2 = Vector2D(-1, -1).normalize()

    test_1 = pline.reflect(normal_1, origin_1)
    assert test_1[0].x == pytest.approx(1, rel=1e-3)
    assert test_1[0].y == pytest.approx(1, rel=1e-3)
    assert test_1[2].x == pytest.approx(0, rel=1e-3)
    assert test_1[2].y == pytest.approx(2, rel=1e-3)
    assert pline.length == pytest.approx(test_1.length, rel=1e-3)


def test_intersect_line_ray():
    """Test the Polyline2D intersect_line_ray method."""
    pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    pline = Polyline2D(pts)

    ray_1 = Ray2D(Point2D(1, -1), Vector2D(0, 1))
    ray_2 = Ray2D(Point2D(1, 1), Vector2D(1, 0))
    ray_3 = Ray2D(Point2D(1, 1), Vector2D(11, 0))
    ray_4 = Ray2D(Point2D(-1, 1), Vector2D(-1, 0))
    ray_5 = Ray2D(Point2D(-1, 1), Vector2D(1, 0))

    assert len(pline.intersect_line_ray(ray_1)) == 2
    assert len(pline.intersect_line_ray(ray_2)) == 1
    assert len(pline.intersect_line_ray(ray_3)) == 1
    assert len(pline.intersect_line_ray(ray_4)) == 0
    assert len(pline.intersect_line_ray(ray_5)) == 1

    line_1 = LineSegment2D(Point2D(-1, 1), Vector2D(0.5, 0))
    line_2 = LineSegment2D(Point2D(1, -1), Vector2D(0, 2))
    line_3 = LineSegment2D(Point2D(1, -1), Vector2D(0, 3))

    assert len(pline.intersect_line_ray(line_1)) == 0
    assert len(pline.intersect_line_ray(line_2)) == 1
    assert len(pline.intersect_line_ray(line_3)) == 2


def test_intersect_line_infinite():
    """Test the Polyline2D intersect_line_infinite method."""
    pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    pline = Polyline2D(pts)

    ray_1 = Ray2D(Point2D(-1, 1), Vector2D(1, 0))
    ray_2 = Ray2D(Point2D(1, 1), Vector2D(1, 0))
    ray_3 = Ray2D(Point2D(1, 1), Vector2D(11, 0))
    ray_4 = Ray2D(Point2D(-1, 1), Vector2D(-1, 0))
    ray_5 = Ray2D(Point2D(-1, 3), Vector2D(-1, 0))
    ray_6 = Ray2D(Point2D(0, 2), Vector2D(-1, -1))

    assert len(pline.intersect_line_infinite(ray_1)) == 1
    assert len(pline.intersect_line_infinite(ray_2)) == 1
    assert len(pline.intersect_line_infinite(ray_3)) == 1
    assert len(pline.intersect_line_infinite(ray_4)) == 1
    assert len(pline.intersect_line_infinite(ray_5)) == 0
    assert len(pline.intersect_line_infinite(ray_6)) > 0

def test_join_segments():
    """Test the join_segments method."""
    pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    l_segs = (LineSegment2D.from_end_points(pts[0], pts[1]),
            LineSegment2D.from_end_points(pts[1], pts[2]),
            LineSegment2D.from_end_points(pts[2], pts[3]),
            LineSegment2D.from_end_points(pts[3], pts[0]))
    p_lines = Polyline2D.join_segments(l_segs, 0.01)
    assert len(p_lines) == 1
    assert isinstance(p_lines[0], Polyline2D)
    assert len(p_lines[0]) == 5
    assert p_lines[0].is_closed(0.01)

    l_segs = (LineSegment2D.from_end_points(pts[0], pts[1]),
            LineSegment2D.from_end_points(pts[2], pts[3]),
            LineSegment2D.from_end_points(pts[1], pts[2]),
            LineSegment2D.from_end_points(pts[3], pts[0]))
    p_lines = Polyline2D.join_segments(l_segs, 0.01)
    assert len(p_lines) == 1
    assert isinstance(p_lines[0], Polyline2D)
    assert len(p_lines[0]) == 5
    assert p_lines[0].is_closed(0.01)

    l_segs = (LineSegment2D.from_end_points(pts[0], pts[1]),
            LineSegment2D.from_end_points(pts[1], pts[2]),
            LineSegment2D.from_end_points(pts[0], pts[3]),
            LineSegment2D.from_end_points(pts[3], pts[2]))
    p_lines = Polyline2D.join_segments(l_segs, 0.01)
    assert len(p_lines) == 1
    assert isinstance(p_lines[0], Polyline2D)
    assert len(p_lines[0]) == 5
    assert p_lines[0].is_closed(0.01)


def test_join_segments_multiple_pline():
    """Test the join_segments method with multiple polylines."""
    pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    extra_pts = (Point2D(3, 3), Point2D(4, 3), Point2D(4, 4), Point2D(3, 4))
    l_segs = (LineSegment2D.from_end_points(extra_pts[0], extra_pts[1]),
              LineSegment2D.from_end_points(pts[0], pts[1]),
              LineSegment2D.from_end_points(pts[1], pts[2]),
              LineSegment2D.from_end_points(pts[0], pts[3]),
              LineSegment2D.from_end_points(pts[3], pts[2]))
    p_lines = Polyline2D.join_segments(l_segs, 0.01)
    assert len(p_lines) == 2

    l_segs = (LineSegment2D.from_end_points(extra_pts[0], extra_pts[1]),
              LineSegment2D.from_end_points(pts[0], pts[1]),
              LineSegment2D.from_end_points(pts[1], pts[2]),
              LineSegment2D.from_end_points(pts[0], pts[3]),
              LineSegment2D.from_end_points(pts[3], pts[2]),
              LineSegment2D.from_end_points(extra_pts[2], extra_pts[3]),
              LineSegment2D.from_end_points(extra_pts[1], extra_pts[2]),
              LineSegment2D.from_end_points(extra_pts[0], extra_pts[3]))
    p_lines = Polyline2D.join_segments(l_segs, 0.01)
    assert len(p_lines) == 2
    for p_line in p_lines:
        assert isinstance(p_line, Polyline2D)
        assert len(p_line) == 5
        assert p_line.is_closed(0.01)


def test_join_segments_disconnected():
    """Test the join_segments method with diconnected polylines."""
    pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    extra_pts = (Point2D(3, 3), Point2D(4, 3), Point2D(4, 4), Point2D(3, 4))

    l_segs = (LineSegment2D.from_end_points(extra_pts[0], extra_pts[1]),
              LineSegment2D.from_end_points(pts[0], pts[1]),
              LineSegment2D.from_end_points(pts[2], pts[3]),
              LineSegment2D.from_end_points(extra_pts[3], extra_pts[2]))
    p_lines = Polyline2D.join_segments(l_segs, 0.01)
    assert len(p_lines) == 4
