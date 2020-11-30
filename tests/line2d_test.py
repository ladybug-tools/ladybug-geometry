# coding=utf-8
import pytest

from ladybug_geometry.geometry2d.pointvector import Point2D, Vector2D
from ladybug_geometry.geometry2d.line import LineSegment2D

import math


def test_linesegment2_init():
    """Test the initialization of LineSegment2D objects and basic properties."""
    pt = Point2D(2, 0)
    vec = Vector2D(0, 2)
    seg = LineSegment2D(pt, vec)
    str(seg)  # test the string representation of the line segment

    assert seg.p == Point2D(2, 0)
    assert seg.v == Vector2D(0, 2)
    assert seg.p1 == Point2D(2, 0)
    assert seg.p2 == Point2D(2, 2)
    assert seg.midpoint == Point2D(2, 1)
    assert seg.point_at(0.25) == Point2D(2, 0.5)
    assert seg.point_at_length(1) == Point2D(2, 1)
    assert seg.length == 2
    assert len(seg.vertices) == 2

    flip_seg = seg.flip()
    assert flip_seg.p == Point2D(2, 2)
    assert flip_seg.v == Vector2D(0, -2)


def test_linesegment2_to_from_dict():
    """Test the to/from dict of LineSegment2D objects."""
    pt = Point2D(2, 0)
    vec = Vector2D(0, 2)
    seg = LineSegment2D(pt, vec)
    seg_dict = seg.to_dict()
    new_seg = LineSegment2D.from_dict(seg_dict)
    assert isinstance(new_seg, LineSegment2D)
    assert new_seg.to_dict() == seg_dict


def test_init_from_endpoints():
    """Test the initalization of LineSegement2D from end points."""
    pt_1 = Point2D(2, 0)
    pt_2 = Point2D(2, 2)
    seg = LineSegment2D.from_end_points(pt_1, pt_2)

    assert seg.p == Point2D(2, 0)
    assert seg.v == Vector2D(0, 2)
    assert seg.p1 == Point2D(2, 0)
    assert seg.p2 == Point2D(2, 2)
    assert seg.length == 2


def test_init_from_sdl():
    """Test the initalization of LineSegement2D from start, direction, length."""
    pt = Point2D(2, 0)
    vec = Vector2D(0, 1)
    seg = LineSegment2D.from_sdl(pt, vec, 2)

    assert seg.p == Point2D(2, 0)
    assert seg.v == Vector2D(0, 2)
    assert seg.p1 == Point2D(2, 0)
    assert seg.p2 == Point2D(2, 2)
    assert seg.length == 2


def test_linesegment2_immutability():
    """Test the immutability of LineSegement2D objects."""
    pt = Point2D(2, 0)
    vec = Vector2D(0, 2)
    seg = LineSegment2D(pt, vec)

    assert isinstance(seg, LineSegment2D)
    with pytest.raises(AttributeError):
        seg.p.x = 3
    with pytest.raises(AttributeError):
        seg.v.x = 3
    with pytest.raises(AttributeError):
        seg.p = Point2D(0, 0)
    with pytest.raises(AttributeError):
        seg.v = Vector2D(2, 2)

    seg_copy = seg.duplicate()
    assert seg.p == seg_copy.p
    assert seg.v == seg_copy.v


def test_equality():
    """Test the equality of LineSegement2D objects."""
    pt = Point2D(2, 0)
    vec = Vector2D(0, 2)
    seg = LineSegment2D(pt, vec)
    seg_dup = seg.duplicate()
    seg_alt = LineSegment2D(Point2D(2, 0.1), vec)

    assert seg is seg
    assert seg is not seg_dup
    assert seg == seg_dup
    assert hash(seg) == hash(seg_dup)
    assert seg != seg_alt
    assert hash(seg) != hash(seg_alt)


def test_equivalent():
    """Testing the equivalence of LineSegment2D objects."""
    seg1 = LineSegment2D(
        Point2D(0.5, 0.5), Vector2D(1.5, 2.5))
    # LineSegment2D (<0.50, 0.50> to <2.00, 3.00>)

    # Test equal seg1, same order
    seg2 = LineSegment2D(
        Point2D(0.5, 0.5), Vector2D(1.5, 2.5))
    assert seg1.is_equivalent(seg2, 1e-10)

    # Test equal, diff order
    seg2 = LineSegment2D(
        Point2D(2.00, 3.00), Vector2D(-1.5, -2.5))
    assert seg1.is_equivalent(seg2, 1e-10)

    # Test not equal first point
    seg2 = LineSegment2D(
        Point2D(2.001, 3.00), Vector2D(-1.5, -2.5))
    assert not seg1.is_equivalent(seg2, 1e-10)

     # Test not equal second point
    seg2 = LineSegment2D(
        Point2D(0.5001, 0.5), Vector2D(1.5, 2.5))
    assert not seg1.is_equivalent(seg2, 1e-10)


def test_move():
    """Test the LineSegement2D move method."""
    pt = Point2D(2, 0)
    vec = Vector2D(0, 2)
    seg = LineSegment2D(pt, vec)

    vec_1 = Vector2D(2, 2)
    new_seg = seg.move(vec_1)
    assert new_seg.p == Point2D(4, 2)
    assert new_seg.v == vec
    assert new_seg.p1 == Point2D(4, 2)
    assert new_seg.p2 == Point2D(4, 4)


def test_scale():
    """Test the LineSegement2D scale method."""
    pt = Point2D(2, 2)
    vec = Vector2D(0, 2)
    seg = LineSegment2D(pt, vec)

    origin_1 = Point2D(0, 2)
    origin_2 = Point2D(1, 1)
    new_seg = seg.scale(2, origin_1)
    assert new_seg.p == Point2D(4, 2)
    assert new_seg.v == Point2D(0, 4)
    assert new_seg.length == 4

    new_seg = seg.scale(2, origin_2)
    assert new_seg.p == Point2D(3, 3)
    assert new_seg.v == Point2D(0, 4)


def test_scale_world_origin():
    """Test the LineSegement2D scale method with None origin."""
    pt = Point2D(2, 2)
    vec = Vector2D(0, 2)
    seg = LineSegment2D(pt, vec)

    new_seg = seg.scale(2)
    assert new_seg.p == Point2D(4, 4)
    assert new_seg.v == Point2D(0, 4)
    assert new_seg.length == 4


def test_rotate():
    """Test the LineSegement2D rotate method."""
    pt = Point2D(2, 2)
    vec = Vector2D(0, 2)
    seg = LineSegment2D(pt, vec)
    origin_1 = Point2D(0, 2)

    test_1 = seg.rotate(math.pi, origin_1)
    assert test_1.p.x == pytest.approx(-2, rel=1e-3)
    assert test_1.p.y == pytest.approx(2, rel=1e-3)
    assert test_1.v.x == pytest.approx(0, rel=1e-3)
    assert test_1.v.y == pytest.approx(-2, rel=1e-3)

    test_2 = seg.rotate(math.pi/2, origin_1)
    assert test_2.p.x == pytest.approx(0, rel=1e-3)
    assert test_2.p.y == pytest.approx(4, rel=1e-3)
    assert test_2.v.x == pytest.approx(-2, rel=1e-3)
    assert test_2.v.y == pytest.approx(0, rel=1e-3)


def test_reflect():
    """Test the LineSegement2D reflect method."""
    pt = Point2D(2, 2)
    vec = Vector2D(0, 2)
    seg = LineSegment2D(pt, vec)

    origin_1 = Point2D(0, 1)
    origin_2 = Point2D(1, 1)
    normal_1 = Vector2D(0, 1)
    normal_2 = Vector2D(-1, 1).normalize()

    assert seg.reflect(normal_1, origin_1).p == Point2D(2, 0)
    assert seg.reflect(normal_1, origin_1).v == Vector2D(0, -2)
    assert seg.reflect(normal_1, origin_2).p == Point2D(2, 0)
    assert seg.reflect(normal_1, origin_2).v == Vector2D(0, -2)

    test_1 = seg.reflect(normal_2, origin_2)
    assert test_1.p == Point2D(2, 2)
    assert test_1.v.x == pytest.approx(2, rel=1e-3)
    assert test_1.v.y == pytest.approx(0, rel=1e-3)

    test_2 = seg.reflect(normal_2, origin_1)
    assert test_2.p.x == pytest.approx(1, rel=1e-3)
    assert test_2.p.y == pytest.approx(3, rel=1e-3)
    assert test_1.v.x == pytest.approx(2, rel=1e-3)
    assert test_1.v.y == pytest.approx(0, rel=1e-3)


def test_subdivide():
    """Test the LineSegement2D subdivide methods."""
    pt = Point2D(2, 2)
    vec = Vector2D(0, 2)
    seg = LineSegment2D(pt, vec)

    divisions = seg.subdivide(0.5)
    assert len(divisions) == 5
    assert divisions[0] == pt
    assert divisions[1] == Point2D(2, 2.5)
    assert divisions[2] == Point2D(2, 3)
    assert divisions[3] == Point2D(2, 3.5)
    assert divisions[4] == Point2D(2, 4)

    divisions = seg.subdivide([1, 0.5, 0.25])
    assert len(divisions) == 5
    assert divisions[0] == pt
    assert divisions[1] == Point2D(2, 3)
    assert divisions[2] == Point2D(2, 3.5)
    assert divisions[3] == Point2D(2, 3.75)
    assert divisions[4] == Point2D(2, 4)

    divisions = seg.subdivide_evenly(4)
    assert len(divisions) == 5
    assert divisions[0] == pt
    assert divisions[1] == Point2D(2, 2.5)
    assert divisions[2] == Point2D(2, 3)
    assert divisions[3] == Point2D(2, 3.5)
    assert divisions[4] == Point2D(2, 4)


def test_closest_point():
    """Test the LineSegement2D closest_point method."""
    pt = Point2D(2, 2)
    vec = Vector2D(0, 2)
    seg = LineSegment2D(pt, vec)

    near_pt = Point2D(3, 3)
    assert seg.closest_point(near_pt) == Point2D(2, 3)
    near_pt = Point2D(2, 0)
    assert seg.closest_point(near_pt) == Point2D(2, 2)
    near_pt = Point2D(2, 5)
    assert seg.closest_point(near_pt) == Point2D(2, 4)


def test_distance_to_point():
    """Test the LineSegement2D distance_to_point method."""
    pt = Point2D(2, 2)
    vec = Vector2D(0, 2)
    seg = LineSegment2D(pt, vec)

    near_pt = Point2D(3, 3)
    assert seg.distance_to_point(near_pt) == 1
    near_pt = Point2D(2, 0)
    assert seg.distance_to_point(near_pt) == 2
    near_pt = Point2D(2, 5)
    assert seg.distance_to_point(near_pt) == 1


def test_intersect_line_ray():
    """Test the LineSegement2D distance_to_point method."""
    pt_1 = Point2D(2, 2)
    vec_1 = Vector2D(0, 2)
    seg_1 = LineSegment2D(pt_1, vec_1)

    pt_2 = Point2D(0, 3)
    vec_2 = Vector2D(4, 0)
    seg_2 = LineSegment2D(pt_2, vec_2)

    pt_3 = Point2D(0, 0)
    vec_3 = Vector2D(1, 1)
    seg_3 = LineSegment2D(pt_3, vec_3)

    assert seg_1.intersect_line_ray(seg_2) == Point2D(2, 3)
    assert seg_1.intersect_line_ray(seg_3) is None


def test_closest_points_between_line():
    """Test the LineSegement2D distance_to_point method."""
    pt_1 = Point2D(2, 2)
    vec_1 = Vector2D(0, 2)
    seg_1 = LineSegment2D(pt_1, vec_1)

    pt_2 = Point2D(0, 3)
    vec_2 = Vector2D(1, 0)
    seg_2 = LineSegment2D(pt_2, vec_2)

    pt_3 = Point2D(0, 0)
    vec_3 = Vector2D(1, 1)
    seg_3 = LineSegment2D(pt_3, vec_3)

    assert seg_1.closest_points_between_line(seg_2) == (Point2D(2, 3), Point2D(1, 3))
    assert seg_1.closest_points_between_line(seg_3) == (Point2D(2, 2), Point2D(1, 1))

    assert seg_1.distance_to_line(seg_2) == 1
    assert seg_1.distance_to_line(seg_3) == pytest.approx(1.41421, rel=1e-3)


def test_to_from_array():
    """Test to/from array method"""
    test_line = LineSegment2D.from_end_points(Point2D(2, 0), Point2D(2, 2))
    line_array = ((2, 0), (2, 2))

    assert test_line == LineSegment2D.from_array(line_array)

    line_array = ((2, 0), (2, 2))
    test_line = LineSegment2D.from_end_points(Point2D(2, 0), Point2D(2, 2))

    assert test_line.to_array() == line_array

    test_line_2 = LineSegment2D.from_array(test_line.to_array())
    assert test_line == test_line_2
