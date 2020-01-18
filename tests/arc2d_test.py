# coding=utf-8
import pytest

from ladybug_geometry.geometry2d.pointvector import Point2D, Vector2D
from ladybug_geometry.geometry2d.line import LineSegment2D
from ladybug_geometry.geometry2d.ray import Ray2D
from ladybug_geometry.geometry2d.arc import Arc2D

import math


def test_arc2_init():
    """Test the initalization of Arc2D objects and basic properties."""
    pt = Point2D(2, 0)
    arc = Arc2D(pt, 1, 0, math.pi)
    str(arc)  # test the string representation of the arc

    assert arc.c == pt
    assert arc.r == 1
    assert arc.a1 == 0
    assert arc.a2 == math.pi
    assert arc.p1 == Point2D(3, 0)
    assert arc.p2.x == pytest.approx(1, rel=1e-3)
    assert arc.p2.y == pytest.approx(0, rel=1e-3)
    assert arc.midpoint == Point2D(2, 1)
    assert arc.length == pytest.approx(math.pi, rel=1e-3)
    assert arc.angle == pytest.approx(math.pi, rel=1e-3)
    assert arc.is_circle is False
    assert arc.is_inverted is False

    arc2 = arc.duplicate()
    assert arc2.c == pt
    assert arc2.r == 1
    assert arc2.a1 == 0
    assert arc2.a2 == math.pi


def test_arc2_init_radius():
    """Test the initalization of Arc2D objects with a non-unit radius."""
    pt = Point2D(2, 2)
    arc = Arc2D(pt, 2,  math.pi, 0)

    assert arc.c == pt
    assert arc.r == 2
    assert arc.a1 == math.pi
    assert arc.a2 == 0
    assert arc.p1.x == pytest.approx(0, rel=1e-3)
    assert arc.p1.y == pytest.approx(2, rel=1e-3)
    assert arc.p2.x == pytest.approx(4, rel=1e-3)
    assert arc.p2.y == pytest.approx(2, rel=1e-3)
    assert arc.midpoint.x == pytest.approx(2, rel=1e-3)
    assert arc.midpoint.y == pytest.approx(0, rel=1e-3)
    assert arc.length == pytest.approx(2 * math.pi, rel=1e-3)
    assert arc.angle == pytest.approx(math.pi, rel=1e-3)
    assert arc.is_circle is False
    assert arc.is_inverted is True


def test_arc2_init_reveresed():
    """Test the initalization of Arc2D objects with reversed direction."""
    pt = Point2D(2, 0)
    arc = Arc2D(pt, 1, 1.5 * math.pi, 0.5 * math.pi)

    assert arc.c == pt
    assert arc.r == 1
    assert arc.a1 == 1.5 * math.pi
    assert arc.a2 == 0.5 * math.pi
    assert arc.p1.x == pytest.approx(2, rel=1e-3)
    assert arc.p1.y == pytest.approx(-1, rel=1e-3)
    assert arc.p2.x == pytest.approx(2, rel=1e-3)
    assert arc.p2.y == pytest.approx(1, rel=1e-3)
    assert arc.midpoint.x == pytest.approx(3, rel=1e-3)
    assert arc.midpoint.y == pytest.approx(0, rel=1e-3)
    assert arc.length == pytest.approx(math.pi, rel=1e-3)
    assert arc.angle == pytest.approx(math.pi, rel=1e-3)
    assert arc.is_circle is False
    assert arc.is_inverted is True


def test_arc2_init_circle():
    """Test the initalization of Arc2D objects as a circle."""
    pt = Point2D(2, 0)
    arc = Arc2D(pt, 3)

    assert arc.c == pt
    assert arc.r == 3
    assert arc.a1 == 0
    assert arc.a2 == 2 * math.pi
    assert arc.is_circle
    assert arc.length == math.pi * 6
    assert arc.area == math.pi * 3 ** 2


def test_arc2_init_from_start_mid_end():
    """Test the initalization of Arc2D objects from_start_mid_end."""
    p1 = Point2D(3, 0)
    m = Point2D(2, 1)
    p2 = Point2D(1, 0)
    arc = Arc2D.from_start_mid_end(p1, m, p2)

    assert arc.c == Point2D(2, 0)
    assert arc.r == 1
    assert arc.a1 == 0
    assert arc.a2 == math.pi
    assert arc.p1 == Point2D(3, 0)
    assert arc.p2.x == pytest.approx(1, rel=1e-3)
    assert arc.p2.y == pytest.approx(0, rel=1e-3)
    assert arc.midpoint == Point2D(2, 1)
    assert arc.length == pytest.approx(math.pi, rel=1e-3)
    assert arc.angle == pytest.approx(math.pi, rel=1e-3)
    assert arc.is_circle is False
    assert arc.is_inverted is False

    p1 = Point2D(0, 0)
    m = Point2D(0, 2)
    p2 = Point2D(-1, 1)
    arc = Arc2D.from_start_mid_end(p1, m, p2)

    assert arc.c == Point2D(0, 1)
    assert arc.r == 1
    assert arc.length == pytest.approx(1.5 * math.pi, rel=1e-3)
    assert arc.is_inverted is True

    p1 = Point2D(0, 0)
    m = Point2D(1, 0)
    p2 = Point2D(2, 0)
    with pytest.raises(Exception):
        arc = Arc2D.from_start_mid_end(p1, m, p2)


def test_arc2_to_from_dict():
    """Test the initalization of Arc2D objects and basic properties."""
    pt = Point2D(2, 0)
    arc = Arc2D(pt, 1, 0, math.pi)
    arc_dict = arc.to_dict()
    new_arc = Arc2D.from_dict(arc_dict)
    assert isinstance(new_arc, Arc2D)
    assert new_arc.to_dict() == arc_dict

    pt = Point2D(2, 0)
    arc = Arc2D(pt, 3)
    arc_dict = arc.to_dict()
    new_arc = Arc2D.from_dict(arc_dict)
    assert isinstance(new_arc, Arc2D)
    assert new_arc.is_circle
    assert new_arc.to_dict() == arc_dict


def test_equality():
    """Test the equality of Arc2D objects."""
    pt = Point2D(2, 0)
    arc = Arc2D(pt, 1, 0, math.pi)
    arc_dup = arc.duplicate()
    arc_alt = Arc2D(pt, 1, 0.1, math.pi)

    assert arc is arc
    assert arc is not arc_dup
    assert arc == arc_dup
    assert hash(arc) == hash(arc_dup)
    assert arc != arc_alt
    assert hash(arc) != hash(arc_alt)


def test_move():
    """Test the Arc2D move method."""
    pt = Point2D(2, 0)
    arc = Arc2D(pt, 1, 0, math.pi)

    vec_1 = Vector2D(2, 2)
    new_arc = arc.move(vec_1)
    assert new_arc.c == Point2D(4, 2)
    assert new_arc.r == arc.r
    assert new_arc.p1 == Point2D(5, 2)
    assert new_arc.p2 == Point2D(3, 2)
    assert new_arc.a1 == arc.a1
    assert new_arc.a2 == arc.a2


def test_scale():
    """Test the Arc2D scale method."""
    pt = Point2D(2, 0)
    arc = Arc2D(pt, 1, 0, math.pi)

    origin_1 = Point2D(2, 0)
    new_arc = arc.scale(2, origin_1)
    assert new_arc.c == Point2D(2, 0)
    assert new_arc.r == 2
    assert new_arc.length == 2 * arc.length
    assert new_arc.a1 == arc.a1
    assert new_arc.a2 == arc.a2


def test_scale_world_origin():
    """Test the Arc2D scale method with None origin."""
    pt = Point2D(2, 0)
    arc = Arc2D(pt, 1, 0, math.pi)

    new_arc = arc.scale(2)
    assert new_arc.c == Point2D(4, 0)
    assert new_arc.r == 2
    assert new_arc.length == 2 * arc.length
    assert new_arc.a1 == arc.a1
    assert new_arc.a2 == arc.a2


def test_rotate():
    """Test the Arc2D rotate method."""
    pt = Point2D(2, 2)
    arc = Arc2D(pt, 1, 0, math.pi)
    origin_1 = Point2D(0, 2)

    test_1 = arc.rotate(math.pi, origin_1)
    assert test_1.c.x == pytest.approx(-2, rel=1e-3)
    assert test_1.c.y == pytest.approx(2, rel=1e-3)
    assert test_1.r == arc.r

    test_2 = arc.rotate(math.pi/2, origin_1)
    assert test_2.c.x == pytest.approx(0, rel=1e-3)
    assert test_2.c.y == pytest.approx(4, rel=1e-3)
    assert test_2.r == arc.r


def test_reflect():
    """Test the Arc2D reflect method."""
    pt = Point2D(2, 2)
    arc = Arc2D(pt, 2, 0, math.pi)

    origin_1 = Point2D(0, 1)
    origin_2 = Point2D(1, 1)
    normal_1 = Vector2D(0, 1)
    normal_2 = Vector2D(-1, 1).normalize()

    assert arc.reflect(normal_1, origin_1).c.x == pytest.approx(2, rel=1e-3)
    assert arc.reflect(normal_1, origin_1).c.y == pytest.approx(0, rel=1e-3)
    assert arc.reflect(normal_1, origin_1).midpoint.x == pytest.approx(2, rel=1e-3)
    assert arc.reflect(normal_1, origin_1).midpoint.y == pytest.approx(-2, rel=1e-3)
    assert arc.reflect(normal_1, origin_2).c == Point2D(2, 0)
    assert arc.reflect(normal_1, origin_2).midpoint.x == pytest.approx(2, rel=1e-3)
    assert arc.reflect(normal_1, origin_2).midpoint.y == pytest.approx(-2, rel=1e-3)

    test_1 = arc.reflect(normal_2, origin_2)
    assert test_1.c.x == pytest.approx(2, rel=1e-3)
    assert test_1.c.y == pytest.approx(2, rel=1e-3)
    assert test_1.midpoint.x == pytest.approx(4, rel=1e-3)
    assert test_1.midpoint.y == pytest.approx(2, rel=1e-3)


def test_subdivide():
    """Test the Arc2D subdivide methods."""
    pt = Point2D(2, 0)
    arc = Arc2D(pt, 1, 1.5 * math.pi, 0.5 * math.pi)

    divisions = arc.subdivide(0.5)
    assert len(divisions) == 8
    divisions = arc.subdivide([1, 0.5, 0.25])
    assert len(divisions) == 10
    divisions = arc.subdivide_evenly(4)
    assert len(divisions) == 5


def test_point_at_angle():
    """Test the Arc2D point_at_angle method."""
    pt = Point2D(2, 0)
    arc = Arc2D(pt, 1, 1.5 * math.pi, 0.5 * math.pi)

    assert arc.point_at_angle(0.5 * math.pi).x == pytest.approx(3, rel=1e-2)
    assert arc.point_at_angle(0.5 * math.pi).y == pytest.approx(0, rel=1e-2)


def test_arc2_closest_point():
    """Test the Arc2D closest_point method."""
    pt = Point2D(2, 0)
    arc = Arc2D(pt, 1, 0, math.pi)

    assert arc.closest_point(Point2D(2, 2)) == Point2D(2, 1)
    assert arc.closest_point(Point2D(3, 1)).x == pytest.approx(2.71, rel=1e-2)
    assert arc.closest_point(Point2D(3, 1)).y == pytest.approx(0.71, rel=1e-2)
    assert arc.closest_point(Point2D(3, -1)) == Point2D(3, 0)
    assert arc.closest_point(Point2D(1, -1)).x == pytest.approx(1, rel=1e-2)
    assert arc.closest_point(Point2D(1, -1)).y == pytest.approx(0, rel=1e-2)

    assert arc.distance_to_point(Point2D(2, 2)) == 1
    assert arc.distance_to_point(Point2D(3, 1)) == pytest.approx(0.4142, rel=1e-2)
    assert arc.distance_to_point(Point2D(3, -1)) == pytest.approx(1, rel=1e-2)
    assert arc.distance_to_point(Point2D(1, -1)) == pytest.approx(1, rel=1e-2)


def test_arc2_intersect_line_ray():
    """Test the Arc2D intersect_line_ray method."""
    pt = Point2D(2, 0)
    arc = Arc2D(pt, 1, 0, math.pi)
    circle = Arc2D(pt, 1)
    seg1 = LineSegment2D(pt, Vector2D(2, 2))
    seg2 = LineSegment2D(Point2D(0, -2), Vector2D(6, 6))
    seg3 = LineSegment2D(pt, Vector2D(0.5, 0.5))

    int1 = arc.intersect_line_ray(seg1)
    assert len(int1) == 1
    assert int1[0].x == pytest.approx(2.71, rel=1e-2)
    assert int1[0].y == pytest.approx(0.71, rel=1e-2)
    int2 = circle.intersect_line_ray(seg1)
    assert len(int2) == 1
    assert int2[0].x == pytest.approx(2.71, rel=1e-2)
    assert int2[0].y == pytest.approx(0.71, rel=1e-2)
    int3 = arc.intersect_line_ray(seg2)
    assert len(int3) == 1
    assert int3[0].x == pytest.approx(2.71, rel=1e-2)
    assert int3[0].y == pytest.approx(0.71, rel=1e-2)
    int4 = circle.intersect_line_ray(seg2)
    assert len(int4) == 2
    assert int4[0].x == pytest.approx(2.71, rel=1e-2)
    assert int4[0].y == pytest.approx(0.71, rel=1e-2)
    assert int4[1].x == pytest.approx(1.29, rel=1e-2)
    assert int4[1].y == pytest.approx(-0.71, rel=1e-2)
    assert arc.intersect_line_ray(seg3) is None


def test_arc2_split_line_infinite():
    """Test the Arc2D split_line_ray_infinite method."""
    pt = Point2D(2, 0)
    arc = Arc2D(pt, 1, 0, math.pi)
    circle = Arc2D(pt, 1)

    int1 = circle.split_line_infinite(Ray2D(Point2D(), Vector2D(1, 0)))
    assert len(int1) == 2
    assert int1[0].p1 == Point2D(3, 0)
    assert int1[0].p2.x == pytest.approx(1, rel=1e-2)
    assert int1[0].p2.y == pytest.approx(0, rel=1e-2)
    assert int1[1].p2 == Point2D(3, 0)
    assert int1[1].p1.x == pytest.approx(1, rel=1e-2)
    assert int1[1].p1.y == pytest.approx(0, rel=1e-2)

    int2 = arc.split_line_infinite(Ray2D(Point2D(0, 0.5), Vector2D(1, 0)))
    assert len(int2) == 3
