# coding=utf-8
import pytest

from ladybug_geometry.geometry3d.arc import Arc3D
from ladybug_geometry.geometry3d.plane import Plane
from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
from ladybug_geometry.geometry3d.polyline import Polyline3D

from ladybug_geometry.geometry2d.arc import Arc2D

import math


def test_arc3_init():
    """Test the initialization of Arc3D objects and basic properties."""
    pt = Point3D(2, 0, 2)
    arc = Arc3D(Plane(o=pt), 1, 0, math.pi)
    str(arc)  # test the string representation of the arc

    assert isinstance(arc.arc2d, Arc2D)
    assert arc.c == pt
    assert arc.radius == arc.arc2d.r == 1
    assert arc.a1 == arc.arc2d.a1 == 0
    assert arc.a2 == arc.arc2d.a2 == math.pi
    assert arc.p1 == Point3D(3, 0, 2)
    assert arc.p2.x == pytest.approx(1, rel=1e-3)
    assert arc.p2.y == pytest.approx(0, rel=1e-3)
    assert arc.p2.z == pytest.approx(2, rel=1e-3)
    assert arc.midpoint == Point3D(2, 1, 2)
    assert arc.min.x == pytest.approx(1, rel=1e-3)
    assert arc.min.z == pytest.approx(2, rel=1e-3)
    assert arc.max == Point3D(3, 1, 2)
    assert arc.length == pytest.approx(math.pi, rel=1e-3)
    assert arc.angle == pytest.approx(math.pi, rel=1e-3)
    assert arc.is_circle is False
    assert arc.is_inverted is False

    arc2 = arc.duplicate()
    assert arc2.c == pt
    assert arc2.radius == 1
    assert arc2.a1 == 0
    assert arc2.a2 == math.pi


def test_arc3_init_radius():
    """Test the initialization of Arc3D objects with a non-unit radius."""
    pt = Point3D(2, 2, 2)
    n = Vector3D(0, 1, 0)
    arc = Arc3D(Plane(n=n, o=pt), 2, math.pi, 0)

    assert arc.c == pt
    assert arc.radius == arc.arc2d.r == 2
    assert arc.a1 == arc.arc2d.a1 == math.pi
    assert arc.a2 == arc.arc2d.a2 == 0
    assert arc.p1.x == pytest.approx(0, rel=1e-3)
    assert arc.p1.z == pytest.approx(2, rel=1e-3)
    assert arc.p2.x == pytest.approx(4, rel=1e-3)
    assert arc.p2.z == pytest.approx(2, rel=1e-3)
    assert arc.midpoint.x == pytest.approx(2, rel=1e-3)
    assert arc.midpoint.z == pytest.approx(4, rel=1e-3)
    assert arc.length == pytest.approx(2 * math.pi, rel=1e-3)
    assert arc.angle == pytest.approx(math.pi, rel=1e-3)
    assert arc.is_circle is False
    assert arc.is_inverted is True


def test_arc3_init_reversed():
    """Test the initialization of Arc3D objects with reversed direction."""
    pt = Point3D(2, 0, 2)
    arc = Arc3D(Plane(o=pt), 1, 1.5 * math.pi, 0.5 * math.pi)

    assert arc.c == pt
    assert arc.radius == 1
    assert arc.a1 == 1.5 * math.pi
    assert arc.a2 == 0.5 * math.pi
    assert arc.p1.x == pytest.approx(2, rel=1e-3)
    assert arc.p1.y == pytest.approx(-1, rel=1e-3)
    assert arc.p1.z == pytest.approx(2, rel=1e-3)
    assert arc.p2.x == pytest.approx(2, rel=1e-3)
    assert arc.p2.y == pytest.approx(1, rel=1e-3)
    assert arc.p2.z == pytest.approx(2, rel=1e-3)
    assert arc.midpoint.x == pytest.approx(3, rel=1e-3)
    assert arc.midpoint.y == pytest.approx(0, rel=1e-3)
    assert arc.min.x == pytest.approx(2, rel=1e-3)
    assert arc.max.x == pytest.approx(3, rel=1e-3)
    assert arc.max.y == pytest.approx(1, rel=1e-3)
    assert arc.length == pytest.approx(math.pi, rel=1e-3)
    assert arc.angle == pytest.approx(math.pi, rel=1e-3)
    assert arc.is_circle is False
    assert arc.is_inverted is True


def test_arc3_init_circle():
    """Test the initialization of Arc3D objects as a circle."""
    pt = Point3D(2, 0, 2)
    arc = Arc3D(Plane(o=pt), 3)

    assert arc.c == pt
    assert arc.radius == 3
    assert arc.a1 == 0
    assert arc.a2 == 2 * math.pi
    assert arc.is_circle
    assert arc.length == math.pi * 6
    assert arc.area == math.pi * 3 ** 2


def test_arc3_init_from_start_mid_end():
    """Test the initialization of Arc3D objects from_start_mid_end."""
    p1 = Point3D(3, 0, 2)
    m = Point3D(2, 1, 2)
    p2 = Point3D(1, 0, 2)
    arc = Arc3D.from_start_mid_end(p1, m, p2)

    assert arc.c == Point3D(2, 0, 2)
    assert arc.radius == 1
    assert arc.a1 == 0
    assert arc.a2 == math.pi
    assert arc.p1 == Point3D(3, 0, 2)
    assert arc.p2.x == pytest.approx(1, rel=1e-3)
    assert arc.p2.y == pytest.approx(0, rel=1e-3)
    assert arc.p2.z == pytest.approx(2, rel=1e-3)
    assert arc.midpoint == Point3D(2, 1, 2)
    assert arc.length == pytest.approx(math.pi, rel=1e-3)
    assert arc.angle == pytest.approx(math.pi, rel=1e-3)
    assert arc.is_circle is False
    assert arc.is_inverted is False

    p1 = Point3D(0, 0)
    m = Point3D(0, 2)
    p2 = Point3D(-1, 1)
    arc = Arc3D.from_start_mid_end(p1, m, p2)

    assert arc.c == Point3D(0, 1)
    assert arc.radius == 1
    assert arc.length == pytest.approx(1.5 * math.pi, rel=1e-3)
    assert arc.is_inverted is True

    p1 = Point3D(0, 0)
    m = Point3D(1, 0)
    p2 = Point3D(2, 0)
    with pytest.raises(Exception):
        arc = Arc3D.from_start_mid_end(p1, m, p2)


def test_arc3_init_from_start_mid_end_flipped():
    """Test the init of Arc3D objects from_start_mid_end using a flipped plane."""
    p1 = Point3D(-91.47, 40.40, 0.76)
    m = Point3D(0.06, 23.23, 97.27)
    p2 = Point3D(91.49, 40.35, 0.99)
    arc = Arc3D.from_start_mid_end(p1, m, p2)

    assert arc.midpoint.z > 0


def test_arc3_to_from_dict():
    """Test the initialization of Arc3D objects and basic properties."""
    pt = Point3D(2, 0, 2)
    arc = Arc3D(Plane(o=pt), 1, 0, math.pi)
    arc_dict = arc.to_dict()
    new_arc = Arc3D.from_dict(arc_dict)
    assert isinstance(new_arc, Arc3D)
    assert new_arc.to_dict() == arc_dict

    pt = Point3D(2, 0, 2)
    arc = Arc3D(Plane(o=pt), 3)
    arc_dict = arc.to_dict()
    new_arc = Arc3D.from_dict(arc_dict)
    assert isinstance(new_arc, Arc3D)
    assert new_arc.is_circle
    assert new_arc.to_dict() == arc_dict


def test_equality():
    """Test the equality of Arc3D objects."""
    pt = Point3D(2, 0, 2)
    arc = Arc3D(Plane(o=pt), 1, 0, math.pi)
    arc_dup = arc.duplicate()
    arc_alt = Arc3D(Plane(o=Point3D(2, 0.1, 2)), 1, 0.1, math.pi)

    assert arc is arc
    assert arc is not arc_dup
    assert arc == arc_dup
    assert hash(arc) == hash(arc_dup)
    assert arc != arc_alt
    assert hash(arc) != hash(arc_alt)


def test_move():
    """Test the Arc3D move method."""
    pt = Point3D(2, 0, 0)
    arc = Arc3D(Plane(o=pt), 1, 0, math.pi)

    vec_1 = Vector3D(2, 2, 2)
    new_arc = arc.move(vec_1)
    assert new_arc.c == Point3D(4, 2, 2)
    assert new_arc.radius == arc.radius
    assert new_arc.p1 == Point3D(5, 2, 2)
    assert new_arc.p2 == Point3D(3, 2, 2)
    assert new_arc.a1 == arc.a1
    assert new_arc.a2 == arc.a2


def test_scale():
    """Test the Arc3D scale method."""
    pt = Point3D(2, 0, 2)
    arc = Arc3D(Plane(o=pt), 1, 0, math.pi)

    origin_1 = Point3D(2, 0, 2)
    new_arc = arc.scale(2, origin_1)
    assert new_arc.c == Point3D(2, 0, 2)
    assert new_arc.radius == 2
    assert new_arc.length == 2 * arc.length
    assert new_arc.a1 == arc.a1
    assert new_arc.a2 == arc.a2


def test_scale_world_origin():
    """Test the Arc3D scale method with None origin."""
    pt = Point3D(2, 0, 2)
    arc = Arc3D(Plane(o=pt), 1, 0, math.pi)

    new_arc = arc.scale(2)
    assert new_arc.c == Point3D(4, 0, 4)
    assert new_arc.radius == 2
    assert new_arc.length == 2 * arc.length
    assert new_arc.a1 == arc.a1
    assert new_arc.a2 == arc.a2


def test_rotate():
    """Test the Arc3D rotate method."""
    pt = Point3D(2, 2, 2)
    arc = Arc3D(Plane(o=pt), 1, 0, math.pi)
    axis_1 = Vector3D(0, 1, 0)
    origin_1 = Point3D(0, 2, 2)

    test_1 = arc.rotate(axis_1, math.pi, origin_1)
    assert test_1.c.x == pytest.approx(-2, rel=1e-3)
    assert test_1.c.z == pytest.approx(2, rel=1e-3)
    assert test_1.radius == arc.radius

    test_2 = arc.rotate(axis_1, math.pi / 2, origin_1)
    assert test_2.c.x == pytest.approx(0, rel=1e-3)
    assert test_2.c.z == pytest.approx(0, rel=1e-3)
    assert test_2.radius == arc.radius


def test_rotate_xy():
    """Test the Arc3D rotate_xy method."""
    pt = Point3D(2, 2, 2)
    arc = Arc3D(Plane(o=pt), 1, 0, math.pi)
    origin_1 = Point3D(0, 2, 2)

    test_1 = arc.rotate_xy(math.pi, origin_1)
    assert test_1.c.x == pytest.approx(-2, rel=1e-3)
    assert test_1.c.y == pytest.approx(2, rel=1e-3)
    assert test_1.radius == arc.radius

    test_2 = arc.rotate_xy(math.pi / 2, origin_1)
    assert test_2.c.x == pytest.approx(0, rel=1e-3)
    assert test_2.c.y == pytest.approx(4, rel=1e-3)
    assert test_2.radius == arc.radius


def test_reflect():
    """Test the Arc3D reflect method."""
    pt = Point3D(2, 2, 0)
    arc = Arc3D(Plane(o=pt), 2, 0, math.pi)

    origin_1 = Point3D(0, 1)
    origin_2 = Point3D(1, 1)
    normal_1 = Vector3D(0, 1)
    normal_2 = Vector3D(-1, 1).normalize()
    normal_3 = Vector3D(1, 0)

    assert arc.reflect(normal_1, origin_1).c.x == pytest.approx(2, rel=1e-3)
    assert arc.reflect(normal_1, origin_1).c.y == pytest.approx(0, rel=1e-3)
    assert arc.reflect(normal_1, origin_1).midpoint.x == pytest.approx(2, rel=1e-3)
    assert arc.reflect(normal_1, origin_1).midpoint.y == pytest.approx(-2, rel=1e-3)
    assert arc.reflect(normal_1, origin_2).c == Point3D(2, 0)
    assert arc.reflect(normal_1, origin_2).midpoint.x == pytest.approx(2, rel=1e-3)
    assert arc.reflect(normal_1, origin_2).midpoint.y == pytest.approx(-2, rel=1e-3)

    test_1 = arc.reflect(normal_2, origin_2)
    assert test_1.c.x == pytest.approx(2, rel=1e-3)
    assert test_1.c.y == pytest.approx(2, rel=1e-3)
    assert test_1.midpoint.x == pytest.approx(4, rel=1e-3)
    assert test_1.midpoint.y == pytest.approx(2, rel=1e-3)

    test_2 = arc.reflect(normal_3, origin_1)
    assert test_2.c.x == pytest.approx(-2, rel=1e-3)
    assert test_2.c.y == pytest.approx(2, rel=1e-3)
    assert test_2.midpoint.x == pytest.approx(-2, rel=1e-3)
    assert test_2.midpoint.y == pytest.approx(4, rel=1e-3)


def test_subdivide():
    """Test the Arc3D subdivide methods."""
    pt = Point3D(2, 0, 2)
    arc = Arc3D(Plane(o=pt), 1, 1.5 * math.pi, 0.5 * math.pi)

    divisions = arc.subdivide(0.5)
    assert len(divisions) == 8
    divisions = arc.subdivide([1, 0.5, 0.25])
    assert len(divisions) == 10
    divisions = arc.subdivide_evenly(4)
    assert len(divisions) == 5


def test_to_polyline():
    """Test the Arc3D to_polyline methods."""
    pt = Point3D(2, 0, 2)
    arc = Arc3D(Plane(o=pt), 1, 1.5 * math.pi, 0.5 * math.pi)

    pline = arc.to_polyline(4, True)
    assert isinstance(pline, Polyline3D)
    assert pline.vertices[0] == arc.p1
    assert pline.vertices[-1] == arc.p2
    assert len(pline.vertices) == 5
    assert pline.interpolated


def test_point_at_angle():
    """Test the Arc3D point_at_angle method."""
    pt = Point3D(2, 0)
    arc = Arc3D(Plane(o=pt), 1, 1.5 * math.pi, 0.5 * math.pi)

    assert arc.point_at_angle(0.5 * math.pi).x == pytest.approx(3, rel=1e-2)
    assert arc.point_at_angle(0.5 * math.pi).y == pytest.approx(0, rel=1e-2)


def test_arc3_closest_point():
    """Test the Arc3D closest_point method."""
    pt = Point3D(2, 0, 2)
    arc = Arc3D(Plane(o=pt), 1, 0, math.pi)

    assert arc.closest_point(Point3D(2, 2)) == Point3D(2, 1, 2)
    assert arc.closest_point(Point3D(3, 1)).x == pytest.approx(2.71, rel=1e-2)
    assert arc.closest_point(Point3D(3, 1)).y == pytest.approx(0.71, rel=1e-2)
    assert arc.closest_point(Point3D(3, -1)) == Point3D(3, 0, 2)
    assert arc.closest_point(Point3D(1, -1)).x == pytest.approx(1, rel=1e-2)
    assert arc.closest_point(Point3D(1, -1)).y == pytest.approx(0, rel=1e-2)

    assert arc.distance_to_point(Point3D(2, 2)) == math.sqrt(5)
    assert arc.distance_to_point(Point3D(3, 1)) == pytest.approx(2.0424428, rel=1e-2)
    assert arc.distance_to_point(Point3D(3, -1, 2)) == pytest.approx(1, rel=1e-2)
    assert arc.distance_to_point(Point3D(1, -1, 2)) == pytest.approx(1, rel=1e-2)


def test_arc3_intersect_with_plane():
    """Test the Arc3D intersect_with_plane method."""
    pt = Point3D(2, 0, 2)
    arc = Arc3D(Plane(o=pt), 1, 0, math.pi)
    circle = Arc3D(Plane(o=pt), 1)

    plane_1 = Plane(Vector3D(0, 1, 0), Point3D(0, 0, 0))
    int1 = circle.intersect_plane(plane_1)

    assert len(int1) == 2
    assert int1[0].x == pytest.approx(3, rel=1e-2)
    assert int1[0].y == pytest.approx(0, rel=1e-2)
    assert int1[1].x == pytest.approx(1, rel=1e-2)
    assert int1[1].y == pytest.approx(0, rel=1e-2)

    plane_2 = Plane(Vector3D(0, 1, 0), Point3D(0, 0.5, 0))
    int2 = arc.intersect_plane(plane_2)
    assert len(int2) == 2


def test_arc3_split_with_plane():
    """Test the Arc3D split_with_plane method."""
    pt = Point3D(2, 0, 2)
    arc = Arc3D(Plane(o=pt), 1, 0, math.pi)
    circle = Arc3D(Plane(o=pt), 1)

    plane_1 = Plane(Vector3D(0, 1, 0), Point3D(0, 0, 0))
    int1 = circle.split_with_plane(plane_1)
    assert len(int1) == 2
    assert int1[0].p1 == Point3D(3, 0, 2)
    assert int1[0].p2.x == pytest.approx(1, rel=1e-2)
    assert int1[0].p2.y == pytest.approx(0, rel=1e-2)
    assert int1[1].p2 == Point3D(3, 0, 2)
    assert int1[1].p1.x == pytest.approx(1, rel=1e-2)
    assert int1[1].p1.y == pytest.approx(0, rel=1e-2)

    plane_2 = Plane(Vector3D(0, 1, 0), Point3D(0, 0.5, 0))
    int2 = arc.split_with_plane(plane_2)
    assert len(int2) == 3
