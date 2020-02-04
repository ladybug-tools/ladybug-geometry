# coding=utf-8
import pytest

from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
from ladybug_geometry.geometry3d.sphere import Sphere
from ladybug_geometry.geometry3d.plane import Plane
from ladybug_geometry.geometry3d.line import LineSegment3D
from ladybug_geometry.geometry3d.arc import Arc3D

import math


def test_sphere_init():
    """Test the initalization of Sphere objects and basic properties."""
    pt = Point3D(2, 0, 2)
    r = 3
    sp = Sphere(pt, r)

    str(sp)  # test the string representation of the line segment
    assert sp.center == Point3D(2, 0, 2)
    assert sp.radius == 3
    assert sp.min.z == -1
    assert sp.max.z == 5
    assert sp.diameter == 6
    assert isinstance(sp.circumference, float)
    assert isinstance(sp.area, float)
    assert isinstance(sp.volume, float)


def test_equality():
    """Test the equality of Sphere objects."""
    sphere = Sphere(Point3D(2, 0, 2), 3)
    sphere_dup = sphere.duplicate()
    sphere_alt = Sphere(Point3D(2, 0.1, 2), 3)

    assert sphere is sphere
    assert sphere is not sphere_dup
    assert sphere == sphere_dup
    assert hash(sphere) == hash(sphere_dup)
    assert sphere != sphere_alt
    assert hash(sphere) != hash(sphere_alt)


def test_sphere_to_from_dict():
    """Test the Sphere to_dict and from_dict methods."""
    sp = Sphere(Point3D(4, 0, 2), 3)
    d = sp.to_dict()
    sp = Sphere.from_dict(d)
    assert sp.center.x == pytest.approx(4, rel=1e-3)
    assert sp.center.y == pytest.approx(0, rel=1e-3)
    assert sp.center.z == pytest.approx(2, rel=1e-3)


def test_sphere_duplicate():
    """Test the Sphere duplicate method."""
    sp = Sphere(Point3D(8.5, 1.2, 2.9), 6.5)
    test = sp.duplicate()
    assert test.radius == 6.5
    assert test.center.x == pytest.approx(8.5, rel=1e-3)
    assert test.center.y == pytest.approx(1.2, rel=1e-3)
    assert test.center.z == pytest.approx(2.9, rel=1e-3)


def test_sphere_rotate():
    """Test the Sphere rotate method."""
    sp = Sphere(Point3D(2, 0, 2), 3)
    test1 = sp.rotate(Vector3D(0, 0, 1), math.pi, Point3D(0, 0, 0))
    assert test1.center.x == pytest.approx(-2, rel=1e-3)
    assert test1.center.y == pytest.approx(0, rel=1e-3)
    assert test1.center.z == pytest.approx(2, rel=1e-3)

    test2 = sp.rotate(Vector3D(1, 0, 0), math.pi, Point3D(0, 0, 0))
    assert test2.center.x == pytest.approx(2, rel=1e-3)
    assert test2.center.y == pytest.approx(0, rel=1e-3)
    assert test2.center.z == pytest.approx(-2, rel=1e-3)


def test_sphere_rotate_xy():
    """Test the Sphere rotate_xy method."""
    sp = Sphere(Point3D(4, 0, 2), 3)
    test = sp.rotate_xy(math.pi / 2, Point3D(0, 0, 0))
    assert test.center.x == pytest.approx(0, rel=1e-3)
    assert test.center.y == pytest.approx(4, rel=1e-3)
    assert test.center.z == pytest.approx(2, rel=1e-3)


def test_sphere_reflect():
    """Test the Sphere reflect method."""
    origin_1 = Point3D(1, 0, 0)
    origin_2 = Point3D(0, 0, 2)
    normal_1 = Vector3D(0, 0, 1)
    normal_2 = Vector3D(1, 0, 0)

    sp = Sphere(Point3D(0, 0, 0), 3)
    test_1 = sp.reflect(normal_1, origin_1)
    assert test_1.center.x == pytest.approx(0, rel=1e-3)
    assert test_1.center.y == pytest.approx(0, rel=1e-3)
    assert test_1.center.z == pytest.approx(0, rel=1e-3)

    test_2 = sp.reflect(normal_2, origin_1)
    assert test_2.center.x == pytest.approx(2, rel=1e-3)
    assert test_2.center.y == pytest.approx(0, rel=1e-3)
    assert test_2.center.z == pytest.approx(0, rel=1e-3)

    test_3 = sp.reflect(normal_1, origin_2)
    assert test_3.center.x == pytest.approx(0, rel=1e-3)
    assert test_3.center.y == pytest.approx(0, rel=1e-3)
    assert test_3.center.z == pytest.approx(4, rel=1e-3)


def test_sphere_move():
    """Test the Sphere move method."""
    sp = Sphere(Point3D(2, 0, 2), 3)
    test = sp.move(Vector3D(2, 3, 6.5))
    assert test.center.x == pytest.approx(4, rel=1e-3)
    assert test.center.y == pytest.approx(3, rel=1e-3)
    assert test.center.z == pytest.approx(8.5, rel=1e-3)


def test_sphere_scale():
    """Test the Sphere scale method."""
    sp = Sphere(Point3D(4, 0, 2), 2.5)
    test = sp.scale(2, Point3D(0, 0, 0))
    assert test.radius == 5
    assert test.center.x == pytest.approx(8, rel=1e-3)
    assert test.center.y == pytest.approx(0, rel=1e-3)
    assert test.center.z == pytest.approx(4, rel=1e-3)


def test_sphere_intersection_with_line_ray():
    """Test the Sphere intersect_line_ray method."""
    lpt = Point3D(-2, 0, 0)
    vec = Vector3D(4, 0, 0)
    seg = LineSegment3D(lpt, vec)
    spt = Point3D(0, 0, 0)
    sp = Sphere(spt, 1.5)

    int1 = sp.intersect_line_ray(seg)
    assert isinstance(int1, LineSegment3D)
    assert int1.p == Point3D(1.5, 0, 0)

    lpt = Point3D(-2, 0, 1.5)
    vec = Vector3D(4, 0, 0)
    seg = LineSegment3D(lpt, vec)
    int2 = sp.intersect_line_ray(seg)
    assert isinstance(int2, Point3D)


def test_sphere_intersection_with_plane():
    """Test the Sphere intersect_plane method."""
    ppt = Point3D(-1.5, 0, 1.46)
    vec = Vector3D(0.1, 0, 1)
    pl = Plane(vec, ppt)
    spt = Point3D(0, 0, 0)
    sp = Sphere(spt, 1.5)
    int1 = sp.intersect_plane(pl)
    assert isinstance(int1, Arc3D)

    ppt = Point3D(0, 0, 0)
    vec = Vector3D(0, 0, 1)
    pl = Plane(vec, ppt)
    int2 = sp.intersect_plane(pl)
    assert int2.c == ppt
    assert int2.radius == 1.5

    ppt = Point3D(0, 0, 1.5)
    vec = Vector3D(0, 0, 1)
    pl = Plane(vec, ppt)
    int3 = sp.intersect_plane(pl)
    assert int3 is None
