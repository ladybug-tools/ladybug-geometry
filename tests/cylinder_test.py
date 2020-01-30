# coding=utf-8
import pytest

from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
from ladybug_geometry.geometry3d.cylinder import Cylinder

import math


def test_cylinder_init():
    """Test the initalization of Cylinder objects and basic properties."""
    center = Point3D(2, 0, 2)
    axis = Vector3D(0, 2, 2)
    radius = 0.7
    c = Cylinder(center, axis, radius)

    str(c)  # test the string representation of the cylinder
    assert c.center == Point3D(2, 0, 2)
    assert c.axis == Vector3D(0, 2, 2)
    assert c.radius == 0.7
    assert c.height == c.axis.magnitude
    assert c.center_end == c.center + c.axis
    assert c.diameter == c.radius * 2
    assert isinstance(c.area, float)
    assert isinstance(c.volume, float)

    p1 = Point3D(1, 1, 0)
    p2 = Point3D(1, 1, 5)
    radius = 1.2
    c2 = Cylinder.from_start_end(p1, p2, radius)
    assert c2.center == Point3D(1, 1, 0)
    assert c2.axis == Vector3D(0, 0, 5)
    assert c2.radius == 1.2
    assert c2.height == c2.axis.magnitude
    assert c2.center_end == c2.center + c2.axis
    assert c2.diameter == c2.radius * 2
    assert isinstance(c2.area, float)
    assert isinstance(c2.volume, float)


def test_equality():
    """Test the equality of Cylinder objects."""
    cylinder = Cylinder(Point3D(2, 0, 2), Vector3D(0, 2, 2), 0.7)
    cylinder_dup = cylinder.duplicate()
    cylinder_alt = Cylinder(Point3D(2, 0.1, 2), Vector3D(0, 2, 2), 0.7)

    assert cylinder is cylinder
    assert cylinder is not cylinder_dup
    assert cylinder == cylinder_dup
    assert hash(cylinder) == hash(cylinder_dup)
    assert cylinder != cylinder_alt
    assert hash(cylinder) != hash(cylinder_alt)


def test_cylinder_to_from_dict():
    """Test the Cylinder to_dict and from_dict methods."""
    c = Cylinder(Point3D(4, 0.5, 2), Vector3D(1, 0, 2.5), 0.7)
    d = c.to_dict()
    c = Cylinder.from_dict(d)
    assert c.center.x == pytest.approx(4, rel=1e-3)
    assert c.center.y == pytest.approx(0.5, rel=1e-3)
    assert c.center.z == pytest.approx(2, rel=1e-3)
    assert c.axis.x == pytest.approx(1, rel=1e-3)
    assert c.axis.y == pytest.approx(0, rel=1e-3)
    assert c.axis.z == pytest.approx(2.5, rel=1e-3)


def test_cylinder_duplicate():
    """Test the Cylinder duplicate method."""
    c = Cylinder(Point3D(0, 0.5, 2), Vector3D(1, 0.5, 2.5), 0.75)
    test = c.duplicate()
    assert test.radius == 0.75
    assert c.center.x == pytest.approx(0, rel=1e-3)
    assert c.center.y == pytest.approx(0.5, rel=1e-3)
    assert c.center.z == pytest.approx(2, rel=1e-3)
    assert c.axis.x == pytest.approx(1, rel=1e-3)
    assert c.axis.y == pytest.approx(0.5, rel=1e-3)
    assert c.axis.z == pytest.approx(2.5, rel=1e-3)


def test_cylinder_rotate():
    """Test the Cylinder rotate method."""
    c = Cylinder(Point3D(2, 0, 2), Vector3D(1, 0.5, 2.5), 0.75)
    test1 = c.rotate(Vector3D(0, 0, 1), math.pi, Point3D(0, 0, 0))
    assert test1.center.x == pytest.approx(-2, rel=1e-3)
    assert test1.center.y == pytest.approx(0, rel=1e-3)
    assert test1.center.z == pytest.approx(2, rel=1e-3)
    assert test1.axis == Vector3D(1, 0.5, 2.5).rotate(Vector3D(0, 0, 1), math.pi)


def test_cylinder_rotate_xy():
    """Test the Cylinder rotate_xy method."""
    c = Cylinder(Point3D(1, 0, 3), Vector3D(1, 2, 0), 0.5)
    test = c.rotate_xy(math.pi / 2, Point3D(0, 0, 0))
    assert test.center.x == pytest.approx(0, rel=1e-3)
    assert test.center.y == pytest.approx(1, rel=1e-3)
    assert test.center.z == pytest.approx(3, rel=1e-3)
    assert test.axis == Vector3D(1, 2, 0).rotate_xy(math.pi / 2)


def test_cylinder_reflect():
    """Test the Cylinder reflect method."""
    origin_1 = Point3D(1, 0, 0)
    origin_2 = Point3D(0, 0, 2)
    normal_1 = Vector3D(0, 0, 1)
    normal_2 = Vector3D(1, 0, 0)

    c = Cylinder(Point3D(0, 0, 0), Vector3D(1, 2, 0), 0.5)
    test_1 = c.reflect(normal_1, origin_1)
    assert test_1.center.x == pytest.approx(0, rel=1e-3)
    assert test_1.center.y == pytest.approx(0, rel=1e-3)
    assert test_1.center.z == pytest.approx(0, rel=1e-3)

    test_2 = c.reflect(normal_2, origin_1)
    assert test_2.center.x == pytest.approx(2, rel=1e-3)
    assert test_2.center.y == pytest.approx(0, rel=1e-3)
    assert test_2.center.z == pytest.approx(0, rel=1e-3)

    test_3 = c.reflect(normal_1, origin_2)
    assert test_3.center.x == pytest.approx(0, rel=1e-3)
    assert test_3.center.y == pytest.approx(0, rel=1e-3)
    assert test_3.center.z == pytest.approx(4, rel=1e-3)


def test_cylinder_move():
    """Test the Cylinder move method."""
    c = Cylinder(Point3D(4, 0.5, 2), Vector3D(1, 0, 2.5), 0.7)
    test = c.move(Vector3D(2, 3, 6.5))
    assert test.center.x == pytest.approx(6, rel=1e-3)
    assert test.center.y == pytest.approx(3.5, rel=1e-3)
    assert test.center.z == pytest.approx(8.5, rel=1e-3)
    assert test.axis.x == pytest.approx(1, rel=1e-3)
    assert test.axis.y == pytest.approx(0, rel=1e-3)
    assert test.axis.z == pytest.approx(2.5, rel=1e-3)


def test_cylinder_scale():
    """Test the Cylinder scale method."""
    c = Cylinder(Point3D(4, 0.5, 2), Vector3D(1, 0, 2.5), 0.7)
    test = c.scale(2, Point3D(0, 0, 0))
    assert test.center.x == pytest.approx(8, rel=1e-3)
    assert test.center.y == pytest.approx(1, rel=1e-3)
    assert test.center.z == pytest.approx(4, rel=1e-3)
    assert test.axis.x == pytest.approx(2, rel=1e-3)
    assert test.axis.y == pytest.approx(0, rel=1e-3)
    assert test.axis.z == pytest.approx(5, rel=1e-3)
    assert test.radius == pytest.approx(1.4, rel=1e-3)
