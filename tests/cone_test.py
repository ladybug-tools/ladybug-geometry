# coding=utf-8
import pytest

from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
from ladybug_geometry.geometry3d.cone import Cone

import math


def test_cone_init():
    """Test the initalization of Cone objects and basic properties."""
    vertex = Point3D(2, 0, 2)
    axis = Vector3D(0, 2, 2)
    angle = 0.7
    c = Cone(vertex, axis, angle)

    str(c)  # test the string representation of the cone
    assert c.vertex == Point3D(2, 0, 2)
    assert c.axis == Vector3D(0, 2, 2)
    assert c.angle == 0.7
    assert c.height == c.axis.magnitude
    assert isinstance(c.slant_height, float)
    assert isinstance(c.area, float)
    assert isinstance(c.volume, float)


def test_equality():
    """Test the equality of Cone objects."""
    cone = Cone(Point3D(2, 0, 2), Vector3D(0, 2, 2), 0.7)
    cone_dup = cone.duplicate()
    cone_alt = Cone(Point3D(2, 0.1, 2), Vector3D(0, 2, 2), 0.7)

    assert cone is cone
    assert cone is not cone_dup
    assert cone == cone_dup
    assert hash(cone) == hash(cone_dup)
    assert cone != cone_alt
    assert hash(cone) != hash(cone_alt)


def test_cone_to_from_dict():
    """Test the Cone to_dict and from_dict methods."""
    c = Cone(Point3D(4, 0.5, 2), Vector3D(1, 0, 2.5), 0.7)
    d = c.to_dict()
    c = Cone.from_dict(d)
    assert c.vertex.x == pytest.approx(4, rel=1e-3)
    assert c.vertex.y == pytest.approx(0.5, rel=1e-3)
    assert c.vertex.z == pytest.approx(2, rel=1e-3)
    assert c.axis.x == pytest.approx(1, rel=1e-3)
    assert c.axis.y == pytest.approx(0, rel=1e-3)
    assert c.axis.z == pytest.approx(2.5, rel=1e-3)


def test_cone_duplicate():
    """Test the Cone duplicate method."""
    c = Cone(Point3D(0, 0.5, 2), Vector3D(1, 0.5, 2.5), 0.75)
    test = c.duplicate()
    assert test.angle == 0.75
    assert c.vertex.x == pytest.approx(0, rel=1e-3)
    assert c.vertex.y == pytest.approx(0.5, rel=1e-3)
    assert c.vertex.z == pytest.approx(2, rel=1e-3)
    assert c.axis.x == pytest.approx(1, rel=1e-3)
    assert c.axis.y == pytest.approx(0.5, rel=1e-3)
    assert c.axis.z == pytest.approx(2.5, rel=1e-3)


def test_cone_rotate():
    """Test the Cone rotate method."""
    c = Cone(Point3D(2, 0, 2), Vector3D(1, 0.5, 2.5), 0.75)
    test1 = c.rotate(Vector3D(0, 0, 1), math.pi, Point3D(0, 0, 0))
    assert test1.vertex.x == pytest.approx(-2, rel=1e-3)
    assert test1.vertex.y == pytest.approx(0, rel=1e-3)
    assert test1.vertex.z == pytest.approx(2, rel=1e-3)
    assert test1.axis == Vector3D(1, 0.5, 2.5).rotate(Vector3D(0, 0, 1), math.pi)


def test_cone_rotate_xy():
    """Test the Cone rotate_xy method."""
    c = Cone(Point3D(1, 0, 3), Vector3D(1, 2, 0), 0.5)
    test = c.rotate_xy(math.pi / 2, Point3D(0, 0, 0))
    assert test.vertex.x == pytest.approx(0, rel=1e-3)
    assert test.vertex.y == pytest.approx(1, rel=1e-3)
    assert test.vertex.z == pytest.approx(3, rel=1e-3)
    assert test.axis == Vector3D(1, 2, 0).rotate_xy(math.pi / 2)


def test_cone_reflect():
    """Test the Cone reflect method."""
    origin_1 = Point3D(1, 0, 0)
    origin_2 = Point3D(0, 0, 2)
    normal_1 = Vector3D(0, 0, 1)
    normal_2 = Vector3D(1, 0, 0)

    c = Cone(Point3D(0, 0, 0), Vector3D(1, 2, 0), 0.5)
    test_1 = c.reflect(normal_1, origin_1)
    assert test_1.vertex.x == pytest.approx(0, rel=1e-3)
    assert test_1.vertex.y == pytest.approx(0, rel=1e-3)
    assert test_1.vertex.z == pytest.approx(0, rel=1e-3)

    test_2 = c.reflect(normal_2, origin_1)
    assert test_2.vertex.x == pytest.approx(2, rel=1e-3)
    assert test_2.vertex.y == pytest.approx(0, rel=1e-3)
    assert test_2.vertex.z == pytest.approx(0, rel=1e-3)

    test_3 = c.reflect(normal_1, origin_2)
    assert test_3.vertex.x == pytest.approx(0, rel=1e-3)
    assert test_3.vertex.y == pytest.approx(0, rel=1e-3)
    assert test_3.vertex.z == pytest.approx(4, rel=1e-3)


def test_cone_move():
    """Test the Cone move method."""
    c = Cone(Point3D(4, 0.5, 2), Vector3D(1, 0, 2.5), 0.7)
    test = c.move(Vector3D(2, 3, 6.5))
    assert test.vertex.x == pytest.approx(6, rel=1e-3)
    assert test.vertex.y == pytest.approx(3.5, rel=1e-3)
    assert test.vertex.z == pytest.approx(8.5, rel=1e-3)
    assert test.axis.x == pytest.approx(1, rel=1e-3)
    assert test.axis.y == pytest.approx(0, rel=1e-3)
    assert test.axis.z == pytest.approx(2.5, rel=1e-3)


def test_cone_scale():
    """Test the Cone scale method."""
    c = Cone(Point3D(4, 0.5, 2), Vector3D(1, 0, 2.5), 0.7)
    test = c.scale(2, Point3D(0, 0, 0))
    assert test.vertex.x == pytest.approx(8, rel=1e-3)
    assert test.vertex.y == pytest.approx(1, rel=1e-3)
    assert test.vertex.z == pytest.approx(4, rel=1e-3)
    assert test.axis.x == pytest.approx(2, rel=1e-3)
    assert test.axis.y == pytest.approx(0, rel=1e-3)
    assert test.axis.z == pytest.approx(5, rel=1e-3)
