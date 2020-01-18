# coding=utf-8
import pytest

from ladybug_geometry.geometry3d.pointvector import Vector3D, Point3D

import math


def test_vector3_init():
    """Test the initalization of Vector3D objects and basic properties."""
    vec = Vector3D(0, 2, 0)
    str(vec)  # test the string representation of the vector

    assert vec.x == 0
    assert vec.y == 2
    assert vec.z == 0
    assert vec[0] == 0
    assert vec[1] == 2
    assert vec[2] == 0
    assert vec.magnitude == 2
    assert vec.magnitude_squared == 4
    assert not vec.is_zero()

    assert len(vec) == 3
    pt_tuple = tuple(i for i in vec)
    assert pt_tuple == (0, 2, 0)

    norm_vec = vec.normalize()
    assert norm_vec.x == 0
    assert norm_vec.z == 0
    assert norm_vec.magnitude == 1


def test_equality():
    """Test the equality of Point3D objects."""
    pt_1 = Point3D(0, 2, 1)
    pt_1_dup = pt_1.duplicate()
    pt_1_alt = Point3D(0.1, 2, 1)

    assert pt_1 is pt_1
    assert pt_1 is not pt_1_dup
    assert pt_1 == pt_1_dup
    assert hash(pt_1) == hash(pt_1_dup)
    assert pt_1 != pt_1_alt
    assert hash(pt_1) != hash(pt_1_alt)


def test_vector3_to_from_dict():
    """Test the initalization of Vector3D objects and basic properties."""
    vec = Vector3D(0, 2, 0)
    vec_dict = vec.to_dict()
    new_vec = Vector3D.from_dict(vec_dict)
    assert isinstance(new_vec, Vector3D)
    assert new_vec.to_dict() == vec_dict

    pt = Point3D(0, 2, 0)
    pt_dict = pt.to_dict()
    new_pt = Point3D.from_dict(pt_dict)
    assert isinstance(new_pt, Point3D)
    assert new_pt.to_dict() == pt_dict


def test_zero_magnitude_vector():
    """Test properties with a zero magnitude vecotr."""
    vec = Vector3D(0, 0, 0)

    assert vec.is_zero()
    assert vec.magnitude == 0
    assert vec.normalize() == vec


def test_test_distance_to_point():
    """Test the test_distance_to_point method."""
    pt_1 = Point3D(0, 2, 0)
    assert pt_1.x == 0
    assert pt_1.y == 2
    assert pt_1.z == 0

    pt_2 = Point3D(2, 2)
    assert pt_1.distance_to_point(pt_2) == 2


def test_is_equivalent():
    """Test the is_equivalent method."""
    pt_1 = Point3D(0, 2, 0)
    pt_2 = Point3D(0.00001, 2, 0)
    pt_3 = Point3D(0, 1, 0)

    assert pt_1.is_equivalent(pt_2, 0.0001) is True
    assert pt_1.is_equivalent(pt_2, 0.0000001) is False
    assert pt_1.is_equivalent(pt_3, 0.0001) is False


def test_vector3_mutability():
    """Test the immutability of Vector3D objects."""
    vec = Vector3D(0, 2, 0)
    assert isinstance(vec, Vector3D)
    vec_copy = vec.duplicate()
    assert vec == vec_copy

    with pytest.raises(AttributeError):
        vec.x = 2
    norm_vec = vec.normalize()  # ensure operations tha yield new vectors are ok
    assert norm_vec.magnitude == pytest.approx(1., rel=1e-3)


def test_point3_mutability():
    """Test the immutability of Point3D objects."""
    pt = Point3D(0, 2, 0)
    assert isinstance(pt, Point3D)
    pt_copy = pt.duplicate()
    assert pt == pt_copy

    with pytest.raises(AttributeError):
        pt.x = 2
    norm_vec = pt.normalize()  # ensure operations tha yield new vectors are ok
    assert norm_vec.magnitude == pytest.approx(1., rel=1e-3)


def test_vector3_dot_cross_reverse():
    """Test the methods for dot, cross, and reverse."""
    vec_1 = Vector3D(0, 2, 0)
    vec_2 = Vector3D(2, 0, 0)
    vec_3 = Vector3D(1, 1, 0)

    assert vec_1.dot(vec_2) == 0
    assert vec_2.dot(vec_1) == 0
    assert vec_1.dot(vec_3) == 2
    assert vec_3.dot(vec_1) == 2

    assert vec_1.cross(vec_2) == Vector3D(0, 0, -4)
    assert vec_2.cross(vec_1) == Vector3D(0, 0, 4)
    assert vec_1.cross(vec_3) == Vector3D(0, 0, -2)
    assert vec_3.cross(vec_1) == Vector3D(0, 0, 2)

    assert vec_1.reverse() == Vector3D(0, -2, 0)
    assert vec_2.reverse() == Vector3D(-2, 0, 0)


def test_vector3_angle():
    """Test the methods that get the angle between Vector3D objects."""
    vec_1 = Vector3D(0, 2, 0)
    vec_2 = Vector3D(2, 0, 0)
    vec_3 = Vector3D(0, -2, 0)
    vec_4 = Vector3D(-2, 0, 0)
    vec_5 = Vector3D(0, 0, 2)
    vec_6 = Vector3D(0, 0, -2)
    assert vec_1.angle(vec_2) == pytest.approx(math.pi/2, rel=1e-3)
    assert vec_1.angle(vec_3) == pytest.approx(math.pi, rel=1e-3)
    assert vec_1.angle(vec_4) == pytest.approx(math.pi/2, rel=1e-3)
    assert vec_1.angle(vec_5) == pytest.approx(math.pi/2, rel=1e-3)
    assert vec_5.angle(vec_6) == pytest.approx(math.pi, rel=1e-3)
    assert vec_1.angle(vec_1) == pytest.approx(0, rel=1e-3)


def test_addition_subtraction():
    """Test the addition and subtraction methods."""
    vec_1 = Vector3D(0, 2, 0)
    vec_2 = Vector3D(2, 0, 0)
    pt_1 = Point3D(2, 0, 0)
    pt_2 = Point3D(0, 2, 0)
    assert isinstance(vec_1 + vec_2, Vector3D)
    assert isinstance(vec_1 + pt_1, Point3D)
    assert isinstance(pt_1 + pt_2, Vector3D)
    assert isinstance(vec_1 - vec_2, Vector3D)
    assert isinstance(vec_1 - pt_1, Point3D)
    assert isinstance(pt_1 - pt_2, Vector3D)
    assert vec_1 + vec_2 == Vector3D(2, 2, 0)
    assert vec_1 + pt_1 == Point3D(2, 2, 0)
    assert pt_1 + pt_2 == Vector3D(2, 2, 0)
    assert vec_1 - vec_2 == Vector3D(-2, 2, 0)
    assert vec_1 - pt_1 == Point3D(-2, 2, 0)
    assert pt_1 - pt_2 == Vector3D(2, -2, 0)

    assert -vec_1 == Vector3D(0, -2, 0)

    vec_1 += vec_2
    assert isinstance(vec_1, Vector3D)
    assert vec_1 == Vector3D(2, 2, 0)


def test_multiplication_division():
    """Test the multiplication and division methods."""
    vec_1 = Vector3D(0, 2, 0)
    vec_2 = Vector3D(2, 0, 0)
    pt_1 = Point3D(2, 0, 0)
    pt_2 = Point3D(0, 2, 0)
    assert vec_1 * 2 == Vector3D(0, 4, 0)
    assert vec_2 * 2 == Vector3D(4, 0, 0)
    assert pt_2 * 2 == Vector3D(0, 4, 0)
    assert pt_1 * 2 == Vector3D(4, 0, 0)
    assert vec_1 / 2 == Vector3D(0, 1, 0)
    assert vec_2 / 2 == Vector3D(1, 0, 0)
    assert pt_2 / 2 == Vector3D(0, 1, 0)
    assert pt_1 / 2 == Vector3D(1, 0, 0)

    vec_1 *= 2
    assert vec_1 == Vector3D(0, 4, 0)

    vec_1 /= 2
    assert vec_1 == Vector3D(0, 2, 0)


def test_move():
    """Test the Point3D move method."""
    pt_1 = Point3D(2, 2, 0)
    vec_1 = Vector3D(0, 2, 2)
    assert pt_1.move(vec_1) == Point3D(2, 4, 2)


def test_scale():
    """Test the Point3D scale method."""
    pt_1 = Point3D(2, 2, 2)
    origin_1 = Point3D(0, 2, 2)
    origin_2 = Point3D(1, 1, 1)
    assert pt_1.scale(2, origin_1) == Point3D(4, 2, 2)
    assert pt_1.scale(2, origin_2) == Point3D(3, 3, 3)


def test_scale_world_origin():
    """Test the Point3D scale method with None origin."""
    pt_1 = Point3D(2, 2, 2)
    pt_2 = Point3D(-2, -2, -2)
    assert pt_1.scale(2) == Point3D(4, 4, 4)
    assert pt_1.scale(0.5) == Point3D(1, 1, 1)
    assert pt_1.scale(-2) == Point3D(-4, -4, -4)
    assert pt_2.scale(2) == Point3D(-4, -4, -4)
    assert pt_2.scale(-2) == Point3D(4, 4, 4)


def test_rotate():
    """Test the Point3D rotate method."""
    pt_1 = Point3D(2, 2, 2)
    axis_1 = Vector3D(-1, 0, 0)
    axis_2 = Vector3D(-1, -1, 0)
    origin_1 = Point3D(0, 2, 0)
    origin_2 = Point3D(0, 0, 0)

    test_1 = pt_1.rotate(axis_1, math.pi, origin_1)
    assert test_1.x == pytest.approx(2, rel=1e-3)
    assert test_1.y == pytest.approx(2, rel=1e-3)
    assert test_1.z == pytest.approx(-2, rel=1e-3)

    test_2 = pt_1.rotate(axis_1, math.pi/2, origin_1)
    assert test_2.x == pytest.approx(2, rel=1e-3)
    assert test_2.y == pytest.approx(4, rel=1e-3)
    assert test_2.z == pytest.approx(0, rel=1e-3)

    test_3 = pt_1.rotate(axis_2, math.pi, origin_2)
    assert test_3.x == pytest.approx(2, rel=1e-3)
    assert test_3.y == pytest.approx(2, rel=1e-3)
    assert test_3.z == pytest.approx(-2, rel=1e-3)

    test_4 = pt_1.rotate(axis_2, math.pi/2, origin_2)
    assert test_4.x == pytest.approx(0.59, rel=1e-2)
    assert test_4.y == pytest.approx(3.41, rel=1e-2)
    assert test_4.z == pytest.approx(0, rel=1e-2)


def test_rotate_xy():
    """Test the Point3D rotate_xy method."""
    pt_1 = Point3D(2, 2, -2)
    origin_1 = Point3D(0, 2, 0)
    origin_2 = Point3D(1, 1, -2)

    test_1 = pt_1.rotate_xy(math.pi, origin_1)
    assert test_1.x == pytest.approx(-2, rel=1e-3)
    assert test_1.y == pytest.approx(2, rel=1e-3)
    assert test_1.z == -2

    test_2 = pt_1.rotate_xy(math.pi/2, origin_1)
    assert test_2.x == pytest.approx(0, rel=1e-3)
    assert test_2.y == pytest.approx(4, rel=1e-3)
    assert test_1.z == -2

    test_3 = pt_1.rotate_xy(math.pi, origin_2)
    assert test_3.x == pytest.approx(0, rel=1e-3)
    assert test_3.y == pytest.approx(0, rel=1e-3)
    assert test_1.z == -2

    test_4 = pt_1.rotate_xy(math.pi/2, origin_2)
    assert test_4.x == pytest.approx(0, rel=1e-3)
    assert test_4.y == pytest.approx(2, rel=1e-3)
    assert test_1.z == -2


def test_reflect():
    """Test the Point3D reflect method."""
    pt_1 = Point3D(2, 2, 2)
    origin_1 = Point3D(0, 1, 0)
    origin_2 = Point3D(1, 1, 0)
    normal_1 = Vector3D(0, 1, 0)
    normal_2 = Vector3D(-1, 1).normalize()

    assert pt_1.reflect(normal_1, origin_1) == Point3D(2, 0, 2)
    assert pt_1.reflect(normal_1, origin_2) == Point3D(2, 0, 2)
    assert pt_1.reflect(normal_2, origin_2) == Point3D(2, 2, 2)

    test_1 = pt_1.reflect(normal_2, origin_1)
    assert test_1.x == pytest.approx(1, rel=1e-3)
    assert test_1.y == pytest.approx(3, rel=1e-3)
    assert test_1.z == pytest.approx(2, rel=1e-3)


def test_project():
    """Test the Point3D project method."""
    pt_1 = Point3D(2, 2, 2)
    origin_1 = Point3D(1, 0, 0)
    origin_2 = Point3D(0, 1, 0)
    normal_1 = Vector3D(0, 1, 0)

    assert pt_1.project(normal_1, origin_1) == Point3D(2, 0, 2)
    assert pt_1.project(normal_1, origin_2) == Point3D(2, 1, 2)
