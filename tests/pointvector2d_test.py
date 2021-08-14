# coding=utf-8
import pytest

from ladybug_geometry.geometry2d.pointvector import Vector2D, Point2D

import math


def test_vector2_init():
    """Test the initialization of Vector2D objects and basic properties."""
    vec = Vector2D(0, 2)
    str(vec)  # test the string representation of the vector

    assert vec.x == 0
    assert vec.y == 2
    assert vec[0] == 0
    assert vec[1] == 2
    assert vec.magnitude == 2
    assert vec.magnitude_squared == 4
    assert not vec.is_zero(0.0000001)

    assert len(vec) == 2
    pt_tuple = tuple(i for i in vec)
    assert pt_tuple == (0, 2)

    norm_vec = vec.normalize()
    assert norm_vec.x == 0
    assert norm_vec.y == 1
    assert norm_vec.magnitude == 1


def test_zero_magnitude_vector():
    """Test properties with a zero magnitude vector."""
    vec = Vector2D(0, 0)

    assert vec.is_zero(0.0000001)
    assert vec.magnitude == 0
    assert vec.normalize() == vec


def test_vector2_to_from_dict():
    """Test the initialization of Vector2D objects and basic properties."""
    vec = Vector2D(0, 2)
    vec_dict = vec.to_dict()
    new_vec = Vector2D.from_dict(vec_dict)
    assert isinstance(new_vec, Vector2D)
    assert new_vec.to_dict() == vec_dict

    pt = Point2D(0, 2)
    pt_dict = pt.to_dict()
    new_pt = Point2D.from_dict(pt_dict)
    assert isinstance(new_pt, Point2D)
    assert new_pt.to_dict() == pt_dict


def test_distance_to_point():
    """Test the test_distance_to_point method."""
    pt_1 = Point2D(0, 2)
    assert pt_1.x == 0
    assert pt_1.y == 2

    pt_2 = Point2D(2, 2)
    assert pt_1.distance_to_point(pt_2) == 2


def test_equality():
    """Test the equality of Point2D objects."""
    pt_1 = Point2D(0, 2)
    pt_1_dup = pt_1.duplicate()
    pt_1_alt = Point2D(0.1, 2)

    assert pt_1 is pt_1
    assert pt_1 is not pt_1_dup
    assert pt_1 == pt_1_dup
    assert hash(pt_1) == hash(pt_1_dup)
    assert pt_1 != pt_1_alt
    assert hash(pt_1) != hash(pt_1_alt)


def test_is_equivalent():
    """Test the is_equivalent method using tolerances."""
    pt_1 = Point2D(0, 2)
    pt_2 = Point2D(0.00001, 2)
    pt_3 = Point2D(0, 1)

    assert pt_1.is_equivalent(pt_2, 0.0001) is True
    assert pt_1.is_equivalent(pt_2, 0.0000001) is False
    assert pt_1.is_equivalent(pt_3, 0.0001) is False


def test_vector2_mutability():
    """Test the mutability and immutability of Vector2D objects."""
    vec = Vector2D(0, 2)
    assert isinstance(vec, Vector2D)
    vec_copy = vec.duplicate()
    assert vec == vec_copy

    with pytest.raises(AttributeError):
        vec.x = 2
    norm_vec = vec.normalize()  # ensure operations tha yield new vectors are ok
    assert norm_vec.magnitude == pytest.approx(1., rel=1e-3)


def test_point2_mutability():
    """Test the mutability and immutability of Point2D objects."""
    pt = Point2D(0, 2)
    assert isinstance(pt, Point2D)
    pt_copy = pt.duplicate()
    assert pt == pt_copy

    with pytest.raises(AttributeError):
        pt.x = 2
    norm_vec = pt.normalize()  # ensure operations tha yield new vectors are ok
    assert norm_vec.magnitude == pytest.approx(1., rel=1e-3)


def test_vector2_dot_cross_determinant_reverse():
    """Test the methods for dot, cross, and determinant."""
    vec_1 = Vector2D(0, 2)
    vec_2 = Vector2D(2, 0)
    vec_3 = Vector2D(1, 1)

    assert vec_1.dot(vec_2) == 0
    assert vec_2.dot(vec_1) == 0
    assert vec_1.dot(vec_3) == 2
    assert vec_3.dot(vec_1) == 2

    assert vec_1.determinant(vec_2) == -4
    assert vec_2.determinant(vec_1) == 4
    assert vec_1.determinant(vec_3) == -2
    assert vec_3.determinant(vec_1) == 2

    assert vec_1.cross() == Vector2D(2, 0)
    assert vec_2.cross() == Vector2D(0, -2)
    assert vec_3.cross() == Vector2D(1, -1)

    assert vec_1.reverse() == Vector2D(0, -2)
    assert vec_2.reverse() == Vector2D(-2, 0)


def test_vector2_angle():
    """Test the methods that get the angle between Vector2D objects."""
    vec_1 = Vector2D(0, 2)
    vec_2 = Vector2D(2, 0)
    vec_3 = Vector2D(0, -2)
    vec_4 = Vector2D(-2, 0)
    assert vec_1.angle(vec_2) == pytest.approx(math.pi/2, rel=1e-3)
    assert vec_1.angle(vec_3) == pytest.approx(math.pi, rel=1e-3)
    assert vec_1.angle(vec_4) == pytest.approx(math.pi/2, rel=1e-3)
    assert vec_1.angle(vec_1) == pytest.approx(0, rel=1e-3)

    assert vec_1.angle_counterclockwise(vec_2) == pytest.approx(3*math.pi/2, rel=1e-3)
    assert vec_1.angle_counterclockwise(vec_3) == pytest.approx(math.pi, rel=1e-3)
    assert vec_1.angle_counterclockwise(vec_4) == pytest.approx(math.pi/2, rel=1e-3)
    assert vec_1.angle_counterclockwise(vec_1) == pytest.approx(0, rel=1e-3)

    assert vec_1.angle_clockwise(vec_2) == pytest.approx(math.pi/2, rel=1e-3)
    assert vec_1.angle_clockwise(vec_3) == pytest.approx(math.pi, rel=1e-3)
    assert vec_1.angle_clockwise(vec_4) == pytest.approx(3*math.pi/2, rel=1e-3)
    assert vec_1.angle_clockwise(vec_1) == pytest.approx(0, rel=1e-3)


def test_addition_subtraction():
    """Test the addition and subtraction methods."""
    vec_1 = Vector2D(0, 2)
    vec_2 = Vector2D(2, 0)
    pt_1 = Point2D(2, 0)
    pt_2 = Point2D(0, 2)
    assert isinstance(vec_1 + vec_2, Vector2D)
    assert isinstance(vec_1 + pt_1, Point2D)
    assert isinstance(pt_1 + pt_2, Vector2D)
    assert isinstance(vec_1 - vec_2, Vector2D)
    assert isinstance(vec_1 - pt_1, Point2D)
    assert isinstance(pt_1 - pt_2, Vector2D)
    assert vec_1 + vec_2 == Vector2D(2, 2)
    assert vec_1 + pt_1 == Point2D(2, 2)
    assert pt_1 + pt_2 == Vector2D(2, 2)
    assert vec_1 - vec_2 == Vector2D(-2, 2)
    assert vec_1 - pt_1 == Point2D(-2, 2)
    assert pt_1 - pt_2 == Vector2D(2, -2)

    assert -vec_1 == Vector2D(0, -2)

    vec_1 += vec_2
    assert isinstance(vec_1, Vector2D)
    assert vec_1 == Vector2D(2, 2)


def test_multiplication_division():
    """Test the multiplication and division methods."""
    vec_1 = Vector2D(0, 2)
    vec_2 = Vector2D(2, 0)
    pt_1 = Point2D(2, 0)
    pt_2 = Point2D(0, 2)
    assert vec_1 * 2 == Vector2D(0, 4)
    assert vec_2 * 2 == Vector2D(4, 0)
    assert pt_2 * 2 == Vector2D(0, 4)
    assert pt_1 * 2 == Vector2D(4, 0)
    assert vec_1 / 2 == Vector2D(0, 1)
    assert vec_2 / 2 == Vector2D(1, 0)
    assert pt_2 / 2 == Vector2D(0, 1)
    assert pt_1 / 2 == Vector2D(1, 0)

    vec_1 *= 2
    assert vec_1 == Vector2D(0, 4)

    vec_1 /= 2
    assert vec_1 == Vector2D(0, 2)


def test_move():
    """Test the Point2D move method."""
    pt_1 = Point2D(2, 2)
    vec_1 = Vector2D(0, 2)
    assert pt_1.move(vec_1) == Point2D(2, 4)


def test_scale():
    """Test the Point2D scale method."""
    pt_1 = Point2D(2, 2)
    origin_1 = Point2D(0, 2)
    origin_2 = Point2D(1, 1)
    assert pt_1.scale(2, origin_1) == Point2D(4, 2)
    assert pt_1.scale(2, origin_2) == Point2D(3, 3)


def test_scale_world_origin():
    """Test the Point2D scale method with None origin."""
    pt_1 = Point2D(2, 2)
    pt_2 = Point2D(-2, -2)
    assert pt_1.scale(2) == Point2D(4, 4)
    assert pt_1.scale(0.5) == Point2D(1, 1)
    assert pt_1.scale(-2) == Point2D(-4, -4)
    assert pt_2.scale(2) == Point2D(-4, -4)
    assert pt_2.scale(-2) == Point2D(4, 4)


def test_rotate():
    """Test the Point2D rotate method."""
    pt_1 = Point2D(2, 2)
    origin_1 = Point2D(0, 2)
    origin_2 = Point2D(1, 1)

    test_1 = pt_1.rotate(math.pi, origin_1)
    assert test_1.x == pytest.approx(-2, rel=1e-3)
    assert test_1.y == pytest.approx(2, rel=1e-3)

    test_2 = pt_1.rotate(math.pi/2, origin_1)
    assert test_2.x == pytest.approx(0, rel=1e-3)
    assert test_2.y == pytest.approx(4, rel=1e-3)

    test_3 = pt_1.rotate(math.pi, origin_2)
    assert test_3.x == pytest.approx(0, rel=1e-3)
    assert test_3.y == pytest.approx(0, rel=1e-3)

    test_4 = pt_1.rotate(math.pi/2, origin_2)
    assert test_4.x == pytest.approx(0, rel=1e-3)
    assert test_4.y == pytest.approx(2, rel=1e-3)


def test_reflect():
    """Test the Point2D reflect method."""
    pt_1 = Point2D(2, 2)
    origin_1 = Point2D(0, 1)
    origin_2 = Point2D(1, 1)
    normal_1 = Vector2D(0, 1)
    normal_2 = Vector2D(-1, 1).normalize()

    assert pt_1.reflect(normal_1, origin_1) == Point2D(2, 0)
    assert pt_1.reflect(normal_1, origin_2) == Point2D(2, 0)
    assert pt_1.reflect(normal_2, origin_2) == Point2D(2, 2)

    test_1 = pt_1.reflect(normal_2, origin_1)
    assert test_1.x == pytest.approx(1, rel=1e-3)
    assert test_1.y == pytest.approx(3, rel=1e-3)


def test_circular_mean():
    """Test the circular mean staticmethod."""
    angles_1 = [math.radians(x) for x in [45, 315]]
    angles_2 = [math.radians(x) for x in [45, 135]]
    angles_3 = [math.radians(x) for x in [90, 270]]

    assert Vector2D.circular_mean(angles_1) == pytest.approx(0, rel=1e-3)
    assert Vector2D.circular_mean(angles_2) == \
        pytest.approx(math.radians(90), rel=1e-3)
    assert Vector2D.circular_mean(angles_3) == \
        pytest.approx(math.radians(180), rel=1e-3)
