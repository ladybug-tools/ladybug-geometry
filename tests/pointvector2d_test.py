# coding=utf-8

from ladybug_geometry.geometry2d.pointvector import Vector2D, Point2D, \
    Vector2DImmutable, Point2DImmutable

import unittest
import pytest
import math


class Point2DTestCase(unittest.TestCase):
    """Test for Vector2D, Point2D"""

    def test_vector2_init(self):
        """Test the initalization of Vector2D objects and basic properties."""
        vec = Vector2D(0, 2)
        str(vec)  # test the string representation of the vector

        assert vec.x == 0
        assert vec.y == 2
        assert vec[0] == 0
        assert vec[1] == 2
        assert vec.magnitude == 2
        assert vec.magnitude_squared == 4

        assert len(vec) == 2
        pt_tuple = tuple(i for i in vec)
        assert pt_tuple == (0, 2)

        norm_vec = vec.normalized()
        assert norm_vec.x == 0
        assert norm_vec.magnitude == 1

        assert vec.magnitude == 2
        vec.normalize()
        assert vec.magnitude == 1

    def test_point2_init(self):
        """Test the initalization of Point2D objects and basic properties."""
        pt_1 = Point2D(0, 2)
        str(pt_1)  # test the string representation of the vector
        assert pt_1.x == 0
        assert pt_1.y == 2

        pt_2 = Point2D(2, 2)
        assert pt_1.distance_to_point(pt_2) == 2

    def test_vector2_mutability(self):
        """Test the mutability and immutability of Vector2D objects."""
        vec = Vector2D(0, 2)
        assert isinstance(vec, Vector2D)
        assert vec.is_mutable is True
        vec.x = 1
        assert vec.x == 1
        vec_copy = vec.duplicate()
        assert vec == vec_copy
        vec_copy.x = 2
        assert vec != vec_copy

        vec_imm = vec.to_immutable()
        assert isinstance(vec_imm, Vector2DImmutable)
        assert vec_imm.is_mutable is False
        with pytest.raises(AttributeError):
            vec_imm.x = 2
        with pytest.raises(AttributeError):
            vec_imm.normalize()
        norm_vec = vec_imm.normalized()  # ensure operations tha yield new vectors are ok
        assert norm_vec.magnitude == pytest.approx(1., rel=1e-3)
        assert vec_imm.x == 1

        vec = Vector2DImmutable(0, 2)
        assert isinstance(vec, Vector2DImmutable)
        assert vec.is_mutable is False
        with pytest.raises(AttributeError):
            vec.x = 1
        assert vec.x == 0
        vec_copy = vec.duplicate()
        assert vec == vec_copy

        vec_mut = vec.to_mutable()
        assert isinstance(vec_mut, Vector2D)
        assert vec_mut.is_mutable is True
        vec_mut.x = 1
        assert vec_mut.x == 1

    def test_point2_mutability(self):
        """Test the mutability and immutability of Point2D objects."""
        pt = Point2D(0, 2)
        assert isinstance(pt, Point2D)
        assert pt.is_mutable is True
        pt.x = 1
        assert pt.x == 1
        pt_copy = pt.duplicate()
        assert pt == pt_copy
        pt_copy.x = 2
        assert pt != pt_copy

        pt_imm = pt.to_immutable()
        assert isinstance(pt_imm, Point2DImmutable)
        assert pt_imm.is_mutable is False
        with pytest.raises(AttributeError):
            pt_imm.x = 2
        norm_vec = pt_imm.normalized()  # ensure operations tha yield new vectors are ok
        assert norm_vec.magnitude == pytest.approx(1., rel=1e-3)
        assert pt_imm.x == 1

        pt = Point2DImmutable(0, 2)
        assert isinstance(pt, Point2DImmutable)
        assert pt.is_mutable is False
        with pytest.raises(AttributeError):
            pt.x = 1
        assert pt.x == 0
        pt_copy = pt.duplicate()
        assert pt == pt_copy

        pt_mut = pt.to_mutable()
        assert isinstance(pt_mut, Point2D)
        assert pt_mut.is_mutable is True
        pt_mut.x = 1
        assert pt_mut.x == 1

    def test_vector2_dot_cross_determinant_reverse(self):
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

        assert vec_1.reversed() == Vector2D(0, -2)
        assert vec_2.reversed() == Vector2D(-2, 0)

        vec_1.reverse()
        assert vec_1 == Vector2D(0, -2)

    def test_vector2_angle(self):
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

    def test_addition_subtraction(self):
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

    def test_multiplication_division(self):
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

    def test_move(self):
        """Test the Point2D move method."""
        pt_1 = Point2D(2, 2)
        vec_1 = Vector2D(0, 2)
        assert pt_1.move(vec_1) == Point2D(2, 4)

    def test_scale(self):
        """Test the Point2D scale method."""
        pt_1 = Point2D(2, 2)
        origin_1 = Point2D(0, 2)
        origin_2 = Point2D(1, 1)
        assert pt_1.scale(2, origin_1) == Point2D(4, 2)
        assert pt_1.scale(2, origin_2) == Point2D(3, 3)

    def test_scale_world_origin(self):
        """Test the Point2D scale_world_origin method."""
        pt_1 = Point2D(2, 2)
        pt_2 = Point2D(-2, -2)
        assert pt_1.scale_world_origin(2) == Point2D(4, 4)
        assert pt_1.scale_world_origin(0.5) == Point2D(1, 1)
        assert pt_1.scale_world_origin(-2) == Point2D(-4, -4)
        assert pt_2.scale_world_origin(2) == Point2D(-4, -4)
        assert pt_2.scale_world_origin(-2) == Point2D(4, 4)

    def test_rotate(self):
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

    def test_reflect(self):
        """Test the Point2D reflect method."""
        pt_1 = Point2D(2, 2)
        origin_1 = Point2D(0, 1)
        origin_2 = Point2D(1, 1)
        normal_1 = Vector2D(0, 1)
        normal_2 = Vector2D(-1, 1).normalized()

        assert pt_1.reflect(normal_1, origin_1) == Point2D(2, 0)
        assert pt_1.reflect(normal_1, origin_2) == Point2D(2, 0)
        assert pt_1.reflect(normal_2, origin_2) == Point2D(2, 2)

        test_1 = pt_1.reflect(normal_2, origin_1)
        assert test_1.x == pytest.approx(1, rel=1e-3)
        assert test_1.y == pytest.approx(3, rel=1e-3)


if __name__ == "__main__":
    unittest.main()
