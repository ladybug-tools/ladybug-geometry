# coding=utf-8

from ladybug_geometry.geometry3d.pointvector import Vector3D, Point3D, \
    Vector3DImmutable, Point3DImmutable

import unittest
import pytest
import math


class Point3DTestCase(unittest.TestCase):
    """Test for Vector3D, Point3D"""

    def test_vector3_init(self):
        """Test the initalization of Vector3D objects and basic properties."""
        vec = Vector3D(0, 2, 0)
        str(vec)  # test the string representation of the vector

        assert vec.x == 0
        assert vec.y == 2
        assert vec.z == 0
        assert vec.magnitude == 2
        assert vec.magnitude_squared == 4

        norm_vec = vec.normalized()
        assert norm_vec.x == 0
        assert norm_vec.z == 0
        assert norm_vec.magnitude == 1

        assert vec.magnitude == 2
        vec.normalize()
        assert vec.magnitude == 1

    def test_point3_init(self):
        """Test the initalization of Point3D objects and basic properties."""
        pt_1 = Point3D(0, 2, 0)
        str(pt_1)  # test the string representation of the vector
        assert pt_1.x == 0
        assert pt_1.y == 2
        assert pt_1.z == 0

        pt_2 = Point3D(2, 2)
        assert pt_1.distance_to_point(pt_2) == 2

    def test_vector3_mutability(self):
        """Test the mutability and immutability of Vector3D objects."""
        vec = Vector3D(0, 2, 0)
        assert isinstance(vec, Vector3D)
        assert vec.is_mutable is True
        vec.x = 1
        assert vec.x == 1
        vec_copy = vec.duplicate()
        assert vec == vec_copy
        vec_copy.x = 2
        assert vec != vec_copy

        vec_imm = vec.to_immutable()
        assert isinstance(vec_imm, Vector3DImmutable)
        assert vec_imm.is_mutable is False
        with pytest.raises(AttributeError):
            vec_imm.x = 2
        with pytest.raises(AttributeError):
            vec_imm.normalize()
        norm_vec = vec_imm.normalized()  # ensure operations tha yield new vectors are ok
        assert norm_vec.magnitude == pytest.approx(1., rel=1e-3)
        assert vec_imm.x == 1

        vec = Vector3DImmutable(0, 2, 0)
        assert isinstance(vec, Vector3DImmutable)
        assert vec.is_mutable is False
        with pytest.raises(AttributeError):
            vec.x = 1
        assert vec.x == 0
        vec_copy = vec.duplicate()
        assert vec == vec_copy

        vec_mut = vec.to_mutable()
        assert isinstance(vec_mut, Vector3D)
        assert vec_mut.is_mutable is True
        vec_mut.x = 1
        assert vec_mut.x == 1

    def test_point3_mutability(self):
        """Test the mutability and immutability of Point3D objects."""
        pt = Point3D(0, 2, 0)
        assert isinstance(pt, Point3D)
        assert pt.is_mutable is True
        pt.x = 1
        assert pt.x == 1
        pt_copy = pt.duplicate()
        assert pt == pt_copy
        pt_copy.x = 2
        assert pt != pt_copy

        pt_imm = pt.to_immutable()
        assert isinstance(pt_imm, Point3DImmutable)
        assert pt_imm.is_mutable is False
        with pytest.raises(AttributeError):
            pt_imm.x = 2
        norm_vec = pt_imm.normalized()  # ensure operations tha yield new vectors are ok
        assert norm_vec.magnitude == pytest.approx(1., rel=1e-3)
        assert pt_imm.x == 1

        pt = Point3DImmutable(0, 2, 0)
        assert isinstance(pt, Point3DImmutable)
        assert pt.is_mutable is False
        with pytest.raises(AttributeError):
            pt.x = 1
        assert pt.x == 0
        pt_copy = pt.duplicate()
        assert pt == pt_copy

        pt_mut = pt.to_mutable()
        assert isinstance(pt_mut, Point3D)
        assert pt_mut.is_mutable is True
        pt_mut.x = 1
        assert pt_mut.x == 1

    def test_vector3_angle(self):
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

    def test_addition_subtraction(self):
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

    def test_move(self):
        """Test the Point3D move method."""
        pt_1 = Point3D(2, 2, 0)
        vec_1 = Vector3D(0, 2, 2)
        assert pt_1.move(vec_1) == Point3D(2, 4, 2)

    def test_scale(self):
        """Test the Point3D scale method."""
        pt_1 = Point3D(2, 2, 2)
        origin_1 = Point3D(0, 2, 2)
        origin_2 = Point3D(1, 1, 1)
        assert pt_1.scale(2, origin_1) == Point3D(4, 2, 2)
        assert pt_1.scale(2, origin_2) == Point3D(3, 3, 3)

    def test_rotate(self):
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

    def test_rotate_xy(self):
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

    def test_reflect(self):
        """Test the Point3D reflect method."""
        pt_1 = Point3D(2, 2, 2)
        origin_1 = Point3D(0, 1, 0)
        origin_2 = Point3D(1, 1, 0)
        normal_1 = Vector3D(0, 1, 0)
        normal_2 = Vector3D(-1, 1).normalized()

        assert pt_1.reflect(normal_1, origin_1) == Point3D(2, 0, 2)
        assert pt_1.reflect(normal_1, origin_2) == Point3D(2, 0, 2)
        assert pt_1.reflect(normal_2, origin_2) == Point3D(2, 2, 2)

        test_1 = pt_1.reflect(normal_2, origin_1)
        assert test_1.x == pytest.approx(1, rel=1e-3)
        assert test_1.y == pytest.approx(3, rel=1e-3)
        assert test_1.z == pytest.approx(2, rel=1e-3)

    def test_project(self):
        """Test the Point3D project method."""
        pt_1 = Point3D(2, 2, 2)
        origin_1 = Point3D(1, 0, 0)
        origin_2 = Point3D(0, 1, 0)
        normal_1 = Vector3D(0, 1, 0)

        assert pt_1.project(normal_1, origin_1) == Point3D(2, 0, 2)
        assert pt_1.project(normal_1, origin_2) == Point3D(2, 1, 2)


if __name__ == "__main__":
    unittest.main()
