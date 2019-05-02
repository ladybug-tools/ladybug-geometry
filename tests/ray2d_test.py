# coding=utf-8

from ladybug_geometry.geometry2d.pointvector import Point2D, Vector2D
from ladybug_geometry.geometry2d.ray import Ray2D, Ray2DImmutable

import unittest
import pytest
import math


class Ray2DTestCase(unittest.TestCase):
    """Test for Ray2D"""

    def test_ray2d_init(self):
        """Test the initalization of Ray2D objects and basic properties."""
        pt = Point2D(2, 0)
        vec = Vector2D(0, 2)
        ray = Ray2D(pt, vec)
        str(ray)  # test the string representation of the ray

        assert ray.p == Point2D(2, 0)
        assert ray.v == Vector2D(0, 2)

        flip_ray = ray.reversed()
        assert flip_ray.p == Point2D(2, 0)
        assert flip_ray.v == Vector2D(0, -2)

        assert ray.p == Point2D(2, 0)
        ray.reverse()
        assert ray.p == Point2D(2, 0)
        assert ray.v == Vector2D(0, -2)

    def test_ray2d_mutability(self):
        """Test the mutability and immutability of Ray2D objects."""
        pt = Point2D(2, 0)
        vec = Vector2D(0, 2)
        ray = Ray2D(pt, vec)

        assert isinstance(ray, Ray2D)
        assert ray.is_mutable is True
        ray.p = Point2D(0, 0)
        assert ray.p == Point2D(0, 0)
        ray.p.x = 1
        assert ray.p == Point2D(1, 0)

        ray_imm = ray.to_immutable()
        assert isinstance(ray_imm, Ray2DImmutable)
        assert ray_imm.is_mutable is False
        with pytest.raises(AttributeError):
            ray_imm.p.x = 3
        with pytest.raises(AttributeError):
            ray_imm.v.x = 3
        with pytest.raises(AttributeError):
            ray_imm.p = Point2D(0, 0)
        with pytest.raises(AttributeError):
            ray_imm.v = Vector2D(2, 2)
        ray_move = ray_imm.move(Vector2D(-1, 0))  # ensure operations that yield new objects are ok
        assert ray_move.p == Point2D(0, 0)

        ray = Ray2DImmutable(pt, vec)
        assert isinstance(ray, Ray2DImmutable)
        assert ray.is_mutable is False
        with pytest.raises(AttributeError):
            ray_imm.p.x = 3
        assert ray.p == Point2D(2, 0)
        ray_copy = ray.duplicate()
        assert ray.p == ray_copy.p
        assert ray.v == ray_copy.v

        ray_mut = ray.to_mutable()
        assert isinstance(ray_mut, Ray2D)
        assert ray_mut.is_mutable is True
        ray_mut.p.x = 1
        assert ray_mut.p.x == 1

    def test_move(self):
        """Test the Ray2D move method."""
        pt = Point2D(2, 0)
        vec = Vector2D(0, 2)
        ray = Ray2D(pt, vec)

        vec_1 = Vector2D(2, 2)
        new_ray = ray.move(vec_1)
        assert new_ray.p == Point2D(4, 2)
        assert new_ray.v == vec

    def test_scale(self):
        """Test the Ray2D scale method."""
        pt = Point2D(2, 2)
        vec = Vector2D(0, 2)
        ray = Ray2D(pt, vec)

        origin_1 = Point2D(0, 2)
        origin_2 = Point2D(1, 1)
        new_ray = ray.scale(2, origin_1)
        assert new_ray.p == Point2D(4, 2)
        assert new_ray.v == Point2D(0, 4)
        assert new_ray.v.magnitude == 4

        new_ray = ray.scale(2, origin_2)
        assert new_ray.p == Point2D(3, 3)
        assert new_ray.v == Point2D(0, 4)

    def test_scale_world_origin(self):
        """Test the Ray2D scale_world_origin method."""
        pt = Point2D(2, 2)
        vec = Vector2D(0, 2)
        ray = Ray2D(pt, vec)

        new_ray = ray.scale_world_origin(2)
        assert new_ray.p == Point2D(4, 4)
        assert new_ray.v == Point2D(0, 4)
        assert new_ray.v.magnitude == 4

    def test_rotate(self):
        """Test the Ray2D rotate method."""
        pt = Point2D(2, 2)
        vec = Vector2D(0, 2)
        ray = Ray2D(pt, vec)
        origin_1 = Point2D(0, 2)

        test_1 = ray.rotate(math.pi, origin_1)
        assert test_1.p.x == pytest.approx(-2, rel=1e-3)
        assert test_1.p.y == pytest.approx(2, rel=1e-3)
        assert test_1.v.x == pytest.approx(0, rel=1e-3)
        assert test_1.v.y == pytest.approx(-2, rel=1e-3)

        test_2 = ray.rotate(math.pi/2, origin_1)
        assert test_2.p.x == pytest.approx(0, rel=1e-3)
        assert test_2.p.y == pytest.approx(4, rel=1e-3)
        assert test_2.v.x == pytest.approx(-2, rel=1e-3)
        assert test_2.v.y == pytest.approx(0, rel=1e-3)

    def test_reflect(self):
        """Test the Ray2D reflect method."""
        pt = Point2D(2, 2)
        vec = Vector2D(0, 2)
        ray = Ray2D(pt, vec)

        origin_1 = Point2D(0, 1)
        origin_2 = Point2D(1, 1)
        normal_1 = Vector2D(0, 1)
        normal_2 = Vector2D(-1, 1).normalized()

        assert ray.reflect(normal_1, origin_1).p == Point2D(2, 0)
        assert ray.reflect(normal_1, origin_1).v == Vector2D(0, -2)
        assert ray.reflect(normal_1, origin_2).p == Point2D(2, 0)
        assert ray.reflect(normal_1, origin_2).v == Vector2D(0, -2)

        test_1 = ray.reflect(normal_2, origin_2)
        assert test_1.p == Point2D(2, 2)
        assert test_1.v.x == pytest.approx(2, rel=1e-3)
        assert test_1.v.y == pytest.approx(0, rel=1e-3)

        test_2 = ray.reflect(normal_2, origin_1)
        assert test_2.p.x == pytest.approx(1, rel=1e-3)
        assert test_2.p.y == pytest.approx(3, rel=1e-3)
        assert test_1.v.x == pytest.approx(2, rel=1e-3)
        assert test_1.v.y == pytest.approx(0, rel=1e-3)

    def test_closest_point(self):
        """Test the Ray2D closest_point method."""
        pt = Point2D(2, 2)
        vec = Vector2D(0, 2)
        ray = Ray2D(pt, vec)

        near_pt = Point2D(3, 3)
        assert ray.closest_point(near_pt) == Point2D(2, 3)
        near_pt = Point2D(2, 0)
        assert ray.closest_point(near_pt) == Point2D(2, 2)
        near_pt = Point2D(1, 5)
        assert ray.closest_point(near_pt) == Point2D(2, 5)

    def test_distance_to_point(self):
        """Test the Ray2D distance_to_point method."""
        pt = Point2D(2, 2)
        vec = Vector2D(0, 2)
        ray = Ray2D(pt, vec)

        near_pt = Point2D(3, 3)
        assert ray.distance_to_point(near_pt) == 1
        near_pt = Point2D(2, 0)
        assert ray.distance_to_point(near_pt) == 2
        near_pt = Point2D(1, 5)
        assert ray.distance_to_point(near_pt) == 1

    def test_intersect_line_ray(self):
        """Test the Ray2D distance_to_point method."""
        pt_1 = Point2D(2, 2)
        vec_1 = Vector2D(0, 2)
        ray_1 = Ray2D(pt_1, vec_1)

        pt_2 = Point2D(0, 3)
        vec_2 = Vector2D(4, 0)
        ray_2 = Ray2D(pt_2, vec_2)

        pt_3 = Point2D(0, 0)
        vec_3 = Vector2D(1, 1)
        ray_3 = Ray2D(pt_3, vec_3)

        assert ray_1.intersect_line_ray(ray_2) == Point2D(2, 3)
        assert ray_1.intersect_line_ray(ray_3) == Point2D(2, 2)

if __name__ == "__main__":
    unittest.main()
