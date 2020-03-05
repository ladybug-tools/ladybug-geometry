# coding=utf-8
import pytest

from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
from ladybug_geometry.geometry3d.ray import Ray3D

import math


def test_ray3d_init():
    """Test the initalization of Ray3D objects and basic properties."""
    pt = Point3D(2, 0, 2)
    vec = Vector3D(0, 2, 0)
    ray = Ray3D(pt, vec)
    str(ray)  # test the string representation of the ray

    assert ray.p == Point3D(2, 0, 2)
    assert ray.v == Vector3D(0, 2, 0)

    flip_ray = ray.reverse()
    assert flip_ray.p == Point3D(2, 0, 2)
    assert flip_ray.v == Vector3D(0, -2, 0)


def test_equality():
    """Test the equality of Ray3D objects."""
    pt = Point3D(2, 0, 2)
    vec = Vector3D(0, 2, 0)
    ray = Ray3D(pt, vec)
    ray_dup = ray.duplicate()
    ray_alt = Ray3D(Point3D(2, 0.1, 2), vec)

    assert ray is ray
    assert ray is not ray_dup
    assert ray == ray_dup
    assert hash(ray) == hash(ray_dup)
    assert ray != ray_alt
    assert hash(ray) != hash(ray_alt)


def test_ray3_to_from_dict():
    """Test the to/from dict of Ray3D objects."""
    pt = Point3D(2, 0, 2)
    vec = Vector3D(0, 2, 0)
    ray = Ray3D(pt, vec)
    ray_dict = ray.to_dict()
    new_ray = Ray3D.from_dict(ray_dict)
    assert isinstance(new_ray, Ray3D)
    assert new_ray.to_dict() == ray_dict


def test_ray3d_immutability():
    """Test the immutability of Ray3D objects."""
    pt = Point3D(2, 0, 0)
    vec = Vector3D(0, 2, 0)
    ray = Ray3D(pt, vec)

    assert isinstance(ray, Ray3D)
    with pytest.raises(AttributeError):
        ray.p.x = 3
    with pytest.raises(AttributeError):
        ray.v.x = 3
    with pytest.raises(AttributeError):
        ray.p = Point3D(0, 0, 0)
    with pytest.raises(AttributeError):
        ray.v = Vector3D(2, 2, 0)

    ray_copy = ray.duplicate()
    assert ray.p == ray_copy.p
    assert ray.v == ray_copy.v


def test_move():
    """Test the Ray3D move method."""
    pt = Point3D(2, 0, 2)
    vec = Vector3D(0, 2, 0)
    ray = Ray3D(pt, vec)

    vec_1 = Vector3D(2, 2, 2)
    new_ray = ray.move(vec_1)
    assert new_ray.p == Point3D(4, 2, 4)
    assert new_ray.v == vec
    assert new_ray.v.magnitude == 2


def test_scale():
    """Test the Ray3D scale method."""
    pt = Point3D(2, 2, 2)
    vec = Vector3D(0, 2, 0)
    ray = Ray3D(pt, vec)

    origin_1 = Point3D(0, 2, 2)
    origin_2 = Point3D(1, 1, 2)
    new_ray = ray.scale(2, origin_1)
    assert new_ray.p == Point3D(4, 2, 2)
    assert new_ray.v == Point3D(0, 4, 0)
    assert new_ray.v.magnitude == 4

    new_ray = ray.scale(2, origin_2)
    assert new_ray.p == Point3D(3, 3, 2)
    assert new_ray.v == Point3D(0, 4, 0)


def test_scale_world_origin():
    """Test the Ray3D scale method with None origin."""
    pt = Point3D(2, 2, 2)
    vec = Vector3D(0, 2, 0)
    ray = Ray3D(pt, vec)

    new_ray = ray.scale(2)
    assert new_ray.p == Point3D(4, 4, 4)
    assert new_ray.v == Point3D(0, 4)
    assert new_ray.v.magnitude == 4


def test_rotate():
    """Test the Ray3D rotate method."""
    pt = Point3D(2, 2, 2)
    vec = Vector3D(0, 2, 0)
    ray = Ray3D(pt, vec)
    origin_1 = Point3D(0, 0, 0)
    axis_1 = Vector3D(1, 0, 0)

    test_1 = ray.rotate(axis_1, math.pi, origin_1)
    assert test_1.p.x == pytest.approx(2, rel=1e-3)
    assert test_1.p.y == pytest.approx(-2, rel=1e-3)
    assert test_1.p.z == pytest.approx(-2, rel=1e-3)
    assert test_1.v.x == pytest.approx(0, rel=1e-3)
    assert test_1.v.y == pytest.approx(-2, rel=1e-3)
    assert test_1.v.z == pytest.approx(0, rel=1e-3)

    test_2 = ray.rotate(axis_1, math.pi/2, origin_1)
    assert test_2.p.x == pytest.approx(2, rel=1e-3)
    assert test_2.p.y == pytest.approx(-2, rel=1e-3)
    assert test_2.p.z == pytest.approx(2, rel=1e-3)
    assert test_2.v.x == pytest.approx(0, rel=1e-3)
    assert test_2.v.y == pytest.approx(0, rel=1e-3)
    assert test_2.v.z == pytest.approx(2, rel=1e-3)


def test_rotate_xy():
    """Test the Ray3D rotate_xy method."""
    pt = Point3D(2, 2, 2)
    vec = Vector3D(0, 2, 0)
    ray = Ray3D(pt, vec)
    origin_1 = Point3D(0, 2, 2)

    test_1 = ray.rotate_xy(math.pi, origin_1)
    assert test_1.p.x == pytest.approx(-2, rel=1e-3)
    assert test_1.p.y == pytest.approx(2, rel=1e-3)
    assert test_1.v.x == pytest.approx(0, rel=1e-3)
    assert test_1.v.y == pytest.approx(-2, rel=1e-3)

    test_2 = ray.rotate_xy(math.pi/2, origin_1)
    assert test_2.p.x == pytest.approx(0, rel=1e-3)
    assert test_2.p.y == pytest.approx(4, rel=1e-3)
    assert test_2.v.x == pytest.approx(-2, rel=1e-3)
    assert test_2.v.y == pytest.approx(0, rel=1e-3)


def test_reflect():
    """Test the Ray3D reflect method."""
    pt = Point3D(2, 2, 2)
    vec = Vector3D(0, 2, 0)
    ray = Ray3D(pt, vec)

    origin_1 = Point3D(0, 1, 2)
    origin_2 = Point3D(1, 1, 2)
    normal_1 = Vector3D(0, 1, 0)
    normal_2 = Vector3D(-1, 1, 0).normalize()

    assert ray.reflect(normal_1, origin_1).p == Point3D(2, 0, 2)
    assert ray.reflect(normal_1, origin_1).v == Vector3D(0, -2, 0)
    assert ray.reflect(normal_1, origin_2).p == Point3D(2, 0, 2)
    assert ray.reflect(normal_1, origin_2).v == Vector3D(0, -2, 0)

    test_1 = ray.reflect(normal_2, origin_2)
    assert test_1.p == Point3D(2, 2, 2)
    assert test_1.v.x == pytest.approx(2, rel=1e-3)
    assert test_1.v.y == pytest.approx(0, rel=1e-3)
    assert test_1.v.z == pytest.approx(0, rel=1e-3)

    test_2 = ray.reflect(normal_2, origin_1)
    assert test_2.p.x == pytest.approx(1, rel=1e-3)
    assert test_2.p.y == pytest.approx(3, rel=1e-3)
    assert test_2.p.z == pytest.approx(2, rel=1e-3)
    assert test_2.v.x == pytest.approx(2, rel=1e-3)
    assert test_2.v.y == pytest.approx(0, rel=1e-3)
    assert test_2.v.z == pytest.approx(0, rel=1e-3)


def test_closest_point():
    """Test the Ray3D closest_point method."""
    pt = Point3D(2, 2, 2)
    vec = Vector3D(0, 2, 0)
    ray = Ray3D(pt, vec)

    near_pt = Point3D(3, 3, 0)
    assert ray.closest_point(near_pt) == Point3D(2, 3, 2)
    near_pt = Point3D(2, 0, 0)
    assert ray.closest_point(near_pt) == Point3D(2, 2, 2)
    near_pt = Point3D(1, 5, 0)
    assert ray.closest_point(near_pt) == Point3D(2, 5, 2)


def test_distance_to_point():
    """Test the Ray3D distance_to_point method."""
    pt = Point3D(2, 2, 2)
    vec = Vector3D(0, 2, 0)
    ray = Ray3D(pt, vec)

    near_pt = Point3D(3, 3, 2)
    assert ray.distance_to_point(near_pt) == 1
    near_pt = Point3D(2, 0, 2)
    assert ray.distance_to_point(near_pt) == 2
    near_pt = Point3D(1, 5, 2)
    assert ray.distance_to_point(near_pt) == 1


def test_to_from_array():
    """Test to/from array method."""
    test_ray = Ray3D(Point3D(2, 0, 2), Vector3D(2, 2, 2))
    ray_array = ((2, 0, 2), (2, 2, 2))

    assert test_ray == Ray3D.from_array(ray_array)

    ray_array = ((2, 0, 2), (2, 2, 2))
    test_ray = Ray3D(Point3D(2, 0, 2), Vector3D(2, 2, 2))

    assert test_ray.to_array() == ray_array

    test_ray_2 = Ray3D.from_array(test_ray.to_array())
    assert test_ray == test_ray_2
