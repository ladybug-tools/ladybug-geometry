# coding=utf-8
import pytest

from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
from ladybug_geometry.geometry3d.plane import Plane
from ladybug_geometry.geometry3d.ray import Ray3D
from ladybug_geometry.geometry3d.line import LineSegment3D
from ladybug_geometry.geometry3d.arc import Arc3D

from ladybug_geometry.geometry2d.pointvector import Point2D

import math


def test_plane_init():
    """Test the initialization of Plane objects and basic properties."""
    pt = Point3D(2, 0, 2)
    vec = Vector3D(0, 2, 0)
    plane = Plane(vec, pt)
    str(plane)  # test the string representation
    hash(plane)

    assert plane.o == Point3D(2, 0, 2)
    assert plane.n == Vector3D(0, 1, 0)
    assert plane.x == Vector3D(1, 0, 0)
    assert plane.y == Vector3D(0, 0, -1)
    assert plane.k == 0

    plane_dup = plane.duplicate()
    assert plane_dup.o == Point3D(2, 0, 2)
    assert plane_dup.n == Vector3D(0, 1, 0)
    assert plane_dup.x == Vector3D(1, 0, 0)
    assert plane_dup.y == Vector3D(0, 0, -1)
    assert plane_dup.k == 0

    plane_flip = plane.flip()
    assert plane_flip.o == Point3D(2, 0, 2)
    assert plane_flip.n == Vector3D(0, -1, 0)
    assert plane_flip.x == Vector3D(1, 0, 0)
    assert plane_flip.y == Vector3D(0, 0, 1)
    assert plane_flip.k == 0


def test_equality():
    """Test the equality of Plane objects."""
    pt = Point3D(2, 0, 2)
    vec = Vector3D(0, 2, 0)
    plane = Plane(vec, pt)
    plane_dup = plane.duplicate()
    plane_alt = Plane(vec, Point3D(2, 0.1, 2))

    assert plane is plane
    assert plane is not plane_dup
    assert plane == plane_dup
    assert hash(plane) == hash(plane_dup)
    assert plane != plane_alt
    assert hash(plane) != hash(plane_alt)


def test_plane_to_from_dict():
    """Test the initialization of Plane objects and basic properties."""
    pt = Point3D(2, 0, 2)
    vec = Vector3D(0, 2, 0)
    plane = Plane(vec, pt)
    plane_dict = plane.to_dict()
    new_plane = Plane.from_dict(plane_dict)
    assert isinstance(new_plane, Plane)
    assert new_plane.to_dict() == plane_dict


def test_init_from_three_points():
    """Test the initialization of Plane from end points."""
    plane = Plane.from_three_points(Point3D(0, 0, 2), Point3D(0, 2, 2),
                                    Point3D(2, 2, 2))
    assert plane.o == Point3D(0, 0, 2)
    assert plane.n == Vector3D(0, 0, -1)
    assert plane.x == Vector3D(1, 0, 0)
    assert plane.y == Vector3D(0, -1, 0)
    assert plane.k == -2

    plane = Plane.from_three_points(Point3D(2, 2, 2), Point3D(0, 2, 2),
                                    Point3D(0, 0, 2))
    assert plane.o == Point3D(2, 2, 2)
    assert plane.n == Vector3D(0, 0, 1)
    assert plane.x == Vector3D(1, 0, 0)
    assert plane.y == Vector3D(0, 1, 0)
    assert plane.k == 2


def test_init_from_normal_k():
    """Test the initialization of Plane from end points."""
    plane = Plane.from_normal_k(Vector3D(0, 0, 1), 2)
    assert plane.o == Point3D(0, 0, 2)
    assert plane.n == Vector3D(0, 0, 1)
    assert plane.x == Vector3D(1, 0, 0)
    assert plane.y == Vector3D(0, 1, 0)
    assert plane.k == 2

    plane = Plane.from_normal_k(Vector3D(0, 0, 1), -2)
    assert plane.o == Point3D(0, 0, -2)
    assert plane.n == Vector3D(0, 0, 1)
    assert plane.x == Vector3D(1, 0, 0)
    assert plane.y == Vector3D(0, 1, 0)
    assert plane.k == -2


def test_linesegment3d_mutability():
    """Test the immutability of Plane objects."""
    pt = Point3D(2, 0, 0)
    vec = Vector3D(0, 2, 0)
    plane = Plane(vec, pt)

    assert isinstance(plane, Plane)
    with pytest.raises(AttributeError):
        plane.o.x = 3
    with pytest.raises(AttributeError):
        plane.n.x = 3
    with pytest.raises(AttributeError):
        plane.o = Point3D(0, 0, 0)
    with pytest.raises(AttributeError):
        plane.n = Vector3D(2, 2, 0)
    # ensure operations that yield new objects are ok
    plane_move = plane.move(Vector3D(-2, 0, 0))
    assert plane_move.o == Point3D(0, 0, 0)


def test_move():
    """Test the Plane move method."""
    pt = Point3D(2, 0, 2)
    vec = Vector3D(0, 2, 0)
    plane = Plane(vec, pt)

    vec_1 = Vector3D(2, 2, 2)
    new_plane = plane.move(vec_1)
    assert new_plane.o == Point3D(4, 2, 4)
    assert new_plane.n == Vector3D(0, 1, 0)
    assert new_plane.x == plane.x
    assert new_plane.y == plane.y
    assert new_plane.k == 2


def test_scale():
    """Test the Plane scale method."""
    pt = Point3D(2, 2, 2)
    vec = Vector3D(0, 2, 0)
    plane = Plane(vec, pt)

    origin_1 = Point3D(0, 2, 2)
    origin_2 = Point3D(1, 1, 2)
    new_plane = plane.scale(2, origin_1)
    assert new_plane.o == Point3D(4, 2, 2)
    assert new_plane.n == Point3D(0, 1, 0)
    assert new_plane.x == plane.x
    assert new_plane.y == plane.y
    assert new_plane.k == 2

    new_plane = plane.scale(2, origin_2)
    assert new_plane.o == Point3D(3, 3, 2)
    assert new_plane.n == Point3D(0, 1, 0)
    assert new_plane.x == plane.x
    assert new_plane.y == plane.y
    assert new_plane.k == 3


def test_scale_world_origin():
    """Test the Plane scale method with None origin."""
    pt = Point3D(2, 2, 2)
    vec = Vector3D(0, 2, 0)
    plane = Plane(vec, pt)

    new_plane = plane.scale(2)
    assert new_plane.o == Point3D(4, 4, 4)
    assert new_plane.n == Point3D(0, 1, 0)
    assert new_plane.x == plane.x
    assert new_plane.y == plane.y
    assert new_plane.k == 4


def test_rotate():
    """Test the Plane rotate method."""
    pt = Point3D(2, 2, 2)
    vec = Vector3D(0, 2, 0)
    plane = Plane(vec, pt)
    origin_1 = Point3D(0, 0, 0)
    axis_1 = Vector3D(1, 0, 0)

    test_1 = plane.rotate(axis_1, math.pi, origin_1)
    assert test_1.o.x == pytest.approx(2, rel=1e-3)
    assert test_1.o.y == pytest.approx(-2, rel=1e-3)
    assert test_1.o.z == pytest.approx(-2, rel=1e-3)
    assert test_1.n.x == pytest.approx(0, rel=1e-3)
    assert test_1.n.y == pytest.approx(-1, rel=1e-3)
    assert test_1.n.z == pytest.approx(0, rel=1e-3)
    assert test_1.x.x == pytest.approx(1, rel=1e-3)
    assert test_1.x.y == pytest.approx(0, rel=1e-3)
    assert test_1.x.z == pytest.approx(0, rel=1e-3)
    assert test_1.y.x == pytest.approx(0, rel=1e-3)
    assert test_1.y.y == pytest.approx(0, rel=1e-3)
    assert test_1.y.z == pytest.approx(1, rel=1e-3)
    assert test_1.k == pytest.approx(2, rel=1e-3)

    test_2 = plane.rotate(axis_1, math.pi/2, origin_1)
    assert test_2.o.x == pytest.approx(2, rel=1e-3)
    assert test_2.o.y == pytest.approx(-2, rel=1e-3)
    assert test_2.o.z == pytest.approx(2, rel=1e-3)
    assert test_2.n.x == pytest.approx(0, rel=1e-3)
    assert test_2.n.y == pytest.approx(0, rel=1e-3)
    assert test_2.n.z == pytest.approx(1, rel=1e-3)
    assert test_2.x.x == pytest.approx(1, rel=1e-3)
    assert test_2.x.y == pytest.approx(0, rel=1e-3)
    assert test_2.x.z == pytest.approx(0, rel=1e-3)
    assert test_2.y.x == pytest.approx(0, rel=1e-3)
    assert test_2.y.y == pytest.approx(1, rel=1e-3)
    assert test_2.y.z == pytest.approx(0, rel=1e-3)
    assert test_2.k == pytest.approx(2, rel=1e-3)


def test_rotate_xy():
    """Test the Plane rotate_xy method."""
    pt = Point3D(2, 2, 2)
    vec = Vector3D(0, 2, 0)
    plane = Plane(vec, pt)
    origin_1 = Point3D(0, 2, 2)

    test_1 = plane.rotate_xy(math.pi, origin_1)
    assert test_1.o.x == pytest.approx(-2, rel=1e-3)
    assert test_1.o.y == pytest.approx(2, rel=1e-3)
    assert test_1.o.z == pytest.approx(2, rel=1e-3)
    assert test_1.n.x == pytest.approx(0, rel=1e-3)
    assert test_1.n.y == pytest.approx(-1, rel=1e-3)
    assert test_1.n.z == pytest.approx(0, rel=1e-3)
    assert test_1.x.x == pytest.approx(-1, rel=1e-3)
    assert test_1.x.y == pytest.approx(0, rel=1e-3)
    assert test_1.x.z == pytest.approx(0, rel=1e-3)
    assert test_1.y.x == pytest.approx(0, rel=1e-3)
    assert test_1.y.y == pytest.approx(0, rel=1e-3)
    assert test_1.y.z == pytest.approx(-1, rel=1e-3)
    assert test_1.k == pytest.approx(-2, rel=1e-3)

    test_2 = plane.rotate_xy(math.pi/2, origin_1)
    assert test_2.o.x == pytest.approx(0, rel=1e-3)
    assert test_2.o.y == pytest.approx(4, rel=1e-3)
    assert test_2.o.z == pytest.approx(2, rel=1e-3)
    assert test_2.n.x == pytest.approx(-1, rel=1e-3)
    assert test_2.n.y == pytest.approx(0, rel=1e-3)
    assert test_2.n.z == pytest.approx(0, rel=1e-3)
    assert test_2.x.x == pytest.approx(0, rel=1e-3)
    assert test_2.x.y == pytest.approx(1, rel=1e-3)
    assert test_2.x.z == pytest.approx(0, rel=1e-3)
    assert test_2.y.x == pytest.approx(0, rel=1e-3)
    assert test_2.y.y == pytest.approx(0, rel=1e-3)
    assert test_2.y.z == pytest.approx(-1, rel=1e-3)
    assert test_2.k == pytest.approx(0, rel=1e-3)


def test_reflect():
    """Test the Plane reflect method."""
    pt = Point3D(2, 2, 2)
    vec = Vector3D(0, 2, 0)
    plane = Plane(vec, pt)

    origin_1 = Point3D(0, 1, 2)
    origin_2 = Point3D(1, 1, 2)
    normal_1 = Vector3D(0, 1, 0)
    normal_2 = Vector3D(-1, 1, 0).normalize()

    assert plane.reflect(normal_1, origin_1).o == Point3D(2, 0, 2)
    assert plane.reflect(normal_1, origin_1).n == Vector3D(0, -1, 0)
    assert plane.reflect(normal_1, origin_2).o == Point3D(2, 0, 2)
    assert plane.reflect(normal_1, origin_2).n == Vector3D(0, -1, 0)

    test_1 = plane.reflect(normal_2, origin_2)
    assert test_1.o == Point3D(2, 2, 2)
    assert test_1.n.x == pytest.approx(1, rel=1e-3)
    assert test_1.n.y == pytest.approx(0, rel=1e-3)
    assert test_1.n.z == pytest.approx(0, rel=1e-3)

    test_2 = plane.reflect(normal_2, origin_1)
    assert test_2.o.x == pytest.approx(1, rel=1e-3)
    assert test_2.o.y == pytest.approx(3, rel=1e-3)
    assert test_2.o.z == pytest.approx(2, rel=1e-3)
    assert test_2.n.x == pytest.approx(1, rel=1e-3)
    assert test_2.n.y == pytest.approx(0, rel=1e-3)
    assert test_2.n.z == pytest.approx(0, rel=1e-3)


def test_xyz_to_xy():
    """Test the xyz_to_xy method."""
    pt = Point3D(2, 2, 2)
    vec = Vector3D(0, 0, 2)
    plane = Plane(vec, pt)

    test_pt = Point3D(4, 4, 2)
    assert plane.xyz_to_xy(test_pt) == Point2D(2, 2)
    test_pt = Point3D(0, 0, 2)
    assert plane.xyz_to_xy(test_pt) == Point2D(-2, -2)
    assert isinstance(plane.xyz_to_xy(test_pt), Point2D)


def test_xy_to_xyz():
    """Test the xy_to_xyz method."""
    pt = Point3D(2, 2, 2)
    vec = Vector3D(0, 0, 2)
    plane = Plane(vec, pt)

    test_pt = Point2D(2, 2)
    assert plane.xy_to_xyz(test_pt) == Point3D(4, 4, 2)
    test_pt = Point2D(-1, -1)
    assert plane.xy_to_xyz(test_pt) == Point3D(1, 1, 2)
    assert isinstance(plane.xy_to_xyz(test_pt), Point3D)


def test_is_point_above():
    """Test the Plane is_point_above method."""
    plane = Plane()
    assert plane.is_point_above(Point3D(0, 0, 1))
    assert not plane.is_point_above(Point3D(0, 0, -1))

    plane = Plane(n=Vector3D(1, 0, 0))
    assert not plane.is_point_above(Point3D(-1, 0, 1))
    assert plane.is_point_above(Point3D(1, 0, -1))


def test_closest_point():
    """Test the Plane closest_point method."""
    pt = Point3D(2, 2, 2)
    vec = Vector3D(0, 2, 0)
    plane = Plane(vec, pt)

    test_pt = Point3D(0, 4, 0)
    assert plane.closest_point(test_pt) == Point3D(0, 2, 0)
    test_pt = Point3D(4, 4, 4)
    assert plane.closest_point(test_pt) == Point3D(4, 2, 4)


def test_distance_to_point():
    """Test the Plane distance_to_point method."""
    pt = Point3D(2, 2, 2)
    vec = Vector3D(0, 2, 0)
    plane = Plane(vec, pt)

    test_pt = Point3D(0, 4, 0)
    assert plane.distance_to_point(test_pt) == 2
    test_pt = Point3D(4, 4, 4)
    assert plane.distance_to_point(test_pt) == 2
    test_pt = Point3D(4, 2, 4)
    assert plane.distance_to_point(test_pt) == 0


def test_closest_points_between_line():
    """Test the Plane closest_points_between_line method."""
    pt = Point3D(2, 2, 2)
    vec = Vector3D(0, 2, 0)
    plane = Plane(vec, pt)

    test_seg = LineSegment3D(Point3D(0, 4, 0), Vector3D(-1, -1, -1))
    pt_1, pt_2 = plane.closest_points_between_line(test_seg)
    assert pt_1 == Point3D(-1, 3, -1)
    assert pt_2 == Point3D(-1, 2, -1)

    test_seg = LineSegment3D(Point3D(0, 0, 0), Vector3D(-1, -1, -1))
    pt_1, pt_2 = plane.closest_points_between_line(test_seg)
    assert pt_1 == Point3D(0, 0, 0)
    assert pt_2 == Point3D(0, 2, 0)

    test_seg = LineSegment3D(Point3D(0, 0, 0), Vector3D(3, 3, 3))
    assert plane.closest_points_between_line(test_seg) is None


def test_distance_to_line():
    """Test the Plane distance_to_line method."""
    pt = Point3D(2, 2, 2)
    vec = Vector3D(0, 2, 0)
    plane = Plane(vec, pt)

    test_seg = LineSegment3D(Point3D(0, 4, 0), Vector3D(1, 1, 1))
    assert plane.distance_to_line(test_seg) == 2

    test_seg = LineSegment3D(Point3D(0, 0, 0), Vector3D(1, 1, 1))
    assert plane.distance_to_line(test_seg) == 1

    test_seg = LineSegment3D(Point3D(0, 0, 0), Vector3D(3, 3, 3))
    assert plane.distance_to_line(test_seg) == 0


def test_intersect_line_ray():
    """Test the Plane intersect_line_ray method."""
    pt = Point3D(2, 2, 2)
    vec = Vector3D(0, 2, 0)
    plane = Plane(vec, pt)

    test_seg = LineSegment3D(Point3D(0, 4, 0), Vector3D(1, 1, 1))
    assert plane.intersect_line_ray(test_seg) is None

    test_seg = LineSegment3D(Point3D(0, 0, 0), Vector3D(3, 3, 3))
    assert plane.intersect_line_ray(test_seg) == Point3D(2, 2, 2)

    test_ray = Ray3D(Point3D(0, 4, 0), Vector3D(1, 1, 1))
    assert plane.intersect_line_ray(test_ray) is None

    test_ray = Ray3D(Point3D(0, 4, 0), Vector3D(-1, -1, -1))
    assert plane.intersect_line_ray(test_ray) == Point3D(-2, 2, -2)


def test_intersect_arc():
    """Test the Plane intersect_arc method."""
    pt = Point3D(2, 0, 2)
    arc = Arc3D(Plane(o=pt), 1, 0, math.pi)
    circle = Arc3D(Plane(o=pt), 1)

    plane_1 = Plane(Vector3D(0, 1, 0), Point3D(0, 0, 0))
    int1 = plane_1.intersect_arc(circle)
    assert len(int1) == 2
    assert int1[0] == Point3D(3, 0, 2)

    plane_2 = Plane(Vector3D(0, 1, 0), Point3D(0, 0.5, 0))
    int2 = plane_2.intersect_arc(arc)
    assert len(int2) == 2

    plane_3 = Plane(Vector3D(1, 0, 0), Point3D(1.5, 0, 0))
    int3 = plane_3.intersect_arc(arc)
    assert len(int3) == 1

    plane_4 = Plane(Vector3D(1, 0, 0), Point3D(0, 2, 0))
    int4 = plane_4.intersect_arc(arc)
    assert int4 is None


def test_intersect_plane():
    """Test the Plane intersect_plane method."""
    pt_1 = Point3D(2, 2, 2)
    vec_1 = Vector3D(0, 2, 0)
    plane_1 = Plane(vec_1, pt_1)
    pt_2 = Point3D(0, 0, 0)
    vec_2 = Vector3D(2, 0, 0)
    plane_2 = Plane(vec_2, pt_2)
    pt_3 = Point3D(0, 0, 0)
    vec_3 = Vector3D(0, 2, 0)
    plane_3 = Plane(vec_3, pt_3)

    assert plane_1.intersect_plane(plane_2) == Ray3D(
        Point3D(0, 2, 0), Vector3D(0, 0, -1))
    assert plane_1.intersect_plane(plane_3) is None
    assert plane_2.intersect_plane(plane_3) == Ray3D(
        Point3D(0, 0, 0), Vector3D(0, 0, 1))


def test_is_coplanar():
    """Test the Plane is_coplanar method."""
    pt_1 = Point3D(2, 2, 2)
    vec_1 = Vector3D(0, 2, 0)
    plane_1 = Plane(vec_1, pt_1)
    pt_2 = Point3D(0, 0, 0)
    vec_2 = Vector3D(2, 0, 0)
    plane_2 = Plane(vec_2, pt_2)
    pt_3 = Point3D(0, 0, 0)
    vec_3 = Vector3D(0, 2, 0)
    plane_3 = Plane(vec_3, pt_3)
    pt_4 = Point3D(0, 2, 0)
    vec_4 = Vector3D(0, 2, 0)
    plane_4 = Plane(vec_4, pt_4)
    pt_5 = Point3D(0, 2, 0)
    vec_5 = Vector3D(0, -2, 0)
    plane_5 = Plane(vec_5, pt_5)

    assert plane_1.is_coplanar(plane_2) is False
    assert plane_1.is_coplanar(plane_3) is False
    assert plane_1.is_coplanar(plane_4) is True
    assert plane_1.is_coplanar(plane_5) is True


def test_is_coplanar_tolerance():
    """Test the Plane is_coplanar_tolerance method."""
    pt_1 = Point3D(2, 2, 2)
    vec_1 = Vector3D(0, 2, 0)
    plane_1 = Plane(vec_1, pt_1)
    pt_2 = Point3D(0, 0, 0)
    vec_2 = Vector3D(2, 0, 0)
    plane_2 = Plane(vec_2, pt_2)
    pt_3 = Point3D(0, 0, 0)
    vec_3 = Vector3D(0, 2, 0)
    plane_3 = Plane(vec_3, pt_3)
    pt_4 = Point3D(0, 2, 0)
    vec_4 = Vector3D(0, 2, 0)
    plane_4 = Plane(vec_4, pt_4)
    pt_5 = Point3D(0, 2, 0)
    vec_5 = Vector3D(0, -2, 0)
    plane_5 = Plane(vec_5, pt_5)

    assert plane_1.is_coplanar_tolerance(plane_2, 0.0001, 0.0001) is False
    assert plane_1.is_coplanar_tolerance(plane_3, 0.0001, 0.0001) is False
    assert plane_1.is_coplanar_tolerance(plane_4, 0.0001, 0.0001) is True
    assert plane_1.is_coplanar_tolerance(plane_5, 0.0001, 0.0001) is True
