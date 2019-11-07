# coding=utf-8
# import pytest

from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
from ladybug_geometry.geometry3d.line import LineSegment3D
from ladybug_geometry.geometry3d.sphere import Sphere
from ladybug_geometry.geometry3d.plane import Plane


def test_sphere_init():
    """Test the initalization of Sphere objects and basic properties."""
    pt = Point3D(2, 0, 2)
    r = 3
    sp = Sphere(pt, r)
    str(sp)  # test the string representation of the line segment

    assert sp.center == Point3D(2, 0, 2)
    assert sp.radius == 3


def test_sphere_intersection_with_line_ray():
    """Test the Sphere intersect_line_ray method."""
    lpt = Point3D(1.5, 0, -3)
    vec = Vector3D(0, 0, 8)
    seg = LineSegment3D(lpt, vec)
    spt = Point3D(0, 0, 0)
    sp = Sphere(spt, 1.5)

    int1 = sp.intersect_line_ray(seg)
    assert len(int1) == 1, "{}".format(int1)


def test_sphere_intersection_with_plane():
    """Test the Sphere intersect_plane method."""
    ppt = Point3D(-1.5, 0, 0)
    vec = Vector3D(1, 0, 1)
    pl = Plane(vec, ppt)
    print pl
    spt = Point3D(0, 0, 0)
    sp = Sphere(spt, 1.5)

    int1 = sp.intersect_plane(pl)
    assert False, int1
