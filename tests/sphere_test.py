# coding=utf-8
# import pytest

from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
from ladybug_geometry.geometry3d.sphere import Sphere3D
from ladybug_geometry.geometry3d.plane import Plane
from ladybug_geometry.geometry3d.line import LineSegment3D
from ladybug_geometry.geometry3d.arc import Arc3D


def test_sphere_init():
    """Test the initalization of Sphere3D objects and basic properties."""
    pt = Point3D(2, 0, 2)
    r = 3
    sp = Sphere3D(pt, r)
    str(sp)  # test the string representation of the line segment

    assert sp.center == Point3D(2, 0, 2)
    assert sp.radius == 3
    assert sp.to_dict()['center'] == (2, 0, 2)


def test_sphere_intersection_with_line_ray():
    """Test the Sphere3D intersect_line_ray method."""
    lpt = Point3D(-2, 0, 0)
    vec = Vector3D(4, 0, 0)
    seg = LineSegment3D(lpt, vec)
    spt = Point3D(0, 0, 0)
    sp = Sphere3D(spt, 1.5)

    int1 = sp.intersect_line_ray(seg)
    assert isinstance(int1, LineSegment3D)
    assert int1.p == Point3D(1.5, 0, 0)


def test_sphere_intersection_with_plane():
    """Test the Sphere3D intersect_plane method."""
    ppt = Point3D(-1.5, 0, 1.46)
    vec = Vector3D(0.1, 0, 1)
    pl = Plane(vec, ppt)
    spt = Point3D(0, 0, 0)
    sp = Sphere3D(spt, 1.5)
    int1 = sp.intersect_plane(pl)
    assert isinstance(int1, Arc3D)

    ppt = Point3D(0, 0, 0)
    vec = Vector3D(0, 0, 1)
    pl = Plane(vec, ppt)
    int1 = sp.intersect_plane(pl)
    assert int1.c == ppt
    assert int1.radius == 1.5
