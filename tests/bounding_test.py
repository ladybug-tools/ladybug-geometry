from ladybug_geometry.geometry2d.pointvector import Point2D, Vector2D
from ladybug_geometry.geometry2d.polygon import Polygon2D
from ladybug_geometry.geometry3d.pointvector import Point3D
from ladybug_geometry.geometry3d.plane import Plane
from ladybug_geometry.geometry3d.polyface import Polyface3D
from ladybug_geometry.bounding import bounding_box, bounding_box_extents, \
    bounding_rectangle, bounding_rectangle_extents

import pytest


def test_bounding_box():
    """Test the bounding_box methods with arrays of 3D objects."""
    plane1 = Plane(o=Point3D(-5, 0, 0))
    plane2 = Plane(o=Point3D(0, -4, 4))
    plane3 = Plane(o=Point3D(2, 2, -4))
    polyface1 = Polyface3D.from_box(2, 4, 2, plane1)
    polyface2 = Polyface3D.from_box(2, 4, 2, plane2)
    polyface3 = Polyface3D.from_box(2, 4, 2, plane3)

    min_pt, max_pt = bounding_box([polyface1, polyface2, polyface3])
    assert min_pt == Point3D(-5, -4, -4)
    assert max_pt == Point3D(4, 6, 6)

    x_dom, y_dom, z_dom = bounding_box_extents([polyface1, polyface2, polyface3])
    assert x_dom == 9
    assert y_dom == 10
    assert z_dom == 10


def test_bounding_box_angle():
    """Test the bounding_box methods with an axis_angle."""
    plane1 = Plane(o=Point3D(-5, 0, 0))
    plane2 = Plane(o=Point3D(0, -4, 4))
    plane3 = Plane(o=Point3D(2, 2, -4))
    polyface1 = Polyface3D.from_box(2, 4, 2, plane1)
    polyface2 = Polyface3D.from_box(2, 4, 2, plane2)
    polyface3 = Polyface3D.from_box(2, 4, 2, plane3)

    min_pt, max_pt = bounding_box([polyface1, polyface2, polyface3], 45)
    assert min_pt.x == pytest.approx(1.45, rel=1e-2)
    assert min_pt.y == pytest.approx(-4.89, rel=1e-2)
    assert min_pt.z == pytest.approx(-4, rel=1e-2)
    assert max_pt.x == pytest.approx(-1.62, rel=1e-2)
    assert max_pt.y == pytest.approx(9.47, rel=1e-2)
    assert max_pt.z == pytest.approx(6, rel=1e-2)

    x_dom, y_dom, z_dom = bounding_box_extents([polyface1, polyface2, polyface3], 45)
    assert x_dom == pytest.approx(10.6103, rel=1e-2)
    assert y_dom == pytest.approx(10.1589, rel=1e-2)
    assert z_dom == pytest.approx(10, rel=1e-2)


def test_bounding_rectangle():
    """Test the bounding_rectangle methods with arrays of 3D objects."""
    plane1 = Plane(o=Point3D(-5, 0, 0))
    plane2 = Plane(o=Point3D(0, -4, 4))
    plane3 = Plane(o=Point3D(2, 2, -4))
    polyface1 = Polyface3D.from_box(2, 4, 2, plane1)
    polyface2 = Polyface3D.from_box(2, 4, 2, plane2)
    polyface3 = Polyface3D.from_box(2, 4, 2, plane3)

    min_pt, max_pt = bounding_rectangle([polyface1, polyface2, polyface3])
    assert min_pt == Point2D(-5, -4)
    assert max_pt == Point2D(4, 6)

    x_dom, y_dom = bounding_rectangle_extents([polyface1, polyface2, polyface3])
    assert x_dom == 9
    assert y_dom == 10


def test_bounding_rectangle_angle():
    """Test the bounding_rectangle methods with an axis_angle."""
    plane1 = Plane(o=Point3D(-5, 0, 0))
    plane2 = Plane(o=Point3D(0, -4, 4))
    plane3 = Plane(o=Point3D(2, 2, -4))
    polyface1 = Polyface3D.from_box(2, 4, 2, plane1)
    polyface2 = Polyface3D.from_box(2, 4, 2, plane2)
    polyface3 = Polyface3D.from_box(2, 4, 2, plane3)

    min_pt, max_pt = bounding_rectangle([polyface1, polyface2, polyface3], 45)
    assert min_pt.x == pytest.approx(1.45, rel=1e-2)
    assert min_pt.y == pytest.approx(-4.89, rel=1e-2)
    assert max_pt.x == pytest.approx(-1.62, rel=1e-2)
    assert max_pt.y == pytest.approx(9.47, rel=1e-2)

    x_dom, y_dom = bounding_rectangle_extents([polyface1, polyface2, polyface3], 45)
    assert x_dom == pytest.approx(10.6103, rel=1e-2)
    assert y_dom == pytest.approx(10.1589, rel=1e-2)


def test_bounding_rectangle_angle_2d():
    """Test the bounding_rectangle methods with an axis_angle and 2D geometry."""
    pt1 = Point2D(-5, 0)
    pt2 = Point2D(0, -4)
    pt3 = Point2D(2, 2)
    polyface1 = Polygon2D.from_rectangle(pt1, Vector2D(0, 1), 2, 4)
    polyface2 = Polygon2D.from_rectangle(pt2, Vector2D(0, 1), 2, 4)
    polyface3 = Polygon2D.from_rectangle(pt3, Vector2D(0, 1), 2, 4)

    min_pt, max_pt = bounding_rectangle([polyface1, polyface2, polyface3], 45)
    assert min_pt.x == pytest.approx(1.45, rel=1e-2)
    assert min_pt.y == pytest.approx(-4.89, rel=1e-2)
    assert max_pt.x == pytest.approx(-1.62, rel=1e-2)
    assert max_pt.y == pytest.approx(9.47, rel=1e-2)

    x_dom, y_dom = bounding_rectangle_extents([polyface1, polyface2, polyface3], 45)
    assert x_dom == pytest.approx(10.6103, rel=1e-2)
    assert y_dom == pytest.approx(10.1589, rel=1e-2)
