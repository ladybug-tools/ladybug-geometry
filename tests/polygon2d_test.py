# coding=utf-8
import pytest
import math
import json

from ladybug_geometry.geometry2d.polygon import Polygon2D
from ladybug_geometry.geometry2d.pointvector import Point2D, Vector2D
from ladybug_geometry.geometry2d.line import LineSegment2D
from ladybug_geometry.geometry2d.ray import Ray2D


def test_polygon2d_init():
    """Test the initialization of Polygon2D objects and basic properties."""
    pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    polygon = Polygon2D(pts)

    str(polygon)  # test the string representation of the polygon

    assert isinstance(polygon.vertices, tuple)
    assert len(polygon.vertices) == 4
    assert len(polygon) == 4
    for point in polygon:
        assert isinstance(point, Point2D)

    assert isinstance(polygon.segments, tuple)
    assert len(polygon.segments) == 4
    for seg in polygon.segments:
        assert isinstance(seg, LineSegment2D)
        assert seg.length == 2

    assert polygon.area == 4
    assert polygon.perimeter == 8
    assert not polygon.is_clockwise
    assert polygon.is_convex
    assert not polygon.is_self_intersecting

    assert polygon.vertices[0] == polygon[0]

    p_array = polygon.to_array()
    assert isinstance(p_array, tuple)
    assert len(p_array) == 4
    for arr in p_array:
        assert isinstance(p_array, tuple)
        assert len(arr) == 2


def test_polygon2d_is_rectangle():
    """Test the Polygon2D.is_rectangle method."""
    ang_tol = math.radians(1)

    pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    polygon = Polygon2D(pts)
    assert polygon.is_rectangle(ang_tol)
    rect_ap = polygon.rectangular_approximation()
    assert polygon.area == pytest.approx(rect_ap.area, rel=1e-3)
    assert rect_ap.is_rectangle(ang_tol)

    pts = (Point2D(0, 0), Point2D(2.5, 0), Point2D(2, 2), Point2D(0, 2))
    polygon = Polygon2D(pts)
    assert not polygon.is_rectangle(ang_tol)
    rect_ap = polygon.rectangular_approximation()
    assert polygon.area == pytest.approx(rect_ap.area, rel=1e-3)
    assert rect_ap.is_rectangle(ang_tol)

    pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(1, 3), Point2D(0, 2))
    polygon = Polygon2D(pts)
    assert not polygon.is_rectangle(ang_tol)
    rect_ap = polygon.rectangular_approximation()
    assert polygon.area == pytest.approx(rect_ap.area, rel=1e-3)
    assert rect_ap.is_rectangle(ang_tol)


def test_polygon2d_pole_of_inaccessibility():
    """Test the Polygon2D.pole_of_inaccessibility method."""
    pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    polygon = Polygon2D(pts)

    pole = polygon.pole_of_inaccessibility(0.01)
    assert isinstance(pole, Point2D)
    assert pole.x == pytest.approx(1.0, rel=1e-3)
    assert pole.y == pytest.approx(1.0, rel=1e-3)


def test_equality():
    """Test the equality of Polygon2D objects."""
    pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    pts_2 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0.1, 2))
    polygon = Polygon2D(pts)
    polygon_dup = polygon.duplicate()
    polygon_alt = Polygon2D(pts_2)

    assert polygon is polygon
    assert polygon is not polygon_dup
    assert polygon == polygon_dup
    assert hash(polygon) == hash(polygon_dup)
    assert polygon != polygon_alt
    assert hash(polygon) != hash(polygon_alt)


def test_polygon2d_to_from_dict():
    """Test the to/from dict of Polygon2D objects."""
    pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    polygon = Polygon2D(pts)
    polygon_dict = polygon.to_dict()
    new_polygon = Polygon2D.from_dict(polygon_dict)
    assert isinstance(new_polygon, Polygon2D)
    assert new_polygon.to_dict() == polygon_dict


def test_polygon2d_init_from_rectangle():
    """Test the initialization of Polygon2D from_rectangle."""
    polygon = Polygon2D.from_rectangle(Point2D(0, 0), Vector2D(0, 1), 2, 2)

    assert isinstance(polygon.vertices, tuple)
    assert len(polygon.vertices) == 4
    for point in polygon.vertices:
        assert isinstance(point, Point2D)

    assert isinstance(polygon.segments, tuple)
    assert len(polygon.segments) == 4
    for seg in polygon.segments:
        assert isinstance(seg, LineSegment2D)
        assert seg.length == 2

    assert polygon.area == 4
    assert polygon.perimeter == 8
    assert not polygon.is_clockwise
    assert polygon.is_convex
    assert not polygon.is_self_intersecting


def test_polygon2d_init_from_regular_polygon():
    """Test the initialization of Polygon2D from_regular_polygon."""
    polygon = Polygon2D.from_regular_polygon(8, 2, Point2D(0, 1))

    assert isinstance(polygon.vertices, tuple)
    assert len(polygon.vertices) == 8
    for point in polygon.vertices:
        assert isinstance(point, Point2D)
    assert isinstance(polygon.segments, tuple)
    assert len(polygon.segments) == 8
    for seg in polygon.segments:
        assert isinstance(seg, LineSegment2D)
        assert seg.length == pytest.approx(1.5307337, rel=1e-3)

    assert polygon.area == pytest.approx(11.3137084, rel=1e-3)
    assert polygon.perimeter == pytest.approx(1.5307337 * 8, rel=1e-3)
    assert not polygon.is_clockwise
    assert polygon.is_convex
    assert not polygon.is_self_intersecting

    polygon = Polygon2D.from_regular_polygon(3)
    assert len(polygon.vertices) == 3
    polygon = Polygon2D.from_regular_polygon(20)
    assert len(polygon.vertices) == 20
    with pytest.raises(AssertionError):
        polygon = Polygon2D.from_regular_polygon(2)


def test_polygon2d_init_from_shape_with_hole():
    """Test the initialization of Polygon2D from_shape_with_hole."""
    bound_pts = [Point2D(0, 0), Point2D(4, 0), Point2D(4, 4), Point2D(0, 4)]
    hole_pts = [Point2D(1, 1), Point2D(3, 1), Point2D(3, 3), Point2D(1, 3)]
    polygon = Polygon2D.from_shape_with_hole(bound_pts, hole_pts)

    assert isinstance(polygon.vertices, tuple)
    assert len(polygon.vertices) == 10
    for point in polygon.vertices:
        assert isinstance(point, Point2D)

    assert isinstance(polygon.segments, tuple)
    assert len(polygon.segments) == 10
    for seg in polygon.segments:
        assert isinstance(seg, LineSegment2D)

    assert polygon.area == 12
    assert polygon.perimeter == pytest.approx(26.828427, rel=1e-3)
    assert not polygon.is_clockwise
    assert not polygon.is_convex


def test_polygon2d_init_from_shape_with_holes():
    """Test the initialization of Polygon2D from_shape_with_holes."""
    bound_pts = [Point2D(0, 0), Point2D(4, 0), Point2D(4, 4), Point2D(0, 4)]
    hole_pts_1 = [Point2D(1, 1), Point2D(1.5, 1), Point2D(1.5, 1.5), Point2D(1, 1.5)]
    hole_pts_2 = [Point2D(2, 2), Point2D(3, 2), Point2D(3, 3), Point2D(2, 3)]
    polygon = Polygon2D.from_shape_with_holes(bound_pts, [hole_pts_1, hole_pts_2])

    assert isinstance(polygon.vertices, tuple)
    assert len(polygon.vertices) == 16
    for point in polygon.vertices:
        assert isinstance(point, Point2D)

    assert isinstance(polygon.segments, tuple)
    assert len(polygon.segments) == 16
    for seg in polygon.segments:
        assert isinstance(seg, LineSegment2D)

    assert polygon.area == 16 - 1.25
    assert polygon.perimeter == pytest.approx(26.24264068, rel=1e-3)
    assert not polygon.is_clockwise
    assert not polygon.is_convex


def test_clockwise():
    """Test the clockwise property."""
    pts_1 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    polygon_1 = Polygon2D(pts_1)
    pts_2 = (Point2D(0, 0), Point2D(0, 2), Point2D(2, 2), Point2D(2, 0))
    polygon_2 = Polygon2D(pts_2)

    assert not polygon_1.is_clockwise
    assert polygon_2.is_clockwise
    assert polygon_1.reverse().is_clockwise

    assert polygon_1.area == 4
    assert polygon_2.area == 4


def test_is_convex():
    """Test the convex property."""
    pts_1 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    polygon_1 = Polygon2D(pts_1)
    pts_2 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 1), Point2D(1, 1),
             Point2D(1, 2), Point2D(0, 2))
    polygon_2 = Polygon2D(pts_2)

    assert polygon_1.is_convex
    assert not polygon_2.is_convex


def test_is_self_intersecting():
    """Test the is_self_intersecting property."""
    pts_1 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    polygon_1 = Polygon2D(pts_1)
    pts_2 = (Point2D(0, 0), Point2D(0, 2), Point2D(2, 0), Point2D(2, 2))
    polygon_2 = Polygon2D(pts_2)

    assert not polygon_1.is_self_intersecting
    assert polygon_2.is_self_intersecting


def test_is_valid():
    """Test the is_valid property."""
    pts_1 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2))
    pts_2 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 0))
    polygon_1 = Polygon2D(pts_1)
    polygon_2 = Polygon2D(pts_2)

    assert polygon_1.is_valid
    assert not polygon_2.is_valid


def test_min_max_center():
    """Test the Polygon2D min, max and center."""
    pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    polygon = Polygon2D(pts)

    assert polygon.min == Point2D(0, 0)
    assert polygon.max == Point2D(2, 2)
    assert polygon.center == Point2D(1, 1)


def test_remove_colinear_vertices():
    """Test the remove_colinear_vertices method of Polygon2D."""
    pts_1 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    pts_2 = (Point2D(0, 0), Point2D(1, 0), Point2D(2, 0), Point2D(2, 2),
             Point2D(0, 2))
    polygon_1 = Polygon2D(pts_1)
    polygon_2 = Polygon2D(pts_2)

    assert len(polygon_1.remove_colinear_vertices(0.0001).vertices) == 4
    assert len(polygon_2.remove_colinear_vertices(0.0001).vertices) == 4


def test_polygon2d_duplicate():
    """Test the duplicate method of Polygon2D."""
    pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    polygon = Polygon2D(pts)
    new_polygon = polygon.duplicate()

    for i, pt in enumerate(new_polygon):
        assert pt == pts[i]

    assert polygon.area == new_polygon.area
    assert polygon.perimeter == new_polygon.perimeter
    assert polygon.is_clockwise == new_polygon.is_clockwise
    assert polygon.is_convex == new_polygon.is_convex
    assert polygon.is_self_intersecting == new_polygon.is_self_intersecting

    new_polygon_2 = new_polygon.duplicate()
    assert new_polygon.area == new_polygon_2.area
    assert new_polygon.perimeter == new_polygon_2.perimeter
    assert new_polygon.is_clockwise == new_polygon_2.is_clockwise
    assert new_polygon.is_convex == new_polygon_2.is_convex
    assert new_polygon.is_self_intersecting == new_polygon_2.is_self_intersecting


def test_reverse():
    """Test the reverse method."""
    pts_1 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    polygon = Polygon2D(pts_1)
    new_polygon = polygon.reverse()

    assert polygon.area == new_polygon.area
    assert polygon.perimeter == new_polygon.perimeter
    assert polygon.is_clockwise is not new_polygon.is_clockwise
    assert polygon.is_convex == new_polygon.is_convex
    assert polygon.is_self_intersecting == new_polygon.is_self_intersecting


def test_offset():
    """Test the offset method."""
    pts_1 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    polygon = Polygon2D(pts_1)
    new_polygon = polygon.offset(0.5)

    assert 0.99 < new_polygon.area < 1.01
    assert not new_polygon.is_clockwise
    assert polygon.is_convex == new_polygon.is_convex
    assert polygon.is_self_intersecting == new_polygon.is_self_intersecting

    rev_poly = polygon.reverse()
    new_rev_polygon = rev_poly.offset(0.5)
    assert 0.99 < new_rev_polygon.area < 1.01
    assert new_rev_polygon.is_clockwise


def test_move():
    """Test the Polygon2D move method."""
    pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    polygon = Polygon2D(pts)

    vec_1 = Vector2D(2, 2)
    new_polygon = polygon.move(vec_1)
    assert new_polygon[0] == Point2D(2, 2)
    assert new_polygon[1] == Point2D(4, 2)
    assert new_polygon[2] == Point2D(4, 4)
    assert new_polygon[3] == Point2D(2, 4)

    assert polygon.area == new_polygon.area
    assert polygon.perimeter == new_polygon.perimeter
    assert polygon.is_clockwise is new_polygon.is_clockwise
    assert polygon.is_convex is new_polygon.is_convex
    assert polygon.is_self_intersecting is new_polygon.is_self_intersecting


def test_scale():
    """Test the Polygon2D scale method."""
    pts_1 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    polygon_1 = Polygon2D(pts_1)
    pts_2 = (Point2D(1, 1), Point2D(2, 1), Point2D(2, 2), Point2D(1, 2))
    polygon_2 = Polygon2D(pts_2)
    origin_1 = Point2D(2, 0)
    origin_2 = Point2D(1, 1)

    new_polygon_1 = polygon_1.scale(2, origin_1)
    assert new_polygon_1[0] == Point2D(-2, 0)
    assert new_polygon_1[1] == Point2D(2, 0)
    assert new_polygon_1[2] == Point2D(2, 4)
    assert new_polygon_1[3] == Point2D(-2, 4)
    assert new_polygon_1.area == polygon_1.area ** 2
    assert new_polygon_1.perimeter == polygon_1.perimeter * 2
    assert new_polygon_1.is_clockwise is polygon_1.is_clockwise
    assert new_polygon_1.is_convex is polygon_1.is_convex
    assert new_polygon_1.is_self_intersecting is polygon_1.is_self_intersecting

    new_polygon_2 = polygon_2.scale(2, origin_2)
    assert new_polygon_2[0] == Point2D(1, 1)
    assert new_polygon_2[1] == Point2D(3, 1)
    assert new_polygon_2[2] == Point2D(3, 3)
    assert new_polygon_2[3] == Point2D(1, 3)
    assert new_polygon_2.area == 4
    assert new_polygon_2.perimeter == polygon_2.perimeter * 2
    assert new_polygon_2.is_clockwise is polygon_2.is_clockwise
    assert new_polygon_2.is_convex is polygon_2.is_convex
    assert new_polygon_2.is_self_intersecting is polygon_2.is_self_intersecting


def test_scale_world_origin():
    """Test the Polygon2D scale method with None origin."""
    pts = (Point2D(1, 1), Point2D(2, 1), Point2D(2, 2), Point2D(1, 2))
    polygon = Polygon2D(pts)

    new_polygon = polygon.scale(2)
    assert new_polygon[0] == Point2D(2, 2)
    assert new_polygon[1] == Point2D(4, 2)
    assert new_polygon[2] == Point2D(4, 4)
    assert new_polygon[3] == Point2D(2, 4)
    assert new_polygon.area == 4
    assert new_polygon.perimeter == polygon.perimeter * 2
    assert new_polygon.is_clockwise is polygon.is_clockwise
    assert new_polygon.is_convex is polygon.is_convex
    assert new_polygon.is_self_intersecting is polygon.is_self_intersecting


def test_rotate():
    """Test the Polygon2D rotate method."""
    pts = (Point2D(1, 1), Point2D(2, 1), Point2D(2, 2), Point2D(1, 2))
    polygon = Polygon2D(pts)
    origin_1 = Point2D(1, 1)

    test_1 = polygon.rotate(math.pi, origin_1)
    assert test_1[0].x == pytest.approx(1, rel=1e-3)
    assert test_1[0].y == pytest.approx(1, rel=1e-3)
    assert test_1[2].x == pytest.approx(0, rel=1e-3)
    assert test_1[2].y == pytest.approx(0, rel=1e-3)
    assert polygon.area == pytest.approx(test_1.area, rel=1e-3)
    assert polygon.perimeter == pytest.approx(test_1.perimeter, rel=1e-3)
    assert polygon.is_clockwise is test_1.is_clockwise
    assert polygon.is_convex is test_1.is_convex
    assert polygon.is_self_intersecting is test_1.is_self_intersecting

    test_2 = polygon.rotate(math.pi / 2, origin_1)
    assert test_2[0].x == pytest.approx(1, rel=1e-3)
    assert test_2[0].y == pytest.approx(1, rel=1e-3)
    assert test_2[2].x == pytest.approx(0, rel=1e-3)
    assert test_2[2].y == pytest.approx(2, rel=1e-3)


def test_reflect():
    """Test the Polygon2D reflect method."""
    pts = (Point2D(1, 1), Point2D(2, 1), Point2D(2, 2), Point2D(1, 2))
    polygon = Polygon2D(pts)

    origin_1 = Point2D(1, 0)
    normal_1 = Vector2D(1, 0)
    normal_2 = Vector2D(-1, -1).normalize()

    test_1 = polygon.reflect(normal_1, origin_1)
    assert test_1[0].x == pytest.approx(1, rel=1e-3)
    assert test_1[0].y == pytest.approx(1, rel=1e-3)
    assert test_1[2].x == pytest.approx(0, rel=1e-3)
    assert test_1[2].y == pytest.approx(2, rel=1e-3)
    assert polygon.area == pytest.approx(test_1.area, rel=1e-3)
    assert polygon.perimeter == pytest.approx(test_1.perimeter, rel=1e-3)
    assert polygon.is_clockwise is not test_1.is_clockwise
    assert polygon.is_convex is test_1.is_convex
    assert polygon.is_self_intersecting is test_1.is_self_intersecting

    test_1 = polygon.reflect(normal_2, Point2D(0, 0))
    assert test_1[0].x == pytest.approx(-1, rel=1e-3)
    assert test_1[0].y == pytest.approx(-1, rel=1e-3)
    assert test_1[2].x == pytest.approx(-2, rel=1e-3)
    assert test_1[2].y == pytest.approx(-2, rel=1e-3)

    test_2 = polygon.reflect(normal_2, origin_1)
    assert test_2[0].x == pytest.approx(0, rel=1e-3)
    assert test_2[0].y == pytest.approx(0, rel=1e-3)
    assert test_2[2].x == pytest.approx(-1, rel=1e-3)
    assert test_2[2].y == pytest.approx(-1, rel=1e-3)


def test_intersect_line_ray():
    """Test the Polygon2D intersect_line_ray method."""
    pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    polygon = Polygon2D(pts)

    ray_1 = Ray2D(Point2D(-1, 1), Vector2D(1, 0))
    ray_2 = Ray2D(Point2D(1, 1), Vector2D(1, 0))
    ray_3 = Ray2D(Point2D(1, 1), Vector2D(11, 0))
    ray_4 = Ray2D(Point2D(-1, 1), Vector2D(-1, 0))

    assert len(polygon.intersect_line_ray(ray_1)) == 2
    assert len(polygon.intersect_line_ray(ray_2)) == 1
    assert len(polygon.intersect_line_ray(ray_3)) == 1
    assert len(polygon.intersect_line_ray(ray_4)) == 0

    line_1 = LineSegment2D(Point2D(-1, 1), Vector2D(0.5, 0))
    line_2 = LineSegment2D(Point2D(-1, 1), Vector2D(2, 0))
    line_3 = LineSegment2D(Point2D(-1, 1), Vector2D(3, 0))

    assert len(polygon.intersect_line_ray(line_1)) == 0
    assert len(polygon.intersect_line_ray(line_2)) == 1
    assert len(polygon.intersect_line_ray(line_3)) == 2


def test_intersect_line_infinite():
    """Test the Polygon2D intersect_line_infinite method."""
    pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    polygon = Polygon2D(pts)

    ray_1 = Ray2D(Point2D(-1, 1), Vector2D(1, 0))
    ray_2 = Ray2D(Point2D(1, 1), Vector2D(1, 0))
    ray_3 = Ray2D(Point2D(1, 1), Vector2D(11, 0))
    ray_4 = Ray2D(Point2D(-1, 1), Vector2D(-1, 0))
    ray_5 = Ray2D(Point2D(-1, 3), Vector2D(-1, 0))
    ray_6 = Ray2D(Point2D(0, 2), Vector2D(-1, -1))

    assert len(polygon.intersect_line_infinite(ray_1)) == 2
    assert len(polygon.intersect_line_infinite(ray_2)) == 2
    assert len(polygon.intersect_line_infinite(ray_3)) == 2
    assert len(polygon.intersect_line_infinite(ray_4)) == 2
    assert len(polygon.intersect_line_infinite(ray_5)) == 0
    assert len(polygon.intersect_line_infinite(ray_6)) > 0


def test_point_relationship():
    """Test the Polygon2D point_relationship method."""
    pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    polygon = Polygon2D(pts)

    assert polygon.point_relationship(Point2D(-1, 1), 0.0001) == -1
    assert polygon.point_relationship(Point2D(1, 1), 0.0001) == 1
    assert polygon.point_relationship(Point2D(0, 1), 0.0001) == 0
    assert polygon.point_relationship(Point2D(2, 2), 0.0001) == 0

    pts_2 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 1), Point2D(1, 1),
             Point2D(1, 2), Point2D(0, 2))
    polygon_2 = Polygon2D(pts_2)
    assert polygon_2.point_relationship(Point2D(0.5, 1), 0.0001) == 1
    assert polygon_2.point_relationship(Point2D(0.5, 0.5), 0.0001) == 1
    assert polygon_2.point_relationship(Point2D(1, 0.5), 0.0001) == 1
    assert polygon_2.point_relationship(Point2D(0, 1), 0.0001) == 0
    assert polygon_2.point_relationship(Point2D(0, 2), 0.0001) == 0
    assert polygon_2.point_relationship(Point2D(-2, 0.5), 0.0001) == -1
    assert polygon_2.point_relationship(Point2D(-2, 2), 0.0001) == -1
    assert polygon_2.point_relationship(Point2D(-1, 1), 0.0001) == -1
    assert polygon_2.point_relationship(Point2D(-1, 1), 0.0001) == -1


def test_is_point_inside():
    """Test the Polygon2D is_point_inside method."""
    pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    polygon = Polygon2D(pts)

    assert not polygon.is_point_inside(Point2D(-1, 1))
    assert not polygon.is_point_inside(Point2D(1, -1))
    assert not polygon.is_point_inside(Point2D(1, 3))
    assert not polygon.is_point_inside(Point2D(3, 1))
    assert polygon.is_point_inside(Point2D(1, 1))

    assert not polygon.is_point_inside(Point2D(-1, 1), Vector2D(0, 1))
    assert not polygon.is_point_inside(Point2D(1, -1), Vector2D(0, 1))
    assert not polygon.is_point_inside(Point2D(1, 3), Vector2D(0, 1))
    assert not polygon.is_point_inside(Point2D(3, 1), Vector2D(0, 1))
    assert polygon.is_point_inside(Point2D(1, 1), Vector2D(0, 1))


def test_is_point_inside_bound_rect():
    """Test the Polygon2D is_point_inside_bound_rect method."""
    pts = (Point2D(0, 0), Point2D(4, 0), Point2D(4, 2), Point2D(2, 2),
           Point2D(2, 4), Point2D(0, 4))
    polygon = Polygon2D(pts)

    assert not polygon.is_point_inside_bound_rect(Point2D(-1, 1))
    assert not polygon.is_point_inside_bound_rect(Point2D(1, -1))
    assert not polygon.is_point_inside_bound_rect(Point2D(1, 5))
    assert not polygon.is_point_inside_bound_rect(Point2D(5, 1))
    assert not polygon.is_point_inside_bound_rect(Point2D(3, 3))
    assert polygon.is_point_inside(Point2D(1, 1))


def test_is_polygon_inside_outside():
    """Test the is_polygon_inside and is_polygon_outside methods."""
    bound_pts = [Point2D(0, 0), Point2D(4, 0), Point2D(4, 4), Point2D(0, 4)]
    hole_pts_1 = [Point2D(1, 1), Point2D(1.5, 1), Point2D(1.5, 1.5), Point2D(1, 1.5)]
    hole_pts_2 = [Point2D(2, 2), Point2D(3, 2), Point2D(3, 3), Point2D(2, 3)]
    hole_pts_3 = [Point2D(2, 2), Point2D(6, 2), Point2D(6, 6), Point2D(2, 6)]
    hole_pts_4 = [Point2D(5, 5), Point2D(6, 5), Point2D(6, 6), Point2D(5, 6)]
    polygon = Polygon2D(bound_pts)
    hole_1 = Polygon2D(hole_pts_1)
    hole_2 = Polygon2D(hole_pts_2)
    hole_3 = Polygon2D(hole_pts_3)
    hole_4 = Polygon2D(hole_pts_4)

    assert polygon.is_polygon_inside(hole_1)
    assert polygon.is_polygon_inside(hole_2)
    assert not polygon.is_polygon_inside(hole_3)
    assert not polygon.is_polygon_inside(hole_4)

    assert not polygon.is_polygon_outside(hole_1)
    assert not polygon.is_polygon_outside(hole_2)
    assert not polygon.is_polygon_outside(hole_3)
    assert polygon.is_polygon_outside(hole_4)


def test_polygon_relationship():
    """Test the polygon_relationship method."""
    # check polygon with itself
    bound_pts = [Point2D(0, 0), Point2D(4, 0), Point2D(4, 4), Point2D(0, 4)]
    polygon = Polygon2D(bound_pts)
    assert polygon.polygon_relationship(polygon, 0.01) == 1

    # check polygon with various types holes with clearly-defined relationships
    hole_pts_1 = [Point2D(1, 1), Point2D(1.5, 1), Point2D(1.5, 1.5), Point2D(1, 1.5)]
    hole_pts_2 = [Point2D(2, 2), Point2D(3, 2), Point2D(3, 3), Point2D(2, 3)]
    hole_pts_3 = [Point2D(2, 2), Point2D(6, 2), Point2D(6, 6), Point2D(2, 6)]
    hole_pts_4 = [Point2D(5, 5), Point2D(6, 5), Point2D(6, 6), Point2D(5, 6)]
    hole_1 = Polygon2D(hole_pts_1)
    hole_2 = Polygon2D(hole_pts_2)
    hole_3 = Polygon2D(hole_pts_3)
    hole_4 = Polygon2D(hole_pts_4)
    assert polygon.polygon_relationship(hole_1, 0.01) == 1
    assert polygon.polygon_relationship(hole_2, 0.01) == 1
    assert polygon.polygon_relationship(hole_3, 0.01) == 0
    assert polygon.polygon_relationship(hole_4, 0.01) == -1

    # check the polygon with an adjacent one within tolerance
    adj_pts = [Point2D(3.999, 0), Point2D(5, 0), Point2D(5, 4), Point2D(4, 4)]
    adj_p = Polygon2D(adj_pts)
    assert polygon.polygon_relationship(adj_p, 0.01) == -1

    # check the polygon colinear with the other
    in_pts = [Point2D(0, 0), Point2D(4, 0), Point2D(4, 2), Point2D(0, 2)]
    in_p = Polygon2D(in_pts)
    assert polygon.polygon_relationship(in_p, 0.01) == 1
    assert in_p.polygon_relationship(polygon, 0.01) == 0

    # check the polygon with an overlapping intersection
    int_pts = [Point2D(-1, 1), Point2D(5, 1), Point2D(5, 3), Point2D(-1, 3)]
    int_p = Polygon2D(int_pts)
    assert polygon.polygon_relationship(int_p, 0.01) == 0

    # check the polygon that contains the other
    cont_pts = [Point2D(-1, -1), Point2D(5, -1), Point2D(5, 5), Point2D(-1, 5)]
    cont_p = Polygon2D(cont_pts)
    assert polygon.polygon_relationship(cont_p, 0.01) == 0
    assert cont_p.polygon_relationship(polygon, 0.01) == 1

    # check the polygon with a concave overlap
    conc_pts = [Point2D(-1, -1), Point2D(5, -1), Point2D(5, 5), Point2D(3, 5),
                Point2D(3, 3), Point2D(2, 3), Point2D(2, 5), Point2D(-1, 5)]
    conc_p = Polygon2D(conc_pts)
    assert polygon.polygon_relationship(conc_p, 0.01) == 0
    assert conc_p.polygon_relationship(polygon, 0.01) == 0

    # check the polygon with partial overlaps extending outward
    pts_1 = (
        Point2D(-6.30, 0.00), Point2D(-6.30, -1.54), Point2D(-3.16, -1.54),
        Point2D(-3.16, -6.24), Point2D(0.00, -6.24), Point2D(0.00, 0.00)
    )
    pts_2 = (
        Point2D(-3.16, -1.54), Point2D(-6.55, -1.54), Point2D(-9.04, -1.54),
        Point2D(-12.10, -1.54), Point2D(-12.10, 0.00), Point2D(0.00, 0.00),
        Point2D(0.00, -6.20), Point2D(-3.16, -6.20)
    )
    poly_1 = Polygon2D(pts_1)
    poly_2 = Polygon2D(pts_2)
    assert poly_1.polygon_relationship(poly_2, 0.01) == 0
    assert poly_2.polygon_relationship(poly_1, 0.01) == 0


def test_polygon_relationship_concave():
    """Test the polygon_relationship method."""
    # check the case of polygon inside a concave hole
    inside_pts = [Point2D(1, 0), Point2D(2, 0), Point2D(2, 1), Point2D(1, 1)]
    outside_pts = [
        Point2D(0, 0), Point2D(1, 0), Point2D(1, 1), Point2D(2, 1), Point2D(2, 0),
        Point2D(3, 0), Point2D(3, 3), Point2D(0, 3)
    ]
    inside_polygon = Polygon2D(inside_pts)
    outside_polygon = Polygon2D(outside_pts)
    assert inside_polygon.polygon_relationship(outside_polygon, 0.01) == -1
    assert outside_polygon.polygon_relationship(inside_polygon, 0.01) == -1


def test_group_by_overlap():
    """Test the group_by_overlap method."""
    bound_pts1 = [Point2D(0, 0), Point2D(4, 0), Point2D(4, 4), Point2D(0, 4)]
    bound_pts2 = [Point2D(2, 2), Point2D(6, 2), Point2D(6, 6), Point2D(2, 6)]
    bound_pts3 = [Point2D(6, 6), Point2D(7, 6), Point2D(7, 7), Point2D(6, 7)]
    polygon1 = Polygon2D(bound_pts1)
    polygon2 = Polygon2D(bound_pts2)
    polygon3 = Polygon2D(bound_pts3)

    all_polys = [polygon1, polygon2, polygon3]

    grouped_polys = Polygon2D.group_by_overlap(all_polys, 0.01)
    assert len(grouped_polys) == 2
    assert len(grouped_polys[0]) == 2
    assert len(grouped_polys[1]) == 1

    grouped_polys = Polygon2D.group_by_overlap(list(reversed(all_polys)), 0.01)
    assert len(grouped_polys) == 2
    assert len(grouped_polys[0]) == 1
    assert len(grouped_polys[1]) == 2


def test_distance_to_point():
    """Test the distance_to_point method."""
    pts = (Point2D(0, 0), Point2D(4, 0), Point2D(4, 2), Point2D(2, 2),
           Point2D(2, 4), Point2D(0, 4))
    polygon = Polygon2D(pts)

    assert polygon.distance_to_point(Point2D(-2, 1)) == 2
    assert polygon.distance_to_point(Point2D(1, -1)) == 1
    assert polygon.distance_to_point(Point2D(1, 5)) == 1
    assert polygon.distance_to_point(Point2D(5, 1)) == 1
    assert polygon.distance_to_point(Point2D(3, 3)) == 1
    assert polygon.distance_to_point(Point2D(1, 1)) == 0

    assert polygon.distance_from_edge_to_point(Point2D(1, 1)) != 0


def test_intersect_segments():
    """Tests that polygons within tolerance distance have vertices updated."""
    tolerance = 0.02
    pts0 = (Point2D(1, 0), Point2D(4, 0), Point2D(4, 1.99), Point2D(1, 1.99))
    polygon0 = Polygon2D(pts0)
    pts1 = (Point2D(0, 2), Point2D(3, 2), Point2D(3, 4), Point2D(0, 4))
    polygon1 = Polygon2D(pts1)

    polygon0, polygon1 = Polygon2D.intersect_segments(polygon0, polygon1, tolerance)

    # Extra vertex added to polygon0, as expected
    assert len(polygon0.segments) == 5
    assert polygon0.vertices[3] == Point2D(3, 1.99)
    assert polygon0.segments[2].p2 == Point2D(3, 1.99)
    assert polygon0.segments[3].p1 == Point2D(3, 1.99)

    # Extra vertex added to polygon1, as expected
    assert len(polygon1.segments) == 5
    assert polygon1.vertices[1] == Point2D(1, 2)
    assert polygon1.segments[0].p2 == Point2D(1, 2)
    assert polygon1.segments[1].p1 == Point2D(1, 2)


def test_intersect_segments_multiple_intersections():
    """Tests that polygons having multiple intersections are ordered correctly."""
    pts1 = (Point2D(0, 0), Point2D(5, 0), Point2D(5, 5), Point2D(0, 5))
    pts2 = (Point2D(6, 4), Point2D(5, 4), Point2D(5, 3), Point2D(6, 3))
    pts3 = (Point2D(5, 3), Point2D(6, 3), Point2D(6, 4), Point2D(5, 4))
    pts4 = (Point2D(7, 4), Point2D(5, 4), Point2D(5, 3), Point2D(6, 3),
            Point2D(6, 2), Point2D(5, 2), Point2D(5, 1), Point2D(7, 1))

    poly1 = Polygon2D(pts1)
    poly2 = Polygon2D(pts2)
    poly3 = Polygon2D(pts3)
    poly4 = Polygon2D(pts4)

    int_polys1 = Polygon2D.intersect_segments(poly1, poly2, 0.01)
    int_polys2 = Polygon2D.intersect_segments(poly1, poly3, 0.01)
    int_polys3 = Polygon2D.intersect_segments(poly1, poly4, 0.01)
    int_polys4 = Polygon2D.intersect_segments(poly1, poly4, 0.01)

    assert int_polys1[0].vertices == int_polys2[0].vertices == \
        (Point2D(0, 0), Point2D(5, 0), Point2D(5, 3), Point2D(5, 4),
         Point2D(5, 5), Point2D(0, 5))
    assert int_polys3[0].vertices == int_polys4[0].vertices == \
        (Point2D(0, 0), Point2D(5, 0), Point2D(5, 1), Point2D(5, 2),
         Point2D(5, 3), Point2D(5, 4), Point2D(5, 5), Point2D(0, 5))


def test_intersect_segments_zero_tolerance():
    """Tests that the default tolerance of 0 does not update nearby polygons."""
    pts0 = (Point2D(1, 0), Point2D(4, 0), Point2D(4, 1.99), Point2D(1, 1.99))
    polygon0 = Polygon2D(pts0)
    pts1 = (Point2D(0, 2), Point2D(3, 2), Point2D(3, 4), Point2D(0, 4))
    polygon1 = Polygon2D(pts1)

    polygon2, polygon3 = Polygon2D.intersect_segments(polygon0, polygon1, 0)

    assert len(polygon2.segments) == 4  # No new points
    assert all([polygon0.vertices[i] == polygon2.vertices[i]
                for i in range(len(polygon0.vertices))])
    assert len(polygon3.segments) == 4  # No new points
    assert all([polygon1.vertices[i] == polygon3.vertices[i]
                for i in range(len(polygon1.vertices))])

    polygon2, polygon3 = Polygon2D.intersect_segments(polygon0, polygon1, 0.02)

    assert len(polygon2.segments) == 5  # Intersection within tolerance
    assert len(polygon3.segments) == 5  # Intersection within tolerance


def test_intersect_segments_with_colinear_edges():
    """Test tolerance of 0 updates polygons which share part of an edge segment."""
    pts0 = (Point2D(1, 0), Point2D(4, 0), Point2D(4, 2), Point2D(1, 2))
    polygon0 = Polygon2D(pts0)
    pts1 = (Point2D(0, 2), Point2D(3, 2), Point2D(3, 4), Point2D(0, 4))
    polygon1 = Polygon2D(pts1)

    polygon0, polygon1 = Polygon2D.intersect_segments(polygon0, polygon1, 0)

    # Extra vertex added to polygon0, as expected
    assert len(polygon0.segments) == 5
    assert polygon0.vertices[3] == Point2D(3, 2)
    assert polygon0.segments[2].p2 == Point2D(3, 2)
    assert polygon0.segments[3].p1 == Point2D(3, 2)

    # Extra vertex added to polygon1, as expected
    assert len(polygon1.segments) == 5
    assert polygon1.vertices[1] == Point2D(1, 2)
    assert polygon1.segments[0].p2 == Point2D(1, 2)
    assert polygon1.segments[1].p1 == Point2D(1, 2)


def test_intersect_polygon_segments_with_3_rectangles():
    """Test vertex shared by 2 polygons is added only once to a 3rd polygon."""
    pts0 = (Point2D(0, 2), Point2D(4, 2), Point2D(4, 4), Point2D(0, 4))
    polygon0 = Polygon2D(pts0)
    pts1 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
    polygon1 = Polygon2D(pts1)
    pts2 = (Point2D(2, 0), Point2D(4, 0), Point2D(4, 2), Point2D(2, 2))
    polygon2 = Polygon2D(pts2)

    polygons = Polygon2D.intersect_polygon_segments([polygon0, polygon1, polygon2], 0)

    # Extra vertex added to largest polygon, as expected
    assert len(polygons[0].segments) == 5
    assert polygons[0].vertices[1] == Point2D(2, 2)
    assert polygons[0].segments[0].p2 == Point2D(2, 2)
    assert polygons[0].segments[1].p1 == Point2D(2, 2)

    assert len(polygon1.segments) == 4  # No extra vertex added
    assert len(polygon2.segments) == 4  # No extra vertex added


def test_intersect_polygon_segments_with_3_angled_rectangles():
    """Tests that a vertex shared by 2 polygons is added only once to a 3rd polygon

    Make sure the added vertex is which is colinear within tolerance.
    The polygons are rotated 45 degrees counter-clockwise to introduce floating-point
    closeness considerations.
    """
    r2 = math.sqrt(2.0)
    tolerance = 0.02
    expected_point = Point2D(r2, 0)
    pts0 = (Point2D(0, 0), Point2D(0.5 * r2 * 0.99, -0.5 * r2 * 0.99),
            Point2D(1.5 * r2 * 0.99, 0.5 * r2 * 0.99), Point2D(r2, r2))
    polygon0 = Polygon2D(pts0)
    pts1 = (Point2D(0.5 * r2, -0.5 * r2), Point2D(r2, -r2),
            Point2D(1.5 * r2, -0.5 * r2), expected_point)
    polygon1 = Polygon2D(pts1)
    pts2 = (expected_point, Point2D(1.5 * r2, -0.5 * r2),
            Point2D(2 * r2, 0), Point2D(1.5 * r2, 0.5 * r2))
    polygon2 = Polygon2D(pts2)

    polygons = Polygon2D.intersect_polygon_segments(
        [polygon0, polygon1, polygon2], tolerance)

    # Extra vertex added to largest polygon, as expected
    assert len(polygons[0].segments) == 5
    assert polygons[0].segments[1].p2 == polygons[0].vertices[2]
    assert polygons[0].segments[2].p1 == polygons[0].vertices[2]

    assert len(polygon1.segments) == 4  # No extra vertex added
    assert len(polygon2.segments) == 4  # No extra vertex added


def test_intersect_polygon_segments_abraham_bug():
    """"Test the polygon intersection with the bug found by Abraham."""
    pts1 = [Point2D(-154.33, -272.63), Point2D(-155.30, -276.70),
            Point2D(-151.23, -277.68), Point2D(-150.26, -273.61),
            Point2D(-146.19, -274.58), Point2D(-145.22, -270.51),
            Point2D(-149.29, -269.54), Point2D(-148.31, -265.47),
            Point2D(-152.38, -264.50), Point2D(-153.35, -268.57),
            Point2D(-157.42, -267.59), Point2D(-158.39, -271.66)]
    pts2 = [Point2D(-161.49, -266.62), Point2D(-153.35, -268.57),
            Point2D(-151.41, -260.43), Point2D(-159.54, -258.49)]
    pts3 = [Point2D(-164.41, -278.82), Point2D(-156.27, -280.77),
            Point2D(-154.33, -272.63), Point2D(-162.46, -270.69)]
    pts4 = [Point2D(-149.29, -269.54), Point2D(-141.15, -271.49),
            Point2D(-139.20, -263.35), Point2D(-147.34, -261.41)]
    pts5 = [Point2D(-152.21, -281.74), Point2D(-144.07, -283.69),
            Point2D(-142.12, -275.55), Point2D(-150.26, -273.61)]

    polygon1 = Polygon2D(pts1)
    polygon2 = Polygon2D(pts2)
    polygon3 = Polygon2D(pts3)
    polygon4 = Polygon2D(pts4)
    polygon5 = Polygon2D(pts5)
    polygons = [polygon1, polygon2, polygon3, polygon4, polygon5]

    new_polygons = Polygon2D.intersect_polygon_segments(polygons, 0.01)
    for polygon in new_polygons:
        assert not polygon.is_self_intersecting


def test_polygon_is_equivalent():
    """Test if polygons are equivalent based on point equivalence."""

    tol = 1e-10
    p1 = Polygon2D.from_array([[0, 0], [6, 0], [7, 3], [0, 4]])

    # Test no points are the same
    p2 = Polygon2D.from_array([[0, 1], [6, 1], [7, 4]])
    assert not p1.is_equivalent(p2, tol)

    # Test when length is not same
    p2 = Polygon2D.from_array([[0, 0], [6, 0], [7, 3]])
    assert not p1.is_equivalent(p2, tol)

    # Test equal condition same order
    p2 = Polygon2D.from_array([[0, 0], [6, 0], [7, 3], [0, 4]])
    assert p1.is_equivalent(p2, tol)

    # Test equal condition different order 1
    p2 = Polygon2D.from_array([[7, 3], [0, 4], [0, 0], [6, 0]])
    assert p1.is_equivalent(p2, tol)

    # Test equal condition different order 2
    p2 = Polygon2D.from_array([[0, 4], [0, 0], [6, 0], [7, 3]])
    assert p1.is_equivalent(p2, tol)


def test_boolean_union():
    """Test the boolean_union method."""
    polygon_a = Polygon2D.from_array((
        (174.731903, -72.989276),
        (-70.77748, -53.08311),
        (-72.252011, 215.281501),
        (129.021448, 126.809651),
        (106.16622, 28.016086),
        (216.756032, 22.117962),
        (174.731903, -72.989276),
    ))
    polygon_b = Polygon2D.from_array((
        (-169.571046, -98.793566),
        (-145.241287, 63.404826),
        (11.796247, 34.651475),
        (8.10992, -129.758713),
        (-76.675603, -216.018767),
        (-169.571046, -98.793566),
    ))

    polygon_union = polygon_a.boolean_union(polygon_b, 0.01)
    assert len(polygon_union) == 1
    assert len(polygon_union[0].vertices) == 11

    polygon_union = Polygon2D.boolean_union_all([polygon_a, polygon_b], 0.01)
    assert len(polygon_union) == 1
    assert len(polygon_union[0].vertices) == 11


def test_boolean_intersect():
    """Test the boolean_intersect method."""
    polygon_a = Polygon2D.from_array((
        (174.731903, -72.989276),
        (-70.77748, -53.08311),
        (-72.252011, 215.281501),
        (129.021448, 126.809651),
        (106.16622, 28.016086),
        (216.756032, 22.117962),
        (174.731903, -72.989276),
    ))
    polygon_b = Polygon2D.from_array((
        (-169.571046, -98.793566),
        (-145.241287, 63.404826),
        (11.796247, 34.651475),
        (8.10992, -129.758713),
        (-76.675603, -216.018767),
        (-169.571046, -98.793566),
    ))

    polygon_intersect = polygon_a.boolean_intersect(polygon_b, 0.01)
    assert len(polygon_intersect) == 1
    assert len(polygon_intersect[0].vertices) == 4

    polygon_intersect = Polygon2D.boolean_intersect_all([polygon_a, polygon_b], 0.01)
    assert len(polygon_intersect) == 1
    assert len(polygon_intersect[0].vertices) == 4


def test_boolean_xor():
    """Test the boolean_xor method."""
    polygon_a = Polygon2D.from_array((
        (174.731903, -72.989276),
        (-70.77748, -53.08311),
        (-72.252011, 215.281501),
        (129.021448, 126.809651),
        (106.16622, 28.016086),
        (216.756032, 22.117962),
        (174.731903, -72.989276),
    ))
    polygon_b = Polygon2D.from_array((
        (-169.571046, -98.793566),
        (-145.241287, 63.404826),
        (11.796247, 34.651475),
        (8.10992, -129.758713),
        (-76.675603, -216.018767),
        (-169.571046, -98.793566),
    ))

    polygon_xor = polygon_a.boolean_xor(polygon_b, 0.01)
    assert len(polygon_xor) == 1
    assert len(polygon_xor[0].vertices) == 15


def test_boolean_difference():
    """Test the boolean_difference method."""
    polygon_a = Polygon2D.from_array((
        (704.237624, 311.067429),
        (585.714469, 165.25985),
        (580.737459, -21.556226),
        (825.140768, -42.669151),
        (939.27375, 202.801646),
        (907.837171, 299.649899),
        (704.237624, 311.067429),
    ))
    polygon_b = Polygon2D.from_array((
        (594.128232, 182.178958),
        (756.152613, 106.844841),
        (746.79128, 86.711012),
        (578.739333, 91.188122),
        (578.55291, 84.190605),
        (675.400366, 81.61047),
        (672.301879, -34.693726),
        (679.299396, -34.880148),
        (682.397883, 81.424047),
        (743.575274, 79.794206),
        (690.277317, -34.836039),
        (696.624755, -37.787315),
        (734.822854, 44.367006),
        (846.894657, -7.741389),
        (849.845932, -1.393951),
        (737.77413, 50.714444),
        (762.500051, 103.893565),
        (874.571854, 51.78517),
        (877.52313, 58.132608),
        (783.385268, 101.90252),
        (829.5314, 201.151025),
        (923.669263, 157.381114),
        (926.620539, 163.728552),
        (832.482676, 207.498464),
        (877.836304, 305.042494),
        (871.488866, 307.99377),
        (826.135238, 210.449739),
        (764.596157, 239.06267),
        (797.371861, 309.554808),
        (791.024422, 312.506084),
        (743.402652, 210.083859),
        (649.930725, 253.54414),
        (646.97945, 247.196702),
        (740.451376, 203.736421),
        (709.151311, 136.418002),
        (597.079508, 188.526396),
        (594.128232, 182.178958),
    ))

    polygon_difference = polygon_a.boolean_difference(polygon_b, 0.01)
    assert len(polygon_difference) == 10


def test_boolean_split():
    """Test the boolean_split method."""
    polygon_a = Polygon2D.from_array((
        (174.731903, -72.989276),
        (-70.77748, -53.08311),
        (-72.252011, 215.281501),
        (129.021448, 126.809651),
        (106.16622, 28.016086),
        (216.756032, 22.117962),
        (174.731903, -72.989276),
    ))
    polygon_b = Polygon2D.from_array((
        (-169.571046, -98.793566),
        (-145.241287, 63.404826),
        (11.796247, 34.651475),
        (8.10992, -129.758713),
        (-76.675603, -216.018767),
        (-169.571046, -98.793566),
    ))

    poly_int, poly1_dif, poly2_dif = \
        Polygon2D.boolean_split(polygon_a, polygon_b, 0.01)
    assert len(poly_int) == 1
    assert len(poly_int[0].vertices) == 4

    assert len(poly1_dif) == 1
    assert len(poly1_dif[0].vertices) == 8

    assert len(poly2_dif) == 1
    assert len(poly2_dif[0].vertices) == 7


def test_joined_intersected_boundary():
    geo_file = './tests/json/polygons_for_joined_boundary.json'
    with open(geo_file, 'r') as fp:
        geo_dict = json.load(fp)
    polygons = [Polygon2D.from_dict(p) for p in geo_dict]

    bound_polygons = Polygon2D.joined_intersected_boundary(polygons, 0.01)
    assert len(bound_polygons) == 6

    sorted_polys = sorted(bound_polygons, key=lambda x: x.area, reverse=True)
    large_poly = sorted_polys[0]
    for sub_poly in sorted_polys[1:]:
        assert large_poly.is_polygon_inside(sub_poly)


def test_gap_crossing_boundary():
    geo_file = './tests/json/polygons_for_gap_boundary.json'
    with open(geo_file, 'r') as fp:
        geo_dict = json.load(fp)
    polygons = [Polygon2D.from_dict(p) for p in geo_dict]

    bound_polygons = Polygon2D.gap_crossing_boundary(polygons, 0.25, 0.01)
    assert len(bound_polygons) == 1
    assert bound_polygons[0].area > 1600
