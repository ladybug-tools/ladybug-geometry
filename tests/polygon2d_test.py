# coding=utf-8
import pytest

from ladybug_geometry.geometry2d.polygon import Polygon2D
from ladybug_geometry.geometry2d.pointvector import Point2D, Vector2D
from ladybug_geometry.geometry2d.line import LineSegment2D
from ladybug_geometry.geometry2d.ray import Ray2D

import math


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
    assert not polygon.is_self_intersecting


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
    assert not polygon.is_self_intersecting


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
    pts1 = [Point2D(-154.33, -272.63), Point2D(-155.30, -276.70), Point2D(-151.23, -277.68),
            Point2D(-150.26, -273.61), Point2D(-146.19, -274.58), Point2D(-145.22, -270.51),
            Point2D(-149.29, -269.54), Point2D(-148.31, -265.47), Point2D(-152.38, -264.50),
            Point2D(-153.35, -268.57), Point2D(-157.42, -267.59), Point2D(-158.39, -271.66)]
    pts2 = [Point2D(-161.49, -266.62), Point2D(-153.35, -268.57), Point2D(-151.41, -260.43),
            Point2D(-159.54, -258.49)]
    pts3 = [Point2D(-164.41, -278.82), Point2D(-156.27, -280.77), Point2D(-154.33, -272.63),
            Point2D(-162.46, -270.69)]
    pts4 = [Point2D(-149.29, -269.54), Point2D(-141.15, -271.49), Point2D(-139.20, -263.35),
            Point2D(-147.34, -261.41)]
    pts5 = [Point2D(-152.21, -281.74), Point2D(-144.07, -283.69), Point2D(-142.12, -275.55),
            Point2D(-150.26, -273.61)]

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
    """ Test if polygons are equivalent based on point equivalence"""

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
