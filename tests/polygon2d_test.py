# coding=utf-8

from ladybug_geometry.geometry2d.polygon import Polygon2D
from ladybug_geometry.geometry2d.pointvector import Point2D, Vector2D
from ladybug_geometry.geometry2d.line import LineSegment2D
from ladybug_geometry.geometry2d.ray import Ray2D

import unittest
import pytest
import math


class Polygon2DTestCase(unittest.TestCase):
    """Test for Polygon2D"""

    def test_polygon2d_init(self):
        """Test the initalization of Polygon2D objects and basic properties."""
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
        assert polygon.is_clockwise is False
        assert polygon.is_convex is True
        assert polygon.is_self_intersecting is False

        assert polygon.vertices[0] == polygon[0]

    def test_polygon2d_to_from_dict(self):
        """Test the to/from dict of Polygon2D objects."""
        pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
        polygon = Polygon2D(pts)
        polygon_dict = polygon.to_dict()
        new_polygon = Polygon2D.from_dict(polygon_dict)
        assert isinstance(new_polygon, Polygon2D)
        assert new_polygon.to_dict() == polygon_dict

    def test_polygon2d_init_from_rectangle(self):
        """Test the initalization of Polygon2D from_rectangle."""
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
        assert polygon.is_clockwise is True
        assert polygon.is_convex is True
        assert polygon.is_self_intersecting is False

    def test_polygon2d_init_from_regular_polygon(self):
        """Test the initalization of Polygon2D from_regular_polygon."""
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
        assert polygon.is_clockwise is False
        assert polygon.is_convex is True
        assert polygon.is_self_intersecting is False

        polygon = Polygon2D.from_regular_polygon(3)
        assert len(polygon.vertices) == 3
        polygon = Polygon2D.from_regular_polygon(20)
        assert len(polygon.vertices) == 20
        with pytest.raises(AssertionError):
            polygon = Polygon2D.from_regular_polygon(2)

    def test_polygon2d_init_from_shape_with_hole(self):
        """Test the initalization of Polygon2D from_shape_with_hole."""
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
        assert polygon.is_clockwise is False
        assert polygon.is_convex is False
        assert polygon.is_self_intersecting is False

    def test_polygon2d_init_from_shape_with_holes(self):
        """Test the initalization of Polygon2D from_shape_with_holes."""
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
        assert polygon.is_clockwise is False
        assert polygon.is_convex is False
        assert polygon.is_self_intersecting is False

    def test_clockwise(self):
        """Test the clockwise property."""
        pts_1 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
        polygon_1 = Polygon2D(pts_1)
        pts_2 = (Point2D(0, 0), Point2D(0, 2), Point2D(2, 2), Point2D(2, 0))
        polygon_2 = Polygon2D(pts_2)

        assert polygon_1.is_clockwise is False
        assert polygon_2.is_clockwise is True
        assert polygon_1.reverse().is_clockwise is True

        assert polygon_1.area == 4
        assert polygon_2.area == 4

    def test_is_convex(self):
        """Test the convex property."""
        pts_1 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
        polygon_1 = Polygon2D(pts_1)
        pts_2 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 1), Point2D(1, 1),
                 Point2D(1, 2), Point2D(0, 2))
        polygon_2 = Polygon2D(pts_2)

        assert polygon_1.is_convex is True
        assert polygon_2.is_convex is False

    def test_is_self_intersecting(self):
        """Test the is_self_intersecting property."""
        pts_1 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
        polygon_1 = Polygon2D(pts_1)
        pts_2 = (Point2D(0, 0), Point2D(0, 2), Point2D(2, 0), Point2D(2, 2))
        polygon_2 = Polygon2D(pts_2)

        assert polygon_1.is_self_intersecting is False
        assert polygon_2.is_self_intersecting is True

    def test_is_valid(self):
        """Test the is_valid property."""
        pts_1 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2))
        pts_2 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 0))
        polygon_1 = Polygon2D(pts_1)
        polygon_2 = Polygon2D(pts_2)

        assert polygon_1.is_valid is True
        assert polygon_2.is_valid is False

    def test_min_max_center(self):
        """Test the Polygon2D min, max and center."""
        pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
        polygon = Polygon2D(pts)

        assert polygon.min == Point2D(0, 0)
        assert polygon.max == Point2D(2, 2)
        assert polygon.center == Point2D(1, 1)

    def test_remove_colinear_vertices(self):
        """Test the remove_colinear_vertices method of Polygon2D."""
        pts_1 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
        pts_2 = (Point2D(0, 0), Point2D(1, 0), Point2D(2, 0), Point2D(2, 2),
                 Point2D(0, 2))
        polygon_1 = Polygon2D(pts_1)
        polygon_2 = Polygon2D(pts_2)

        assert len(polygon_1.remove_colinear_vertices(0.0001).vertices) == 4
        assert len(polygon_2.remove_colinear_vertices(0.0001).vertices) == 4

    def test_polygon2d_duplicate(self):
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

    def test_move(self):
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

    def test_scale(self):
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

    def test_scale_world_origin(self):
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

    def test_rotate(self):
        """Test the Polygon2D rotate method."""
        pts = (Point2D(1, 1), Point2D(2, 1), Point2D(2, 2), Point2D(1, 2))
        polygon = Polygon2D(pts)
        origin_1 = Point2D(1, 1)

        test_1 = polygon.rotate(math.pi, origin_1)
        assert test_1[0].x == pytest.approx(1, rel=1e-3)
        assert test_1[0].y == pytest.approx(1, rel=1e-3)
        assert test_1[2].x == pytest.approx(0, rel=1e-3)
        assert test_1[2].y == pytest.approx(0, rel=1e-3)

        test_2 = polygon.rotate(math.pi/2, origin_1)
        assert test_2[0].x == pytest.approx(1, rel=1e-3)
        assert test_2[0].y == pytest.approx(1, rel=1e-3)
        assert test_2[2].x == pytest.approx(0, rel=1e-3)
        assert test_2[2].y == pytest.approx(2, rel=1e-3)

    def test_reflect(self):
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

    def test_intersect_line_ray(self):
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

    def test_intersect_line_infinite(self):
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

    def test_point_relationship(self):
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

    def test_is_point_inside(self):
        """Test the Polygon2D is_point_inside method."""
        pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
        polygon = Polygon2D(pts)

        assert polygon.is_point_inside(Point2D(-1, 1)) is False
        assert polygon.is_point_inside(Point2D(1, -1)) is False
        assert polygon.is_point_inside(Point2D(1, 3)) is False
        assert polygon.is_point_inside(Point2D(3, 1)) is False
        assert polygon.is_point_inside(Point2D(1, 1)) is True

        assert polygon.is_point_inside(Point2D(-1, 1), Vector2D(0, 1)) is False
        assert polygon.is_point_inside(Point2D(1, -1), Vector2D(0, 1)) is False
        assert polygon.is_point_inside(Point2D(1, 3), Vector2D(0, 1)) is False
        assert polygon.is_point_inside(Point2D(3, 1), Vector2D(0, 1)) is False
        assert polygon.is_point_inside(Point2D(1, 1), Vector2D(0, 1)) is True

    def test_is_point_inside_bound_rect(self):
        """Test the Polygon2D is_point_inside_bound_rect method."""
        pts = (Point2D(0, 0), Point2D(4, 0), Point2D(4, 2), Point2D(2, 2),
               Point2D(2, 4), Point2D(0, 4))
        polygon = Polygon2D(pts)

        assert polygon.is_point_inside_bound_rect(Point2D(-1, 1)) is False
        assert polygon.is_point_inside_bound_rect(Point2D(1, -1)) is False
        assert polygon.is_point_inside_bound_rect(Point2D(1, 5)) is False
        assert polygon.is_point_inside_bound_rect(Point2D(5, 1)) is False
        assert polygon.is_point_inside_bound_rect(Point2D(3, 3)) is False
        assert polygon.is_point_inside(Point2D(1, 1)) is True

    def test_is_polygon_inside_outside(self):
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

        assert polygon.is_polygon_inside(hole_1) is True
        assert polygon.is_polygon_inside(hole_2) is True
        assert polygon.is_polygon_inside(hole_3) is False
        assert polygon.is_polygon_inside(hole_4) is False

        assert polygon.is_polygon_outside(hole_1) is False
        assert polygon.is_polygon_outside(hole_2) is False
        assert polygon.is_polygon_outside(hole_3) is False
        assert polygon.is_polygon_outside(hole_4) is True


if __name__ == "__main__":
    unittest.main()
