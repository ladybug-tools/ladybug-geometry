# coding=utf-8

from ladybug_geometry.geometry2d.polygon import Polygon2D
from ladybug_geometry.geometry2d.pointvector import Point2D, Point2DImmutable, Vector2D
from ladybug_geometry.geometry2d.line import LineSegment2DImmutable

import unittest
import pytest


class Polygon2DTestCase(unittest.TestCase):
    """Test for Polygon2D"""

    def test_polygon2d_init(self):
        """Test the initalization of Polygon2D objects and basic properties."""
        pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
        polygon = Polygon2D(pts)

        str(polygon)  # test the string representation of the ray

        assert isinstance(polygon.vertices, tuple)
        assert len(polygon.vertices) == 4
        for point in polygon.vertices:
            assert isinstance(point, Point2DImmutable)

        assert isinstance(polygon.segments, tuple)
        assert len(polygon.segments) == 4
        for seg in polygon.segments:
            assert isinstance(seg, LineSegment2DImmutable)
            assert seg.length == 2

        assert polygon.area == 4
        assert polygon.perimeter == 8
        assert polygon.is_clockwise is False
        assert polygon.is_convex is True
        assert polygon.is_self_intersecting is False

    def test_polygon2d_init_from_rectangle(self):
        """Test the initalization of Polygon2D from_rectangle."""
        polygon = Polygon2D.from_rectangle(Point2D(0, 0), Vector2D(0, 1), 2, 2)

        str(polygon)  # test the string representation of the ray

        assert isinstance(polygon.vertices, tuple)
        assert len(polygon.vertices) == 4
        for point in polygon.vertices:
            assert isinstance(point, Point2DImmutable)

        assert isinstance(polygon.segments, tuple)
        assert len(polygon.segments) == 4
        for seg in polygon.segments:
            assert isinstance(seg, LineSegment2DImmutable)
            assert seg.length == 2

        assert polygon.area == 4
        assert polygon.perimeter == 8
        assert polygon.is_clockwise is True
        assert polygon.is_convex is True
        assert polygon.is_self_intersecting is False

    def test_polygon2d_init_from_shape_with_hole(self):
        """Test the initalization of Polygon2D from_shape_with_hole."""
        bound_pts = [Point2D(0, 0), Point2D(4, 0), Point2D(4, 4), Point2D(0, 4)]
        hole_pts = [Point2D(1, 1), Point2D(3, 1), Point2D(3, 3), Point2D(1, 3)]
        polygon = Polygon2D.from_shape_with_hole(bound_pts, hole_pts)

        str(polygon)  # test the string representation of the ray

        assert isinstance(polygon.vertices, tuple)
        assert len(polygon.vertices) == 10
        for point in polygon.vertices:
            assert isinstance(point, Point2DImmutable)

        assert isinstance(polygon.segments, tuple)
        assert len(polygon.segments) == 10
        for seg in polygon.segments:
            assert isinstance(seg, LineSegment2DImmutable)

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

        str(polygon)  # test the string representation of the ray

        assert isinstance(polygon.vertices, tuple)
        assert len(polygon.vertices) == 16
        for point in polygon.vertices:
            assert isinstance(point, Point2DImmutable)

        assert isinstance(polygon.segments, tuple)
        assert len(polygon.segments) == 16
        for seg in polygon.segments:
            assert isinstance(seg, LineSegment2DImmutable)

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

    def test_is_self_intersecting(self):
        """Test the is_self_intersecting property."""
        pts_1 = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
        polygon_1 = Polygon2D(pts_1)
        pts_2 = (Point2D(0, 0), Point2D(0, 2), Point2D(2, 0), Point2D(2, 2))
        polygon_2 = Polygon2D(pts_2)

        assert polygon_1.is_self_intersecting is False
        assert polygon_2.is_self_intersecting is True


if __name__ == "__main__":
    unittest.main()
