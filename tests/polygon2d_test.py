# coding=utf-8

from ladybug_geometry.geometry2d.pointvector import Point2D, Vector2D
from ladybug_geometry.geometry2d.line import LineSegment2D, LineSegment2DImmutable

import unittest
import pytest
import math


class Polygon2DTestCase(unittest.TestCase):
    """Test for Polygon2D"""

    def test_polygon2_init(self):
        """Test the initalization of Polygon2D objects and basic properties."""
        pt = Point2D(2, 0)
        vec = Vector2D(0, 2)
        seg = LineSegment2D(pt, vec)
        str(seg)  # test the string representation of the line segment

        assert seg.p == Point2D(2, 0)
        assert seg.v == Vector2D(0, 2)
        assert seg.p1 == Point2D(2, 0)
        assert seg.p2 == Point2D(2, 2)
        assert seg.length == 2
        assert seg.length_squared == 4

        flip_seg = seg.flipped()
        assert flip_seg.p == Point2D(2, 2)
        assert flip_seg.v == Vector2D(0, -2)

        assert seg.p == Point2D(2, 0)
        seg.flip()
        assert seg.p == Point2D(2, 2)
        assert seg.v == Vector2D(0, -2)


if __name__ == "__main__":
    unittest.main()
