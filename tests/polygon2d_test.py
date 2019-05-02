# coding=utf-8

from ladybug_geometry.geometry2d.pointvector import Point2D
from ladybug_geometry.geometry2d.polygon import Polygon2D

import unittest
import pytest
import math


class Polygon2DTestCase(unittest.TestCase):
    """Test for Polygon2D"""

    def test_polygon2_init(self):
        """Test the initalization of Polygon2D objects and basic properties."""
        pts = (Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2))
        polygon = Polygon2D(pts)

        str(polygon)  # test the string representation of the ray

        assert isinstance(polygon.vertices, tuple)
        assert len(polygon.vertices) == 4


if __name__ == "__main__":
    unittest.main()
