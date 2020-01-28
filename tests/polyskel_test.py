# coding=utf-8
import pytest

from ladybug_geometry.geometry2d.polygon import Polygon2D
from ladybug_geometry.geometry2d.pointvector import Point2D, Vector2D
from ladybug_geometry.geometry2d.line import LineSegment2D
from ladybug_geometry.geometry2d.ray import Ray2D

import math


def test_polyskel():
    """Test the initalization of Polygon2D objects and basic properties."""
    pts = (Point2D(0, 0),
           Point2D(2, 0),
           Point2D(2, 2),
           Point2D(0, 2)) # square

    polygon = Polygon2D(pts)

    # assert isinstance(polygon.vertices, tuple)
    # assert len(polygon.vertices) == 4
    # assert len(polygon) == 4
    # for point in polygon:
    #     assert isinstance(point, Point2D)
    #
    # assert isinstance(polygon.segments, tuple)
    # assert len(polygon.segments) == 4
    # for seg in polygon.segments:
    #     assert isinstance(seg, LineSegment2D)
    #     assert seg.length == 2
    #
    # assert polygon.area == 4
    # assert polygon.perimeter == 8
    # assert polygon.is_clockwise is False
    # assert polygon.is_convex is True
    # assert polygon.is_self_intersecting is False
    #
    # assert polygon.vertices[0] == polygon[0]
