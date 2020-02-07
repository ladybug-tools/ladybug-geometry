# coding=utf-8
"""Classes for computing straight skeleton for 2D concave polygons."""
from __future__ import division

import pytest

from ladybug_geometry.geometry2d.polygon import Polygon2D
from ladybug_geometry.geometry2d.pointvector import Point2D
from ladybug_geometry.geometry2d.line import LineSegment2D
from ladybug_geometry import polyskel as lb_polyskel

from pprint import pprint as pp
import numpy as np

# from tests.polyskel_convex_test import helper_mtx2line2d, helper_mtx2polygon2d, \
#     helper_assert_polygon_equality, skeleton_edges


# TODO Figure out if we need this later
# For comparison
import sys
paths = ['/app/polyskel/']
for path in paths:
    if path not in sys.path: sys.path.insert(0,path)

import polyskel as orig_polyskel

def test_polygon_init():
    """ Test rectangle init."""
    polygon = [
        [0,0],
        [2,0],
        [2,2],
        [1,1],
        [0,2]
    ]
    # With LB
    SLAV = lb_polyskel._SLAV(polygon, [])
    contour = lb_polyskel._normalize_contour(polygon)
    lav = lb_polyskel._LAV.from_polygon(contour, SLAV)

    # With LB
    SLAV = lb_polyskel._SLAV(polygon, [])
    contour = lb_polyskel._normalize_contour(polygon)
    lb_lav = lb_polyskel._LAV.from_polygon(contour, SLAV)

    # With orig
    SLAV = orig_polyskel._SLAV(polygon, [])
    contour = orig_polyskel._normalize_contour(polygon)
    orig_lav = orig_polyskel._LAV.from_polygon(contour, SLAV)

    # iterate through LAV
    for i in range(4):
        if i == 0:
            vertex1 = lb_lav.head
            vertex2 = orig_lav.head
        else:
            vertex1 = vertex1.next
            vertex2 = vertex2.next

        # Compare
        helper_check_lavertex(vertex1, vertex2)

def helper_check_lavertex(v1, v2):
    """ Checking equality of different LAVertex properties
    """
    tol = 1e10
    assert (v1.point.x - v2.point.x) < tol and \
           (v1.point.y - v2.point.y) < tol

    assert v1.is_reflex == v2.is_reflex



if __name__ == "__main__":
    test_polygon_init()

