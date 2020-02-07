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

from tests.polyskel_convex_test import helper_mtx2line2d, helper_mtx2polygon2d, \
    helper_assert_polygon_equality, skeleton_edges


# TODO Figure out if we need this later
# For comparison
import sys
paths = ['/app/polyskel/']
for path in paths:
    if path not in sys.path: sys.path.insert(0,path)

import polyskel as orig_polyskel

def test_polyskel_simple_concave():
    """ Test simplest possible concave polygon: triangle
    with poked in side."""

    polygon = [
        [0.,  0. ],
        [1.,  0.5],
        [2.,  0. ],
        [1.,  1. ],
    ]

    # Make actual geom that we already solved
    chk_edges = [
        [(1.0, 0.7207592200561265), (1.0, 1.0)],
        [(1.0, 0.7207592200561265), (2.0, 0.0)],
        [(1.0, 0.7207592200561265), (1.0, 0.7207592200561265)],
        [(1.0, 0.7207592200561265), (1.0, 0.5)],
        [(1.0, 0.7207592200561265), (0.0, 0.0)]
        ]

    assert helper_assert_polygon_equality(polygon, chk_edges)
    #print('---- lb test ----')
    assert helper_assert_polygon_equality(polygon, chk_edges, lb=False)

def test_polyskel_concave():
    """ Test rectangle with side poked in."""
    polygon = [
        [0,0],
        [2,0],
        [2,2],
        [1,1],
        [0,2]
    ]

    # Make actual geom that we already solved
    chk_edges = [
        [(1.0, 0.41421356237309503), (1.0, 1.0)],
        [(0.585786437626905, 0.585786437626905), (1.0, 0.41421356237309503)],
        [(0.585786437626905, 0.585786437626905), (0.0, 0.0)],
        [(0.585786437626905, 0.585786437626905), (0.0, 2.0)],
        [(1.414213562373095, 0.585786437626905), (1.0, 0.41421356237309503)],
        [(1.414213562373095, 0.585786437626905), (2.0, 2.0)],
        [(1.414213562373095, 0.585786437626905), (2.0, 0.0)]
    ]

    assert helper_assert_polygon_equality(polygon, chk_edges, lb=False)
    print('\n---- lb test ----')
    assert helper_assert_polygon_equality(polygon, chk_edges, lb=True)


def test_polyskel_concave_two_holes():
    """ Test concave with two holes"""
    polygon = [
        [0,0],
        [0,2],
        [1,1],
        [2,2],
        [2,0],
        [0.7,0.2]
    ]

    hole1 = [
        [0.6, 0.6],
        [1.5, 0.6],
        [1, 0.8],
        [0.6, 1.2]
    ]
    hole2 = [
        [1.1, 0.25],
        [1.5, 0.25],
        [1.3,0.5]
    ]

    # Make actual geoms we already solved
    chk_edges = [
        [(1.3, 0.3461249694973139), (1.3, 0.5)],
        [(1.3, 0.3461249694973139), (1.5, 0.25)],
        [(1.3, 0.3461249694973139), (1.1, 0.25)],
        [(0.9393204034059653, 0.7079770243431964), (1.0, 0.8)],
        [(0.9393204034059653, 0.7079770243431964), (1.5, 0.6)],
        [(0.7757359312880715, 0.7757359312880714), (0.6, 1.2)],
        [(0.7757359312880715, 0.7757359312880714),
         (0.9393204034059653, 0.7079770243431964)],
        [(0.7757359312880715, 0.7757359312880714), (0.6, 0.6)]
    ]

    holes = [hole1,hole2]
    assert helper_assert_polygon_equality(polygon, chk_edges, holes)



if __name__ == "__main__":
    test_polyskel_simple_concave()
    test_polyskel_concave()
    test_polyskel_concave_two_holes()
