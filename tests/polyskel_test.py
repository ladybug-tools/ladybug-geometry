# coding=utf-8
"""Classes for computing straight skeleton for 2D polygons."""
from __future__ import division

import pytest

from ladybug_geometry.geometry2d.polygon import Polygon2D
from ladybug_geometry.geometry2d.pointvector import Point2D
from ladybug_geometry.geometry2d.line import LineSegment2D
from ladybug_geometry import polyskel as lb_polyskel

import math
from pprint import pprint as pp

# TODO Figure out if we need this later
# # For comparison
# import sys
# paths = ['/app/polyskel/']
# for path in paths:
#     if path not in sys.path: sys.path.insert(0,path)
# import polyskel as orig_polyskel

#TODO: move to updated polyskel once we import lb-geom
def skeleton_edges(skeleton):
    """
    Consumes list of polyskeleton subtrees.
    Skeleton edges are the segments defined by source point and each sink points.

    Args:
        skeleton: list of polyskel.Subtree, which are namedTuples of consisting of
        a source point, and list of sink points.

    Returns:
        list of LineSegment2Ds
    """
    L = []
    for subtree in skeleton:
        source = subtree.source
        for sink in subtree.sinks:
            edge = LineSegment2D.from_end_points(
                Point2D(source[0], source[1]),
                Point2D(sink[0],sink[1])
            )
            L.append(edge)
    return L

def helper_mtx2line2d(lst):
    """ List of tuples of point coordinates to LineSegment2D"""
    p1, p2 = lst[0], lst[1]
    line2d = LineSegment2D.from_end_points(
        Point2D(*p1), Point2D(*p2)
        )
    return line2d

def helper_mtx2polygon2d(lst):
    polygon = Polygon2D([Point2D(*pt) for pt in lst])
    return polygon


def helper_assert_polygon_equality(polygon, chk_edges, holes=None, lb=True):
    # Make into euclid classes
    if holes is None:
        holes = []
    else:
        holes = [helper_mtx2polygon2d(hole) for hole in holes]

    polygon = helper_mtx2polygon2d(polygon)
    chk_edges = [helper_mtx2line2d(edge) for edge in chk_edges]

    #TODO: temp until we updted polyskel with lb geom?
    vertices = [(v.x,v.y) for v in polygon.vertices]

    # Run function
    if lb:
        skel = lb_polyskel.skeletonize(vertices, [])
    else:
        skel = orig_polyskel.skeletonize(vertices, [])
    tst_edges = skeleton_edges(skel)

    # Tests
    # Check types
    is_instant = isinstance(tst_edges, list)
    for edge in tst_edges:
        is_instant = is_instant and isinstance(edge, LineSegment2D)

    if not is_instant:
        print('Instants not match')
        raise Exception('Instance error.')

    # Check lines
    # Assumes order is constant
    is_equal = True
    for chk_edge, tst_edge in zip(chk_edges, tst_edges):
        is_equal = is_equal and chk_edge == tst_edge
        break

    if not is_equal:
        print('Edges not equal btwn chk_edgs:')
        pp(chk_edges)
        print('and actual edges:')
        pp(tst_edges)
        print('\nSpecifically:')
        pp(chk_edge)
        print('!=')
        pp(tst_edge)
        print('\n')
        raise Exception('Equality error.')

    return is_equal

def test_polyskel_triangle():
    """Test simplest geometry, a triangle."""
    polygon = [
        [0.,0.],
        [7.,0.],
        [4.,4.]
    ]
    # Make actual geom that we already sovled
    chk_edges = [
        [(3.8284271247461903, 1.5857864376269049), (4.0, 4.0)],
        [(3.8284271247461903, 1.5857864376269049), (7.0, 0.0)],
        [(3.8284271247461903, 1.5857864376269049), (0.0, 0.0)]
    ]

    assert helper_assert_polygon_equality(polygon, chk_edges)


def test_polyskel_square():
    """Test square."""
    polygon = [
        [0.,  0.],
        [10., 0.],
        [10.,10.],
        [0., 10.]
    ]
    # Make actual geom that we already sovled
    chk_edges = [
        [(5.0, 5.0), (0.,   0.)],
        [(5.0, 5.0), (0.,  10.)],
        [(5.0, 5.0), (5.0, 5.0)],
        [(5.0, 5.0), (10., 10.)],
        [(5.0, 5.0), (10.,  0.)]
    ]

    assert helper_assert_polygon_equality(polygon, chk_edges)

def test_polyskel_pentagon():
    """Test concave."""
    polygon = [
        [0.,  0.],
        [10., 0.],
        [10.,10.],
        [5., 15.],
        [0., 10.]
    ]
    # Make actual geom that we already sovled
    v1 = 7.9289321881345245
    chk_edges = [
        [(5.0, 7.9289321881345245), (0.,  10.)],
        [(5.0, 7.9289321881345245), (5.0, 15.)],
        [(5.0, 7.9289321881345245), (5., 7.9289321881345245)],
        [(5.0, 7.9289321881345245), (10., 10.)],
        [(5.0, 5.0),                (5., 7.9289321881345245)],
        [(5.0, 5.0),                (10.,  0.)],
        [(5.0, 5.0),                (0.,   0.)]
    ]

    assert helper_assert_polygon_equality(polygon, chk_edges, lb=True)

# def test_polyskel_concave():
#     """ Test rectangle with side poked in."""
#     polygon = [
#         [0,0],
#         [2,0],
#         [2,2],
#         [1,1],
#         [0,2]
#     ]
#
#     # Make actual geom that we already solved
#     chk_edges = [
#         [(1.0, 0.41421356237309503), (1.0, 1.0)],
#         [(0.585786437626905, 0.585786437626905), (1.0, 0.41421356237309503)],
#         [(0.585786437626905, 0.585786437626905), (0.0, 0.0)],
#         [(0.585786437626905, 0.585786437626905), (0.0, 2.0)],
#         [(1.414213562373095, 0.585786437626905), (1.0, 0.41421356237309503)],
#         [(1.414213562373095, 0.585786437626905), (2.0, 2.0)],
#         [(1.414213562373095, 0.585786437626905), (2.0, 0.0)]
#     ]
#
#     assert helper_assert_polygon_equality(polygon, chk_edges)
#
# def test_polyskel_concave_two_holes():
#     """ Test concave with two holes"""
#     polygon = [
#         [0,0],
#         [0,2],
#         [1,1],
#         [2,2],
#         [2,0],
#         [0.7,0.2]
#     ]
#
#     hole1 = [
#         [0.6, 0.6],
#         [1.5, 0.6],
#         [1, 0.8],
#         [0.6, 1.2]
#     ]
#     hole2 = [
#         [1.1, 0.25],
#         [1.5, 0.25],
#         [1.3,0.5]
#     ]
#
#     # Make actual geoms we already solved
#     chk_edges = [
#         [(1.3, 0.3461249694973139), (1.3, 0.5)],
#         [(1.3, 0.3461249694973139), (1.5, 0.25)],
#         [(1.3, 0.3461249694973139), (1.1, 0.25)],
#         [(0.9393204034059653, 0.7079770243431964), (1.0, 0.8)],
#         [(0.9393204034059653, 0.7079770243431964), (1.5, 0.6)],
#         [(0.7757359312880715, 0.7757359312880714), (0.6, 1.2)],
#         [(0.7757359312880715, 0.7757359312880714),
#          (0.9393204034059653, 0.7079770243431964)],
#         [(0.7757359312880715, 0.7757359312880714), (0.6, 0.6)]
#     ]
#
#     holes = [hole1,hole2]
#     assert helper_assert_polygon_equality(polygon, chk_edges, holes)

if __name__ == "__main__":
    test_polyskel_triangle()
    test_polyskel_square()
    test_polyskel_pentagon()
    #test_polyskel_concave()
    #test_polyskel_concave_two_holes()