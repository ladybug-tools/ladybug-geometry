# coding=utf-8
import pytest

from ladybug_geometry.geometry2d.polygon import Polygon2D
from ladybug_geometry.geometry2d.pointvector import Point2D, Vector2D
from ladybug_geometry.geometry2d.line import LineSegment2D
from ladybug_geometry.geometry2d.ray import Ray2D

import math

from pprint import pprint as pp

import sys
paths = ['/app/polyskel/']
for path in paths:
    if path not in sys.path: sys.path.insert(0,path)
import polyskel

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

    # Make into euclid classes
    polygon = helper_mtx2polygon2d(polygon)
    chk_edges = [helper_mtx2line2d(edge) for edge in chk_edges]

    #TODO: temp until we updted polyskel with lb geom
    vertices = [(v.x,v.y) for v in polygon.vertices]

    # Run function
    skel = polyskel.skeletonize(vertices, [])
    tst_edges = skeleton_edges(skel)

    # Tests
    # Check types
    assert isinstance(tst_edges, list)
    for edge in tst_edges:
        assert isinstance(edge, LineSegment2D)

    # Check lines
    # Assumes order is constant
    for chk_edge, tst_edge in zip(chk_edges, tst_edges):
        assert chk_edge == tst_edge


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

    # Make into euclid classes
    polygon = helper_mtx2polygon2d(polygon)
    chk_edges = [helper_mtx2line2d(edge) for edge in chk_edges]

    #TODO: temp until we updted polyskel with lb geom
    vertices = [(v.x,v.y) for v in polygon.vertices]

    # Run function
    skel = polyskel.skeletonize(vertices, [])
    tst_edges = skeleton_edges(skel)

    # Tests
    # Check types
    assert isinstance(tst_edges, list)
    for edge in tst_edges:
        assert isinstance(edge, LineSegment2D)

    # Check lines
    # Assumes order is constant
    for chk_edge, tst_edge in zip(chk_edges, tst_edges):
        assert chk_edge == tst_edge

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
    holes = [hole1,hole2]

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

    # Make into euclid classes
    polygon = helper_mtx2polygon2d(polygon)
    holes = [helper_mtx2polygon2d(hole) for hole in holes]
    chk_edges = [helper_mtx2line2d(edge) for edge in chk_edges]

    #TODO: temp until we updted polyskel with lb geom
    vertices = [(v.x,v.y) for v in polygon.vertices]

    # Run function
    skel = polyskel.skeletonize(vertices, holes)
    tst_edges = skeleton_edges(skel)

    # Tests
    # Check types
    assert isinstance(tst_edges, list)
    for edge in tst_edges:
        assert isinstance(edge, LineSegment2D)

    # Check lines
    # Assumes order is constant
    for chk_edge, tst_edge in zip(chk_edges, tst_edges):
        assert chk_edge == tst_edge



if __name__ == "__main__":
    test_polyskel_triangle()
    test_polyskel_concave()
    test_polyskel_concave_two_holes()