# coding=utf-8
"""Classes for computing straight skeleton for 2D concave polygons."""
from __future__ import division

import pytest

from ladybug_geometry.geometry2d.polygon import Polygon2D
from ladybug_geometry.geometry2d.pointvector import Point2D
from ladybug_geometry.geometry2d.line import LineSegment2D

from pprint import pprint as pp
import numpy as np

# For comparison import both polyskel
#import sys
#paths = ['/app/polyskel/']
#for path in paths:
#    if path not in sys.path: sys.path.insert(0,path)

#import polyskel as orig_polyskel
from ladybug_geometry import polyskel as lb_polyskel
from tests.test_data import polyskel as orig_polyskel

def helper_check_lavertex(v1, v2):
    """ Checking equality of different LAVertex properties
    """
    tol = 1e10
    assert (v1.point.x - v2.point.x) < tol and \
           (v1.point.y - v2.point.y) < tol

    assert v1.is_reflex == v2.is_reflex

#TODO: move to updated polyskel once we import lb-geom?
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

# TODO Make this a from_array() method
def helper_mtx2line2d(lst):
    """ List of tuples of point coordinates to LineSegment2D"""
    p1, p2 = lst[0], lst[1]
    line2d = LineSegment2D.from_end_points(
        Point2D(*p1), Point2D(*p2)
        )
    return line2d

# TODO Make this a from_array() method
def helper_mtx2polygon2d(lst):
    polygon = Polygon2D([Point2D(*pt) for pt in lst])
    return polygon


def helper_assert_polygon_equality(polygon, chk_edges, holes=None, lb=True):
    """
    Consumes polygons and holes as a list of list of vertices, and the corresponding
    list of skeleton edges for checking. This function compares the passed chk_edges
    with the equivalent produced in the ladybug-geomtry library and returns a boolean
    if both are equal.

    Args:
         polygon: list of list of polygon vertices as floats in ccw order.
         chk_edges: list of list of line segments as floats.
         holes: list of list of polygon hole vertices as floats in cw order.
         lb: Boolean flag on wheter to use Ladybug or Bottfy's implementation of
         polyskel.
    Returns:
        Boolean of equality.
    """
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
        skel = lb_polyskel.skeletonize(vertices, holes)
    else:
        skel = orig_polyskel.skeletonize(vertices, holes)
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
        is_equal = is_equal and np.allclose(
            np.array(chk_edge.to_array()),
            np.array(tst_edge.to_array()),
            atol = 1e-10
            )
        break

    if not is_equal:
        print('Edges not equal btwn chk_edgs:')
        pp(chk_edges)
        print('and actual edges:')
        pp(tst_edges)
        print('\nSpecifically:')
        pp(chk_edge.to_array())
        print('!=')
        pp(tst_edge.to_array())
        print('\n')
        raise Exception('Equality error.')

    return is_equal

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
    """Test polygon."""
    polygon = [
        [0.,  0.],
        [10., 0.],
        [10.,10.],
        [5., 15.],
        [0., 10.]
    ]
    # Make actual geom that we already sovled
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

def test_polyskel_complex_convex():
    """
    Complicated convex with many edges.
    """

    polygon = [
        [0.,  0. ],
        [2., -1. ],
        [4., -1.5],
        [10., 0. ],
        [11., 5. ],
        [11., 7. ],
        [10.,10. ],
        [5., 15. ],
        [2., 15. ],
        [0., 10. ]
    ]

    # Make actual geom that we already sovled
    chk_edges = [
        [(3.8612649295852073, 12.250850349054728), (2.0, 15.0)],
        [(3.8612649295852073, 12.250850349054728), (5.0, 15.0)],
        [(3.0723397575512923, 1.8988103951543103), (2.0, -1.0)],
        [(3.0723397575512923, 1.8988103951543103), (0.0, 0.0)],
        [(4.000000000000001, 2.6231056256176615), (4.0, -1.5)],
        [(4.000000000000001, 2.6231056256176615),
        (3.0723397575512923, 1.8988103951543103)],
        [(4.501197221853484, 9.133148620085219), (0.0, 10.0)],
        [(4.501197221853484, 9.133148620085219),
        (3.8612649295852073, 12.250850349054728)],
        [(5.336088038559574, 7.117543887292627),
        (4.501197221853484, 9.133148620085219)],
        [(5.336088038559574, 7.117543887292627), (10.0, 10.0)],
        [(5.386560499934439, 4.398979599985999), (10.0, 0.0)],
        [(5.386560499934439, 4.398979599985999),
        (4.000000000000001, 2.6231056256176615)],
        [(5.5, 6.107472869073913), (5.336088038559574, 7.117543887292627)],
        [(5.5, 6.107472869073913), (11.0, 7.0)],
        [(5.5, 5.544607324760315), (5.5, 6.107472869073913)],
        [(5.5, 5.544607324760315), (11.0, 5.0)],
        [(5.5, 5.544607324760315), (5.386560499934439, 4.398979599985999)]
        ]

    assert helper_assert_polygon_equality(polygon, chk_edges, lb=True)


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
        [(1.0, 0.7207592200561265), (0.0, 0.0)],
        [(1.0, 0.7207592200561265), (1.0, 1.0)],
        [(1.0, 0.7207592200561265), (1.0, 0.7207592200561265)],
        [(1.0, 0.7207592200561265), (2.0, 0.0)],
        [(1.0, 0.7207592200561265), (1.0, 0.5)]
        ]

    assert helper_assert_polygon_equality(polygon, chk_edges)
    # print('---- lb test ----')
    # assert helper_assert_polygon_equality(polygon, chk_edges, lb=False)

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
    #print('\n---- lb test ----')
    assert helper_assert_polygon_equality(polygon, chk_edges, lb=True)


def test_polyskel_concave_two_holes():
    """ Test concave with two holes"""
    poly = [
        [0.7,0.2],
        [2,0],
        [2,2],
        [1,1],
        [0,2],
        [0,0]
    ]

    hole1 = [
        [0.6, 1.2],
        [1, 0.8],
        [1.5, 0.6],
        [0.6, 0.6]
    ]

    hole2 = [
        [1.3,0.5],
        [1.5, 0.25],
        [1.1, 0.25]
    ]

    # Make actual geoms we already solved
    chk_edges = [
        [(1.3, 0.5615500122010744), (1.3, 0.5)],
        [(1.0, 0.9), (1.0, 1.0)],
        [(0.5292893218813451, 1.3707106781186549), (1.0, 0.9)],
        [(0.5292893218813451, 1.3707106781186549), (0.6, 1.2)],
        [(1.919258240356725, 0.5192582403567252), (1.5, 0.6)],
        [(0.6872836548685387, 0.402142090490224), (0.7, 0.2)],
        [(0.9297522368841191, 0.3835997375212605),
         (0.6872836548685387, 0.402142090490224)],
        [(0.9297522368841191, 0.3835997375212605), (1.3, 0.5615500122010744)],
        [(1.000478771872252, 0.202167624909425),
        (0.9297522368841191, 0.3835997375212605)],
        [(1.000478771872252, 0.202167624909425), (1.1, 0.25)],
        [(1.712871519717387, 0.14768865830159658),
         (1.000478771872252, 0.202167624909425)],
        [(1.712871519717387, 0.14768865830159658), (1.5, 0.25)],
        [(1.7774704951779594, 0.2593828974950352), (2.0, 0.0)],
        [(1.7774704951779594, 0.2593828974950352),
         (1.712871519717387, 0.14768865830159658)],
        [(0.35570251186280166, 0.35570251186280166), (0.6, 0.6)],
        [(0.35570251186280166, 0.35570251186280166),
         (0.6872836548685387, 0.402142090490224)],
        [(1.7468046131497028, 0.3468046131497027),
         (1.7774704951779594, 0.2593828974950352)],
        [(1.7468046131497028, 0.3468046131497027), (1.3, 0.5615500122010744)],
        [(1.7468046131497028, 0.3468046131497027),
         (1.919258240356725, 0.5192582403567252)],
        [(0.29999999999999993, 1.2757359312880716), (0.0, 2.0)],
        [(0.29999999999999993, 1.2757359312880716),
         (0.5292893218813451, 1.3707106781186549)],
        [(0.30000000000000004, 0.39771899525487936),
         (0.35570251186280166, 0.35570251186280166)],
        [(0.30000000000000004, 0.39771899525487936), (0.0, 0.0)],
        [(0.30000000000000004, 0.39771899525487936),
         (0.29999999999999993, 1.2757359312880716)],
        [(1.5888207055980816, 1.007325370887889), (2.0, 2.0)],
        [(1.5888207055980816, 1.007325370887889),
         (1.919258240356725, 0.5192582403567252)],
        [(1.0659396157980559, 0.9000000000000001),
         (1.5888207055980816, 1.007325370887889)],
        [(1.0659396157980559, 0.9000000000000001), (1.0, 0.8)],
        [(1.0659396157980559, 0.9000000000000001), (1.0, 0.9)]
        ]

    holes = [hole1,hole2]
    assert helper_assert_polygon_equality(poly, chk_edges, holes, lb=True)

if __name__ == "__main__":
    # Base
    test_polygon_init()
    # Convex
    test_polyskel_triangle()
    test_polyskel_square()
    test_polyskel_pentagon()
    test_polyskel_complex_convex()
    # Concave
    test_polyskel_simple_concave()
    test_polyskel_concave()
    test_polyskel_concave_two_holes()
