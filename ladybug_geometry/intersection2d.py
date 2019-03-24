# coding=utf-8
"""Utility functions for computing intersections between geometry objects."""
from __future__ import division

from .geometry2d.pointvector import Point2D


def intersect_line2d_line2d(A, B):
    """Get the intersection between any Ray2D or LineSegment2D objects.

    Args:
        A: A LineSegment2D or Ray2D object.
        B: Another LineSegment2D or Ray2D to intersect.

    Returns:
        Point2D of intersection if it exists. None if no intersection exists.
    """
    d = B.v.y * A.v.x - B.v.x * A.v.y
    if d == 0:
        return None

    dy = A.p.y - B.p.y
    dx = A.p.x - B.p.x
    ua = (B.v.x * dy - B.v.y * dx) / d
    if not A._u_in(ua):
        return None
    ub = (A.v.x * dy - A.v.y * dx) / d
    if not B._u_in(ub):
        return None

    return Point2D(A.p.x + ua * A.v.x, A.p.y + ua * A.v.y)


def closest_point2d_on_line2d(P, L):
    """Get the closest Point2D on a LineSegment2 or Ray2 to the input P.

    Args:
        P: A Point2D object.
        B: A LineSegment2D or Ray2D object along wich the closest point
            will be determined.

    Returns:
        Point2D for the closest point on L to P.
    """
    d = L.v.magnitude_squared
    assert d != 0, '{} length must not equal 0.'.format(L.__class__.__name__)
    u = ((P.x - L.p.x) * L.v.x + (P.y - L.p.y) * L.v.y) / d
    if not L._u_in(u):
        u = max(min(u, 1.0), 0.0)
    return Point2D(L.p.x + u * L.v.x, L.p.y + u * L.v.y)
