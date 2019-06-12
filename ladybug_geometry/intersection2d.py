# coding=utf-8
"""Utility functions for computing intersections between geometry in 2D space."""
from __future__ import division

from .geometry2d.pointvector import Point2D, Vector2D

import math


def intersect_line2d(A, B):
    """Get the intersection between any Ray2D or LineSegment2D objects as a Point2D.

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


def intersect_line2d_infinite(A, B):
    """Get the intersection between a Ray2D/LineSegment2D and another extended infinitely.

    Args:
        A: A LineSegment2D or Ray2D object.
        B: ALineSegment2D or Ray2D that will be extended infinitely for intersection.

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
    return Point2D(A.p.x + ua * A.v.x, A.p.y + ua * A.v.y)


def does_intersection_exist_line2d(A, B):
    """Boolean denoting whether an intersection exists between Ray2D or LineSegment2D.

    This is slightly faster than actually computing the intersection but should only be
    used in cases where the actual point of intersection is not needed.

    Args:
        A: A LineSegment2D or Ray2D object.
        B: Another LineSegment2D or Ray2D to intersect.

    Returns:
        True if an intersection exists. False if no intersection exists.
    """
    d = B.v.y * A.v.x - B.v.x * A.v.y
    if d == 0:
        return False
    dy = A.p.y - B.p.y
    dx = A.p.x - B.p.x
    ua = (B.v.x * dy - B.v.y * dx) / d
    if not A._u_in(ua):
        return False
    ub = (A.v.x * dy - A.v.y * dx) / d
    if not B._u_in(ub):
        return False
    return True


def intersect_line2d_arc2d(L, C):
    """Get the intersection between any Ray2D/LineSegment2D and an Arc2D.

    Args:
        L: A LineSegment2D or Ray2D object.
        C: An Arc2D object along which the closest point will be determined.

    Returns:
        A list of 2 Point2D objects if a full intersection exists.
        A list with a single Point2D object if the line is tangent or intersects
        only once. None if no intersection exists.
    """
    a = L.v.magnitude_squared
    b = 2 * (L.v.x * (L.p.x - C.c.x) + L.v.y * (L.p.y - C.c.y))
    c = C.c.magnitude_squared + L.p.magnitude_squared - 2 * C.c.dot(L.p) - C.r ** 2
    det = b ** 2 - 4 * a * c
    if det < 0:
        return None
    sq = math.sqrt(det)
    u1 = (-b + sq) / (2 * a)
    u2 = (-b - sq) / (2 * a)
    pt1 = Point2D(L.p.x + u1 * L.v.x, L.p.y + u1 * L.v.y) if L._u_in(u1) else None
    pt2 = Point2D(L.p.x + u2 * L.v.x, L.p.y + u2 * L.v.y) if L._u_in(u2) else None

    if u1 == u2:  # Tangent
        pt = Point2D(L.p.x + u1 * L.v.x, L.p.y + u1 * L.v.y)
        return pt if C._pt_in(pt) else None

    pts = [p for p in (pt1, pt2) if p is not None and C._pt_in(p)]
    return pts if len(pts) != 0 else None


def intersect_line2d_infinite_arc2d(L, C):
    """Get the intersection between an Arc2D and a Ray2D/LineSegment2D extended infinitely.

    Args:
        L: A LineSegment2D or Ray2D that will be extended infinitely for intersection.
        C: An Arc2D object along which the closest point will be determined.

    Returns:
        A list of 2 Point2D objects if a full intersection exists.
        A list with a single Point2D object if the line is tangent or intersects
        only once. None if no intersection exists.
    """
    a = L.v.magnitude_squared
    b = 2 * (L.v.x * (L.p.x - C.c.x) + L.v.y * (L.p.y - C.c.y))
    c = C.c.magnitude_squared + L.p.magnitude_squared - 2 * C.c.dot(L.p) - C.r ** 2
    det = b ** 2 - 4 * a * c
    if det < 0:
        return None
    sq = math.sqrt(det)
    u1 = (-b + sq) / (2 * a)
    u2 = (-b - sq) / (2 * a)

    if u1 == u2:  # Tangent
        pt = Point2D(L.p.x + u1 * L.v.x, L.p.y + u1 * L.v.y)
        return pt if C._pt_in(pt) else None

    pt1 = Point2D(L.p.x + u1 * L.v.x, L.p.y + u1 * L.v.y)
    pt2 = Point2D(L.p.x + u2 * L.v.x, L.p.y + u2 * L.v.y)
    pts = [p for p in (pt1, pt2) if C._pt_in(p)]
    return pts if len(pts) != 0 else None


def closest_point2d_on_line2d(P, L):
    """Get the closest Point2D on a LineSegment2D or Ray2D to the input P.

    Args:
        P: A Point2D object.
        L: A LineSegment2D or Ray2D object along which the closest point
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


def closest_point2d_between_line2d(A, B):
    """Get the two closest Point2D between two LineSegment2D objects.

    Note that the line segments should not intersect for the result to be valid.

    Args:
        Args:
            A: A LineSegment2D object.
            B: Another LineSegment2D to which closest points will be determined.

    Returns:
        dists[0]: The distance between the two LineSegment2D objects.
        pts[0]: A tuple of two Point2D objects representing:
            1) The point on A that is closest to B
            2) The point on B that is closest to A
    """
    # one of the 4 endpoints must be a closest point
    pt_1 = closest_point2d_on_line2d(A.p, B)
    dist_1 = pt_1.distance_to_point(A.p)
    a_p2 = A.p2
    pt_2 = closest_point2d_on_line2d(a_p2, B)
    dist_2 = pt_2.distance_to_point(a_p2)
    pt_3 = closest_point2d_on_line2d(B.p, A)
    dist_3 = pt_3.distance_to_point(B.p)
    b_p2 = B.p2
    pt_4 = closest_point2d_on_line2d(b_p2, A)
    dist_4 = pt_4.distance_to_point(b_p2)

    # sort the closest points based on their distance
    dists = [dist_1, dist_2, dist_3, dist_4]
    pts = [(A.p, pt_1), (a_p2, pt_2), (pt_3, B.p), (pt_4, b_p2)]
    dists, i = zip(*sorted(zip(dists, range(len(pts)))))
    return dists[0], pts[i[0]]


def closest_point2d_on_arc2d(P, C):
    """Get the closest Point2D on a Arc2D to the input P.

    Args:
        P: A Point2D object.
        C: An Arc2D object along which the closest point will be determined.

    Returns:
        Point2D for the closest point on C to P.
    """
    v = P - C.c
    v = v.normalize() * C.r
    if C.is_circle:
        return Point2D(C.c.x + v.x, C.c.y + v.y)
    else:
        a = Vector2D(1, 0).angle_counterclockwise(v)
        if (not C.is_inverted and C.a1 < a < C.a2) or \
                (C.is_inverted and C.a1 > a > C.a2):
            return Point2D(C.c.x + v.x, C.c.y + v.y)
        else:
            if C.p1.distance_to_point(P) <= C.p2.distance_to_point(P):
                return C.p1
    return C.p2
