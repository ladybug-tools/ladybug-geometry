# coding=utf-8
"""Utility functions for computing intersections between geometry in 2D space."""
from __future__ import division

from .geometry2d.pointvector import Point2D, Vector2D

import math


def intersect_line2d(line_ray_a, line_ray_b):
    """Get the intersection between any Ray2D or LineSegment2D objects as a Point2D.

    This function calculates scaling parameters for ua and ub where:
        A.p + ua * A.v = B.p + ub * B.v
    Which represents the intersection point between line A and line B.

    The derivation of ua is achieved by crossing both sides of the above equation
    with the direction vector of B, and rearranging the formula:
        A.p + ua * A.v = B.p + ub * B.v
        (A.p + ua * A.v) x B.v = (B.p + ub * B.v) x B.v # Cross both sides with B.v
        (A.p x B.v) + (ua * A.v x B.v) = (B.p x B.v) + (ub * B.v x B.v) # B.v x B.v = 0
        ua = (B.p - A.p) x B.v / (A.v x B.v)

    Args:
        line_ray_a: A LineSegment2D or Ray2D object.
        line_ray_b: Another LineSegment2D or Ray2D to intersect.

    Returns:
        Point2D of intersection if it exists. None if no intersection exists.
    """
    # d is the determinant between lines, if 0 lines are collinear
    d = line_ray_b.v.y * line_ray_a.v.x - line_ray_b.v.x * line_ray_a.v.y
    if d == 0:
        return None

    # (dx, dy) = A.p - B.p
    dy = line_ray_a.p.y - line_ray_b.p.y
    dx = line_ray_a.p.x - line_ray_b.p.x

    # Find parameters ua and ub for intersection between two lines

    # Calculate scaling parameter for line_ray_b
    ua = (line_ray_b.v.x * dy - line_ray_b.v.y * dx) / d
    # Checks the bounds of ua to ensure it obeys ray/line behavior
    if not line_ray_a._u_in(ua):
        return None

    # Calculate scaling parameter for line_ray_b
    ub = (line_ray_a.v.x * dy - line_ray_a.v.y * dx) / d
    # Checks the bounds of ub to ensure it obeys ray/line behavior
    if not line_ray_b._u_in(ub):
        return None

    return Point2D(line_ray_a.p.x + ua * line_ray_a.v.x,
                   line_ray_a.p.y + ua * line_ray_a.v.y)


def intersect_line2d_infinite(line_ray_a, line_ray_b):
    """Get the intersection between a Ray2D/LineSegment2D and another extended infinitely.

    Args:
        line_ray_a: A LineSegment2D or Ray2D object.
        line_ray_b: ALineSegment2D or Ray2D that will be extended infinitely
            for intersection.

    Returns:
        Point2D of intersection if it exists. None if no intersection exists.
    """
    d = line_ray_b.v.y * line_ray_a.v.x - line_ray_b.v.x * line_ray_a.v.y
    if d == 0:
        return None
    dy = line_ray_a.p.y - line_ray_b.p.y
    dx = line_ray_a.p.x - line_ray_b.p.x
    ua = (line_ray_b.v.x * dy - line_ray_b.v.y * dx) / d
    if not line_ray_a._u_in(ua):
        return None
    return Point2D(line_ray_a.p.x + ua * line_ray_a.v.x,
                   line_ray_a.p.y + ua * line_ray_a.v.y)


def does_intersection_exist_line2d(line_ray_a, line_ray_b):
    """Boolean denoting whether an intersection exists between Ray2D or LineSegment2D.

    This is slightly faster than actually computing the intersection but should only be
    used in cases where the actual point of intersection is not needed.

    Args:
        line_ray_a: A LineSegment2D or Ray2D object.
        line_ray_b: Another LineSegment2D or Ray2D to intersect.

    Returns:
        True if an intersection exists. False if no intersection exists.
    """
    d = line_ray_b.v.y * line_ray_a.v.x - line_ray_b.v.x * line_ray_a.v.y
    if d == 0:
        return False
    dy = line_ray_a.p.y - line_ray_b.p.y
    dx = line_ray_a.p.x - line_ray_b.p.x
    ua = (line_ray_b.v.x * dy - line_ray_b.v.y * dx) / d
    if not line_ray_a._u_in(ua):
        return False
    ub = (line_ray_a.v.x * dy - line_ray_a.v.y * dx) / d
    if not line_ray_b._u_in(ub):
        return False
    return True


def intersect_line2d_arc2d(line_ray, arc):
    """Get the intersection between any Ray2D/LineSegment2D and an Arc2D.

    Args:
        line_ray: A LineSegment2D or Ray2D object.
        arc: An Arc2D object along which the closest point will be determined.

    Returns:
        A list of 2 Point2D objects if a full intersection exists.
        A list with a single Point2D object if the line is tangent or intersects
        only once. None if no intersection exists.
    """
    a = line_ray.v.magnitude_squared
    b = 2 * (line_ray.v.x * (line_ray.p.x - arc.c.x) +
             line_ray.v.y * (line_ray.p.y - arc.c.y))
    c = arc.c.magnitude_squared + line_ray.p.magnitude_squared - \
        2 * arc.c.dot(line_ray.p) - arc.r ** 2
    det = b ** 2 - 4 * a * c
    if det < 0:
        return None
    sq = math.sqrt(det)
    u1 = (-b + sq) / (2 * a)
    u2 = (-b - sq) / (2 * a)
    pt1 = Point2D(line_ray.p.x + u1 * line_ray.v.x,
                  line_ray.p.y + u1 * line_ray.v.y) if line_ray._u_in(u1) else None
    pt2 = Point2D(line_ray.p.x + u2 * line_ray.v.x,
                  line_ray.p.y + u2 * line_ray.v.y) if line_ray._u_in(u2) else None

    if u1 == u2:  # Tangent
        pt = Point2D(line_ray.p.x + u1 * line_ray.v.x,
                     line_ray.p.y + u1 * line_ray.v.y)
        return pt if arc._pt_in(pt) else None

    pts = [p for p in (pt1, pt2) if p is not None and arc._pt_in(p)]
    return pts if len(pts) != 0 else None


def intersect_line2d_infinite_arc2d(line_ray, arc):
    """Get the intersection between an Arc2D and a Ray2D/LineSegment2D extended infinitely.

    Args:
        line_ray: A LineSegment2D or Ray2D that will be extended infinitely
            for intersection.
        arc: An Arc2D object along which the closest point will be determined.

    Returns:
        A list of 2 Point2D objects if a full intersection exists.
        A list with a single Point2D object if the line is tangent or intersects
        only once. None if no intersection exists.
    """
    a = line_ray.v.magnitude_squared
    b = 2 * (line_ray.v.x * (line_ray.p.x - arc.c.x) +
             line_ray.v.y * (line_ray.p.y - arc.c.y))
    c = arc.c.magnitude_squared + line_ray.p.magnitude_squared - \
        2 * arc.c.dot(line_ray.p) - arc.r ** 2
    det = b ** 2 - 4 * a * c
    if det < 0:
        return None
    sq = math.sqrt(det)
    u1 = (-b + sq) / (2 * a)
    u2 = (-b - sq) / (2 * a)

    if u1 == u2:  # Tangent
        pt = Point2D(line_ray.p.x + u1 * line_ray.v.x, line_ray.p.y + u1 * line_ray.v.y)
        return pt if arc._pt_in(pt) else None

    pt1 = Point2D(line_ray.p.x + u1 * line_ray.v.x, line_ray.p.y + u1 * line_ray.v.y)
    pt2 = Point2D(line_ray.p.x + u2 * line_ray.v.x, line_ray.p.y + u2 * line_ray.v.y)
    pts = [p for p in (pt1, pt2) if arc._pt_in(p)]
    return pts if len(pts) != 0 else None


def closest_point2d_on_line2d(point, line_ray):
    """Get the closest Point2D on a LineSegment2D or Ray2D to the input point.

    Args:
        point: A Point2D object.
        line_ray: A LineSegment2D or Ray2D object along which the closest point
            will be determined.

    Returns:
        Point2D for the closest point on the line_ray to point.
    """
    d = line_ray.v.magnitude_squared
    assert d != 0, '{} length must not equal 0.'.format(line_ray.__class__.__name__)
    u = ((point.x - line_ray.p.x) * line_ray.v.x +
         (point.y - line_ray.p.y) * line_ray.v.y) / d
    if not line_ray._u_in(u):
        u = max(min(u, 1.0), 0.0)
    return Point2D(line_ray.p.x + u * line_ray.v.x, line_ray.p.y + u * line_ray.v.y)


def closest_point2d_between_line2d(line_ray_a, line_ray_b):
    """Get the two closest Point2D between two LineSegment2D objects.

    Note that the line segments should not intersect for the result to be valid.

    Args:
        line_ray_a: A LineSegment2D object.
        line_ray_b: Another LineSegment2D to which closest points will
            be determined.

    Returns:
        A tuple with two elements

        - dists[0]: The distance between the two LineSegment2D objects.
        - pts[0]: A tuple of two Point2D objects representing:

        1) The point on line_ray_a that is closest to line_ray_b
        2) The point on line_ray_b that is closest to line_ray_a
    """
    # one of the 4 endpoints must be a closest point
    pt_1 = closest_point2d_on_line2d(line_ray_a.p, line_ray_b)
    dist_1 = pt_1.distance_to_point(line_ray_a.p)
    a_p2 = line_ray_a.p2
    pt_2 = closest_point2d_on_line2d(a_p2, line_ray_b)
    dist_2 = pt_2.distance_to_point(a_p2)
    pt_3 = closest_point2d_on_line2d(line_ray_b.p, line_ray_a)
    dist_3 = pt_3.distance_to_point(line_ray_b.p)
    b_p2 = line_ray_b.p2
    pt_4 = closest_point2d_on_line2d(b_p2, line_ray_a)
    dist_4 = pt_4.distance_to_point(b_p2)

    # sort the closest points based on their distance
    dists = [dist_1, dist_2, dist_3, dist_4]
    pts = [(line_ray_a.p, pt_1), (a_p2, pt_2), (pt_3, line_ray_b.p), (pt_4, b_p2)]
    dists, i = zip(*sorted(zip(dists, range(len(pts)))))
    return dists[0], pts[i[0]]


def closest_point2d_on_arc2d(point, arc):
    """Get the closest Point2D on a Arc2D to the input point.

    Args:
        point: A Point2D object.
        arc: An Arc2D object along which the closest point will be determined.

    Returns:
        Point2D for the closest point on arc to point.
    """
    v = point - arc.c
    v = v.normalize() * arc.r
    if arc.is_circle:
        return Point2D(arc.c.x + v.x, arc.c.y + v.y)
    else:
        a = Vector2D(1, 0).angle_counterclockwise(v)
        if (not arc.is_inverted and arc.a1 < a < arc.a2) or \
                (arc.is_inverted and arc.a1 > a > arc.a2):
            return Point2D(arc.c.x + v.x, arc.c.y + v.y)
        else:
            if arc.p1.distance_to_point(point) <= arc.p2.distance_to_point(point):
                return arc.p1
    return arc.p2
