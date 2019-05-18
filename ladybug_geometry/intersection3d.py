# coding=utf-8
"""Utility functions for computing intersections between geometry in 3D space."""
from __future__ import division

from .geometry3d.pointvector import Point3D


def intersect_line3d_plane(L, PL):
    """Get the intersection between a Ray3D/LineSegment3D and a Plane.

    Args:
        L: A LineSegment3D or Ray3D object.
        PL: A Plane object to intersect.

    Returns:
        Point2D of intersection if it exists. None if no intersection exists.
    """
    d = PL.n.dot(L.v)
    if not d:  # parallel
        return None
    u = (PL.k - PL.n.dot(L.p)) / d
    if not L._u_in(u):  # line or ray does not have its domain in the plane
        return None
    return Point3D(L.p.x + u * L.v.x, L.p.y + u * L.v.y, L.p.z + u * L.v.z)


def intersect_plane_plane(A, B):
    """Get the intersection between two Plane objects.

    Args:
        A: A Plane object.
        B: Another Plane object to intersect.

    Returns:
        Two objects that define the intersection between two planes:
            1) A Point3D that lies along the intersection of the two planes.
            2) A Vector3D that describes the direction of the intersection.
        Will be None if no intersection exists (planes are parallel).
    """
    n1_m = A.n.magnitude_squared
    n2_m = B.n.magnitude_squared
    n1d2 = A.n.dot(B.n)
    det = n1_m * n2_m - n1d2 ** 2
    if det == 0:  # parallel
        return None
    c1 = (A.k * n2_m - B.k * n1d2) / det
    c2 = (B.k * n1_m - A.k * n1d2) / det
    return Point3D(c1 * A.n.x + c2 * B.n.x,
                   c1 * A.n.y + c2 * B.n.y,
                   c1 * A.n.z + c2 * B.n.z), A.n.cross(B.n)


def closest_point3d_on_line3d(P, L):
    """Get the closest Point3D on a LineSegment3D or Ray3D to the input P.

    Args:
        P: A Point3D object.
        L: A LineSegment3D or Ray3D object along wich the closest point
            will be determined.

    Returns:
        Point3D for the closest point on L to P.
    """
    d = L.v.magnitude_squared
    assert d != 0, 'Length of LineSegment3D must be greater than 0.'
    u = ((P.x - L.p.x) * L.v.x +
         (P.y - L.p.y) * L.v.y +
         (P.z - L.p.z) * L.v.z) / d
    if not L._u_in(u):
        u = max(min(u, 1.0), 0.0)
    return Point3D(L.p.x + u * L.v.x, L.p.y + u * L.v.y, L.p.z + u * L.v.z)


def closest_point3d_on_line3d_infinite(P, L):
    """Get the closest Point3D on an infinite extension of a LineSegment3D or Ray3D.

    Args:
        P: A Point3D object.
        L: A LineSegment3D or Ray3D object along wich the closest point
            will be determined.

    Returns:
        Point3D for the closest point on L to P.
    """
    d = L.v.magnitude_squared
    assert d != 0, 'Length of LineSegment3D must be greater than 0.'
    u = ((P.x - L.p.x) * L.v.x +
         (P.y - L.p.y) * L.v.y +
         (P.z - L.p.z) * L.v.z) / d
    return Point3D(L.p.x + u * L.v.x, L.p.y + u * L.v.y, L.p.z + u * L.v.z)


def closest_point3d_on_plane(P, PL):
    """Get the closest Point3D on a Plane to the input P.

    Args:
        P: A Point3D object.
        PL: A Plane object in which the closest point will be determined.

    Returns:
        Point3D for the closest point on the plane to P.
    """
    n = PL.n
    d = P.dot(PL.n) - PL.k
    return Point3D(P.x - n.x * d, P.y - n.y * d, P.z - n.z * d)


def closest_point3d_between_line3d_plane(L, PL):
    """Get the two closest Point3D between a LineSegment3D/Ray3D and a Plane.

    Args:
        Args:
            L: A LineSegment3D or Ray3D object along wich the closest point
                will be determined.
            PL: A Plane object on which a closest point will be determined.

    Returns:
        Two Point3D objects representing:
            1) The point on L that is closest to PL
            2) The point on PL that is closest to L
        Will be None if there is an intersection between L and PL
    """
    d = PL.n.dot(L.v)
    if not d:  # parallel, choose an endpoint
        return L.p, closest_point3d_on_plane(L.p, PL)
    u = (PL.k - PL.n.dot(L.p)) / d
    if not L._u_in(u):  # intersects out of range of L, choose the nearest endpoint
        u = max(min(u, 1.0), 0.0)
        close_pt = Point3D(L.p.x + u * L.v.x, L.p.y + u * L.v.y, L.p.z + u * L.v.z)
        return close_pt, closest_point3d_on_plane(close_pt, PL)
    return None  # intersection
