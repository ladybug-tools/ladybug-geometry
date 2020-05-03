# coding=utf-8
"""Utility functions for computing intersections between geometry in 3D space."""
from __future__ import division

from .geometry3d.pointvector import Point3D

import math


def intersect_line3d_plane(line_ray, plane):
    """Get the intersection between a Ray3D/LineSegment3D and a Plane.

    Args:
        line_ray: A LineSegment3D or Ray3D object.
        plane: A Plane object to intersect.

    Returns:
        Point2D of intersection if it exists. None if no intersection exists.
    """
    d = plane.n.dot(line_ray.v)
    if not d:  # parallel
        return None
    u = (plane.k - plane.n.dot(line_ray.p)) / d
    if not line_ray._u_in(u):  # line or ray does not have its domain in the plane
        return None
    return Point3D(line_ray.p.x + u * line_ray.v.x,
                   line_ray.p.y + u * line_ray.v.y,
                   line_ray.p.z + u * line_ray.v.z)


def intersect_plane_plane(plane_a, plane_b):
    """Get the intersection between two Plane objects.

    Args:
        plane_a: A Plane object.
        plane_b: Another Plane object to intersect.

    Returns:
        Two objects that define the intersection between two planes

        1) A Point3D that lies along the intersection of the two planes.
        2) A Vector3D that describes the direction of the intersection.

        Will be None if no intersection exists (planes are parallel).
    """
    n1_m = plane_a.n.magnitude_squared
    n2_m = plane_b.n.magnitude_squared
    n1d2 = plane_a.n.dot(plane_b.n)
    det = n1_m * n2_m - n1d2 ** 2
    if det == 0:  # parallel
        return None
    c1 = (plane_a.k * n2_m - plane_b.k * n1d2) / det
    c2 = (plane_b.k * n1_m - plane_a.k * n1d2) / det
    return Point3D(c1 * plane_a.n.x + c2 * plane_b.n.x,
                   c1 * plane_a.n.y + c2 * plane_b.n.y,
                   c1 * plane_a.n.z + c2 * plane_b.n.z), plane_a.n.cross(plane_b.n)


def closest_point3d_on_line3d(point, line_ray):
    """Get the closest Point3D on a LineSegment3D or Ray3D to the input point.

    Args:
        point: A Point3D object.
        line_ray: A LineSegment3D or Ray3D object along which the closest point
            will be determined.

    Returns:
        Point3D for the closest point on line_ray to point.
    """
    d = line_ray.v.magnitude_squared
    assert d != 0, 'Length of LineSegment3D must be greater than 0.'
    u = ((point.x - line_ray.p.x) * line_ray.v.x +
         (point.y - line_ray.p.y) * line_ray.v.y +
         (point.z - line_ray.p.z) * line_ray.v.z) / d
    if not line_ray._u_in(u):
        u = max(min(u, 1.0), 0.0)
    return Point3D(line_ray.p.x + u * line_ray.v.x,
                   line_ray.p.y + u * line_ray.v.y,
                   line_ray.p.z + u * line_ray.v.z)


def closest_point3d_on_line3d_infinite(point, line_ray):
    """Get the closest Point3D on an infinite extension of a LineSegment3D or Ray3D.

    Args:
        point: A Point3D object.
        line_ray: A LineSegment3D or Ray3D object along which the closest point
            will be determined.

    Returns:
        Point3D for the closest point on the line_ray to the point.
    """
    d = line_ray.v.magnitude_squared
    assert d != 0, 'Length of LineSegment3D must be greater than 0.'
    u = ((point.x - line_ray.p.x) * line_ray.v.x +
         (point.y - line_ray.p.y) * line_ray.v.y +
         (point.z - line_ray.p.z) * line_ray.v.z) / d
    return Point3D(line_ray.p.x + u * line_ray.v.x,
                   line_ray.p.y + u * line_ray.v.y,
                   line_ray.p.z + u * line_ray.v.z)


def closest_point3d_on_plane(point, plane):
    """Get the closest Point3D on a Plane to the input point.

    Args:
        point: A Point3D object.
        plane: A Plane object in which the closest point will be determined.

    Returns:
        Point3D for the closest point on the plane to point.
    """
    n = plane.n
    d = point.dot(plane.n) - plane.k
    return Point3D(point.x - n.x * d, point.y - n.y * d, point.z - n.z * d)


def closest_point3d_between_line3d_plane(line_ray, plane):
    """Get the two closest Point3D between a LineSegment3D/Ray3D and a Plane.

    Args:
        line_ray: A LineSegment3D or Ray3D object along which the closest point
            will be determined.
        plane: A Plane object on which a closest point will be determined.

    Returns:
        Two Point3D objects representing

        1) The point on the line_ray that is closest to the plane.
        2) The point on the plane that is closest to the line_ray.

        Will be None if there is an intersection between line_ray and the plane
    """
    d = plane.n.dot(line_ray.v)
    if not d:  # parallel, choose an endpoint
        return line_ray.p, closest_point3d_on_plane(line_ray.p, plane)
    u = (plane.k - plane.n.dot(line_ray.p)) / d
    if not line_ray._u_in(u):  # intersects out of range of L, choose nearest endpoint
        u = max(min(u, 1.0), 0.0)
        close_pt = Point3D(line_ray.p.x + u * line_ray.v.x,
                           line_ray.p.y + u * line_ray.v.y,
                           line_ray.p.z + u * line_ray.v.z)
        return close_pt, closest_point3d_on_plane(close_pt, plane)
    return None  # intersection


def intersect_line3d_sphere(line_ray, sphere):
    """Get the intersection between this Sphere object and a Ray2D/LineSegment2D.

    Args:
        line_ray: A LineSegment3D or Ray3D for intersection.
        sphere: A Sphere to intersect.

    Returns:
        Two Point3D objects if a full intersection exists.
        A Point3D if a point of tangency exists.
        Will be None if no intersection exists.

    """
    L = line_ray
    S = sphere
    a = L.v.magnitude_squared
    b = 2 * (L.v.x * (L.p.x - S.center.x) +
             L.v.y * (L.p.y - S.center.y) +
             L.v.z * (L.p.z - S.center.z))
    c = S.center.magnitude_squared + \
        L.p.magnitude_squared - \
        2 * S.center.dot(L.p) - \
        S.radius ** 2
    det = b ** 2 - 4 * a * c
    if det < 0:
        return None
    sq = math.sqrt(det)
    u1 = (-b + sq) / (2 * a)
    u2 = (-b - sq) / (2 * a)
    if not L._u_in(u1):
        u1 = max(min(u1, 1.0), 0.0)
    if not L._u_in(u2):
        u2 = max(min(u2, 1.0), 0.0)
    p1 = Point3D(L.p.x + u1 * L.v.x, L.p.y + u1 * L.v.y, L.p.z + u1 * L.v.z)
    if u1 == u2:
        return p1
    else:
        p2 = Point3D(L.p.x + u2 * L.v.x, L.p.y + u2 * L.v.y, L.p.z + u2 * L.v.z)
        return p1, p2


def intersect_plane_sphere(plane, sphere):
    """Get the intersection of a plane with this Sphere object

    Args:
        plane: A Plane object.
        sphere: A Sphere to intersect.

    Returns:
        If a full intersection exists

        1) A Point3D that represents the center of the intersection circle.
        2) A Vector3D that represents the normal of the intersection circle.
        3) A number that represents the radius of the intersection circle.

        A Point3D Object if a point of tangency exists.
        None if no intersection exists.
    """
    r = sphere.radius
    pt_c = sphere.center
    pt_o = plane.o
    v_n = plane.n.normalize()

    # Resulting circle radius. Radius² = r² - [(c-p).n]²
    d = (pt_o - pt_c).dot(v_n)
    if abs(r) < abs(d):  # No intersection if (r ** 2 - d ** 2) negative
        return None
    cut_r = math.sqrt(r ** 2 - d ** 2)

    # Intersection circle center point. Center_point = p - [(c-p).n]n
    cut_center = pt_c + (d * v_n)

    return (cut_center, v_n, cut_r) if cut_r != 0 else cut_center
