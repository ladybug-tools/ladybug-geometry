# coding=utf-8
"""Sphere3D"""
from __future__ import division

from .pointvector import Point3D
# from ..intersection3d import intersect_line3d_sphere3d, intersect_plane_sphere3d

import math

from .plane import Plane
from .arc import Arc3D
from .line import LineSegment3D


class Sphere3D(object):
    """Sphere3D object.

    Args:
        center: A Point2D representing the center of the arc.
        radius: A number representing the radius of the arc.

    Properties:
        * center
        * radius
        * min
        * max
        * diameter
        * circumference
        * area
        * volume
    """
    __slots__ = ('_center', '_radius')

    def __init__(self, center, radius):
        """Initilize Sphere3D.
        """
        assert isinstance(center, Point3D), \
            "Expected Point3D. Got {}.".format(type(center))
        assert radius > 0, 'Sphere radius must be greater than 0. Got {}.'.format(radius)
        self._center = center
        self._radius = radius

    @classmethod
    def from_dict(cls, data):
        """Create a Sphere3D from a dictionary.

        Args:
            data: A python dictionary in the following format

        .. code-block:: python

            {
            "type": "Sphere3D"
            "center": (10, 0, 0),
            "radius": 5,
            }
        """
        return cls(Point3D.from_array(data['center']), data['radius'])

    @property
    def center(self):
        """Center of sphere."""
        return self._center

    @property
    def radius(self):
        """Radius of sphere."""
        return self._radius

    @property
    def min(self):
        """A Point3D for the minimum bounding rectangle vertex around this geometry."""
        return Point3D(self.center.x - self.radius, self.center.y - self.radius,
                       self.center.z - self.radius)

    @property
    def max(self):
        """A Point3D for the maximum bounding rectangle vertex around this geometry."""
        return Point3D(self.center.x + self.radius, self.center.y + self.radius,
                       self.center.z + self.radius)

    @property
    def diameter(self):
        """Diameter of sphere"""
        return self.radius * 2

    @property
    def circumference(self):
        """Circumference of sphere"""
        return 2 * math.pi * self.radius

    @property
    def area(self):
        """Area of sphere"""
        return 4 * math.pi * self.radius ** 2

    @property
    def volume(self):
        """Volume of sphere"""
        return 4 / 3 * math.pi * self.radius ** 3

    def move(self, moving_vec):
        """Get a sphere that has been moved along a vector.

        Args:
            moving_vec: A Vector3D with the direction and distance to move the shere.
        """
        return Sphere3D(self.center.move(moving_vec), self.radius)

    def scale(self, factor, origin=None):
        """Scale a arc by a factor from an origin point.

        Args:
            factor: A number representing how much the sphere should be scaled.
            origin: A Point3D representing the origin from which to scale.
                If None, it will be scaled from the World origin (0, 0, 0).
        """
        return Sphere3D(self.center.scale(factor, origin), self.radius * factor)

    def intersect_plane(self, plane):
        """Get the intersection of a plane with this Sphere3D object

        Args:
            plane: A Plane object

        Returns:
            Arc3D representing a full circle if it exists.
            None if no full intersection exists.
        """
        return intersect_plane_sphere3d(plane, self)

    def intersect_line_ray(self, line_ray):
        """Get the intersection between this Sphere3D object and a Ray2D/LineSegment2D.

        Args:
            line_ray: A LineSegment3D or Ray3D that will be extended infinitely
                for intersection.

        Returns:
            A list of 2 Point3D objects if a full intersection exists.
            None if no full intersection exists.
        """
        return intersect_line3d_sphere3d(line_ray, self)

    def duplicate(self):
        """Get a copy of this object."""
        return self.__copy__()

    def to_dict(self):
        """Get Sphere as a dictionary."""
        return {'type': 'Sphere3D', 'center': self.center.to_array(),
                'radius': self.radius}

    def __copy__(self):
        return self.Sphere3D(self.center, self.radius)

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __repr__(self):
        return 'Sphere3D (center {}) (radius {}))'.format(self.center, self.radius)


def intersect_line3d_sphere3d(line_ray, sphere):
    """Get the intersection between this Sphere3D object and a Ray2D/LineSegment2D.

    Args:
        line_ray: A LineSegment3D or Ray3D that will be extended infinitely
            for intersection.
        sphere: A Sphere3D to intersect.

    Returns:
        A list of 2 Point3D objects if a full intersection exists.
        None if no full intersection exists.
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
    return LineSegment3D.from_end_points(Point3D(L.p.x + u1 * L.v.x,
                                                 L.p.y + u1 * L.v.y,
                                                 L.p.z + u1 * L.v.z),
                                         Point3D(L.p.x + u2 * L.v.x,
                                                 L.p.y + u2 * L.v.y,
                                                 L.p.z + u2 * L.v.z))


def intersect_plane_sphere3d(plane, sphere):
    """Get the intersection of a plane with this Sphere3D object

    Args:
        plane: A Plane object
        sphere: A Sphere3D to intersect.

    Returns:
        Arc3D representing a full circle if it exists.
        None if no full intersection exists.
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
    if cut_r == 0:  # Tangent intersection ignored - results in a point
        return None

    # Intersection circle center point. Center_point = p - [(c-p).n]n
    cut_center = pt_c + (d * v_n)
    cut_plane = Plane(v_n, cut_center)
    return Arc3D(cut_plane, cut_r)
