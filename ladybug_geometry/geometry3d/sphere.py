# coding=utf-8
"""Sphere"""
from __future__ import division

from .pointvector import Point3D
from .plane import Plane
from .arc import Arc3D

import math


class Sphere(object):
    """Sphere object.

    Args:
        center
        radius

    Properties:
        * center
        * radius
    """
    __slots__ = ('_center', '_radius')

    def __init__(self, center, radius):
        """Initilize Sphere.
        """
        assert isinstance(center, Point3D), \
            "Expected Point3D. Got {}.".format(type(center))
        assert radius > 0, 'Sphere radius must be greater than 0. Got {}.'.format(radius)
        self._center = center
        self._radius = radius

    @classmethod
    def from_dict(cls, data):
        """Create a Sphere from a dictionary.

        Args:
            data: A python dictionary in the following format

        .. code-block:: python

            {
            "type": "Sphere"
            "center": (10, 0, 0),
            "radius": 5,
            }
        """
        return cls(Point3D(data['center'][0], data['center'][1], data['center'][2],
                           data['radius']))

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
        return Point3D(self.center[0] - self.radius, self.center[1] - self.radius,
                       self.center[2] - self.radius)

    @property
    def max(self):
        """A Point3D for the maximum bounding rectangle vertex around this geometry."""
        return Point3D(self.center[0] + self.radius, self.center[1] + self.radius,
                       self.center[2] + self.radius)

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
        return Sphere(self.center.move(moving_vec), self.radius)

    def scale(self, factor, origin=None):
        """Scale a arc by a factor from an origin point.

        Args:
            factor: A number representing how much the sphere should be scaled.
            origin: A Point2D representing the origin from which to scale.
                If None, it will be scaled from the World origin (0, 0, 0).
        """
        return Sphere(self.center.scale(factor, origin), self.radius * factor)

    def intersect_plane(self, plane):
        """Get the intersection of a plane with this Sphere object

        Args:
            plane: A Plane object

        Returns:
            Arc3D representing a full circle if it exists.
            None if no intersection exists.
        """
        r = self.radius
        pt_c = self.center
        pt_o = plane.o
        v_n = plane.n.normalize()

        # Resulting circle radius. Radius² = r² - [(c-p).n]²
        v_d = ((pt_c + pt_o) * v_n)
        d = v_d.magnitude  # d = plane.distance_to_point(self._center)
        if r < d:  # No intersection if negative
            return None
        cut_r = math.sqrt(r ** 2 - d ** 2)
        if cut_r == 0:  # Tangent intersection ignored - results in a point
            return None

        # Resulting circle center point. Center_point = p - [(c-p).n]n
        cut_center = pt_c + v_d * v_n
        cut_plane = Plane(v_n, cut_center)
        return Arc3D(cut_plane, cut_r)

    def intersect_line_ray(self, line_ray):
        """Get the intersection between this Sphere object and a Ray2D/LineSegment2D.

        Args:
            line_ray: A LineSegment3D or Ray3D that will be extended infinitely
                for intersection.

        Returns:
            A list of 2 Point3D objects if a full intersection exists.
            A list with a single Point3D object if the line is tangent or intersects
            only once. None if no intersection exists.
            return
        """
        sc = self.center
        r = self.radius
        p1 = line_ray.p1
        p2 = line_ray.p2
        dp = p2 - p1
        a = dp.x * dp.x + dp.y * dp.y + dp.z * dp.z
        b = 2 * (dp.x * (p1.x - sc.x) + dp.y * (p1.y - sc.y) + dp.z * (p1.z - sc.z))
        c = sc.x * sc.x + sc.y * sc.y + sc.z * sc.z
        c += p1.x * p1.x + p1.y * p1.y + p1.z * p1.z
        c -= 2 * (sc.x * p1.x + sc.y * p1.y + sc.z * p1.z)
        c -= r * r
        bb4ac = b * b - 4 * a * c

        if abs(a) < math.ldexp(1.0, -53) or bb4ac < 0:
            return None  # No intersection

        mu1 = (-b + math.sqrt(bb4ac)) / (2 * a)
        mu2 = (-b - math.sqrt(bb4ac)) / (2 * a)
        pt1 = p1 + mu1 * (p2 - p1)
        pt2 = p1 + mu2 * (p2 - p1)

        return [pt1] if mu1 == mu2 else [pt1, pt2]

    def duplicate(self):
        """Get a copy of this object."""
        return self.__copy__()

    def to_dict(self):
        """Get Sphere as a dictionary."""
        return {'type': 'Sphere', 'center': self.center.to_dict(),
                'radius': self.radius}

    def __copy__(self):
        return self.__class__(self.center, self.radius)

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __repr__(self):
        return 'Sphere (center {}) (radius {}))'.format(self.center, self.radius)
