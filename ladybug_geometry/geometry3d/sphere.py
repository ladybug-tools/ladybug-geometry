# coding=utf-8
"""Sphere"""
from __future__ import division

from .pointvector import Point3D
from .plane import Plane
from .arc import Arc3D
from .line import LineSegment3D
from ..intersection3d import intersect_line3d_sphere, intersect_plane_sphere


import math


class Sphere(object):
    """Sphere object.

    Args:
        center: A Point3D representing the center of the arc.
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
        """Initialize Sphere."""
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
                "radius": 5
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
        """A Point3D for the minimum bounding box vertex around this geometry."""
        return Point3D(self.center.x - self.radius, self.center.y - self.radius,
                       self.center.z - self.radius)

    @property
    def max(self):
        """A Point3D for the maximum bounding box vertex around this geometry."""
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
        """Surface area of sphere"""
        return 4 * math.pi * self.radius ** 2

    @property
    def volume(self):
        """Volume of sphere"""
        return 4 / 3 * math.pi * self.radius ** 3

    def move(self, moving_vec):
        """Get a sphere that has been moved along a vector.

        Args:
            moving_vec: A Vector3D with the direction and distance to move the sphere.
        """
        return Sphere(self.center.move(moving_vec), self.radius)

    def rotate(self, axis, angle, origin):
        """Rotate this sphere by a certain angle around an axis and origin.

        Right hand rule applies:
        If axis has a positive orientation, rotation will be clockwise.
        If axis has a negative orientation, rotation will be counterclockwise.

        Args:
            axis: A Vector3D axis representing the axis of rotation.
            angle: An angle for rotation in radians.
            origin: A Point3D for the origin around which the sphere will be rotated.
        """
        return Sphere(self.center.rotate(axis, angle, origin), self.radius)

    def rotate_xy(self, angle, origin):
        """Get a sphere that is rotated counterclockwise in the world XY plane by an angle.

        Args:
            angle: An angle for rotation in radians.
            origin: A Point3D for the origin around which the sphere will be rotated.
        """
        return Sphere(self.center.rotate_xy(angle, origin), self.radius)

    def reflect(self, normal, origin):
        """Get a sphere reflected across a plane with the input normal vector and origin.

        Args:
            normal: A Vector3D representing the normal vector for the plane across
                which the arc will be reflected. THIS VECTOR MUST BE NORMALIZED.
            origin: A Point3D representing the origin from which to reflect.
        """
        return Sphere(self.center.reflect(normal, origin), self.radius)

    def scale(self, factor, origin=None):
        """Scale a sphere by a factor from an origin point.

        Args:
            factor: A number representing how much the sphere should be scaled.
            origin: A Point3D representing the origin from which to scale.
                If None, it will be scaled from the World origin (0, 0, 0).
        """
        return Sphere(self.center.scale(factor, origin), self.radius * factor)

    def intersect_plane(self, plane):
        """Get the intersection of a plane with this Sphere object

        Args:
            plane: A Plane object.

        Returns:
            Arc3D representing a full circle if it exists.
            None if no full intersection exists.
        """
        ip = intersect_plane_sphere(plane, self)  # ip = [center pt, vector, radius]
        return None if ip is None or isinstance(ip, Point3D) else \
            Arc3D(Plane(ip[1], ip[0]), ip[2])

    def intersect_line_ray(self, line_ray):
        """Get the intersection between this Sphere object and a Ray2D/LineSegment2D.

        Args:
            line_ray: A LineSegment3D or Ray3D that will be extended infinitely
                for intersection.

        Returns:
            A LineSegment3D object if a full intersection exists.
            A Point if a tangent intersection exists.
            None if no full intersection exists.
        """
        il = intersect_line3d_sphere(line_ray, self)
        return None if il is None else \
            il if isinstance(il, Point3D) else \
            LineSegment3D.from_end_points(il[0], il[1])

    def duplicate(self):
        """Get a copy of this object."""
        return self.__copy__()

    def to_dict(self):
        """Get Sphere as a dictionary."""
        return {'type': 'Sphere',
                'center': self.center.to_array(),
                'radius': self.radius}

    def __copy__(self):
        return Sphere(self._center, self._radius)

    def __key(self):
        """A tuple based on the object properties, useful for hashing."""
        return (self._center, self._radius)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(other, Sphere) and self.__key() == other.__key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __repr__(self):
        return 'Sphere (center {}) (radius {}))'.format(self.center, self.radius)
