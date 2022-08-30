# coding=utf-8
"""Cylinder"""
from __future__ import division

from .pointvector import Point3D, Vector3D
from .plane import Plane
from .arc import Arc3D

import math


class Cylinder(object):
    """Cylinder object.

    Args:
        center: A Point3D at the center of the bottom base of the cylinder.
        axis: A Vector3D representing the direction and height of the cylinder.
            The vector extends from the bottom base center to the top base center.
        radius: A number representing the radius of the cylinder.

    Properties:
        * center
        * axis
        * radius
        * center_end
        * diameter
        * height
        * area
        * volume
        * min
        * max
        * base_bottom
        * base_top
    """
    __slots__ = ('_center', '_axis', '_radius',
                 '_base_bottom', '_base_top', '_min', '_max')

    def __init__(self, center, axis, radius):
        """Initialize Cylinder."""
        assert isinstance(center, Point3D), \
            "Expected Point3D. Got {}.".format(type(center))
        assert isinstance(axis, Vector3D), \
            "Expected Vector3D. Got {}.".format(type(axis))
        assert radius > 0, \
            'Cylinder radius must be greater than 0. Got {}.'.format(radius)
        self._center = center
        self._axis = axis
        self._radius = radius
        self._base_bottom = None
        self._base_top = None
        self._min = None
        self._max = None

    @classmethod
    def from_start_end(cls, p1, p2, radius):
        """Initialize a new cylinder from start and end points.

        Args:
            p1: The start point of the cylinder, represents the center of the
                bottom base of the cylinder.
            p2: The end point of the cylinder, represents the center of the top
                base of the cylinder
            radius: A number representing the radius of the cylinder.
        """
        axis = p2 - p1
        return cls(p1, axis, radius)

    @classmethod
    def from_dict(cls, data):
        """Create a Cylinder from a dictionary.

        Args:
            data: A python dictionary in the following format

        .. code-block:: python

            {
                "type": "Cylinder"
                "center": (10, 0, 0),
                "axis": (0, 0, 1),
                "radius": 1.0
            }
        """
        return cls(Point3D.from_array(data['center']),
                   Vector3D.from_array(data['axis']),
                   data['radius'])

    @property
    def center(self):
        """Center of Cylinder."""
        return self._center

    @property
    def axis(self):
        """Axis of Cylinder."""
        return self._axis

    @property
    def radius(self):
        """Radius of Cylinder"""
        return self._radius

    @property
    def center_end(self):
        """Center of the opposite end of Cylinder."""
        return self.center + self.axis

    @property
    def diameter(self):
        """Diameter of Cylinder"""
        return self.radius * 2

    @property
    def height(self):
        """Height of Cylinder"""
        return self.axis.magnitude

    @property
    def area(self):
        """Surface area of a Cylinder"""
        return 2 * math.pi * self.radius * self.height + 2 * math.pi * self.radius ** 2

    @property
    def volume(self):
        """Volume of a Cylinder"""
        return math.pi * self.radius ** 2 * self.height

    @property
    def base_bottom(self):
        """Get an Arc3D representing the bottom circular base of the cylinder."""
        if self._base_bottom is None:
            self._base_bottom = Arc3D(Plane(self.axis, self.center), self.radius)
        return self._base_bottom

    @property
    def base_top(self):
        """Get an Arc3D representing the top  circular base of the cylinder."""
        if self._base_top is None:
            plane = Plane(self.axis, self.center + self.axis)
            self._base_top = Arc3D(plane, self.radius)
        return self._base_top

    @property
    def min(self):
        """A Point3D for the minimum bounding box vertex around this geometry."""
        if self._min is None:
            self._calculate_min_max()
        return self._min

    @property
    def max(self):
        """A Point3D for the maximum bounding box vertex around this geometry."""
        if self._max is None:
            self._calculate_min_max()
        return self._max

    def move(self, moving_vec):
        """Get a Cylinder that has been moved along a vector.

        Args:
            moving_vec: A Vector3D with the direction and distance to move the Cylinder.
        """
        return Cylinder(self.center.move(moving_vec), self.axis, self.radius)

    def rotate(self, axis, angle, origin):
        """Rotate this Cylinder by a certain angle around an axis and origin.

        Right hand rule applies:
        If axis has a positive orientation, rotation will be clockwise.
        If axis has a negative orientation, rotation will be counterclockwise.

        Args:
            axis: A Vector3D axis representing the axis of rotation.
            angle: An angle for rotation in radians.
            origin: A Point3D for the origin around which the cylinder will be rotated.
        """
        return Cylinder(self.center.rotate(axis, angle, origin),
                        self.axis.rotate(axis, angle),
                        self.radius)

    def rotate_xy(self, angle, origin):
        """Get a Cylinder that is rotated counterclockwise in the world XY plane by an angle.

        Args:
            angle: An angle for rotation in radians.
            origin: A Point3D for the origin around which the cylinder will be rotated.
        """
        return Cylinder(self.center.rotate_xy(angle, origin),
                        self.axis.rotate_xy(angle),
                        self.radius)

    def reflect(self, normal, origin):
        """Get a Cylinder reflected across a plane with the input normal vector and origin.

        Args:
            normal: A Vector3D representing the normal vector for the plane across
                which the arc will be reflected. THIS VECTOR MUST BE NORMALIZED.
            origin: A Point3D representing the origin from which to reflect.
        """
        return Cylinder(self.center.reflect(normal, origin),
                        self.axis.reflect(normal),
                        self.radius)

    def scale(self, factor, origin=None):
        """Scale a Cylinder by a factor from an origin point.

        Args:
            factor: A number representing how much the Cylinder should be scaled.
            origin: A Point3D representing the origin from which to scale.
                If None, it will be scaled from the World origin (0, 0, 0).
        """
        return Cylinder(self.center.scale(factor, origin),
                        self.axis * factor,
                        self.radius * factor)

    def duplicate(self):
        """Get a copy of this object."""
        return self.__copy__()

    def to_dict(self):
        """Get Cylinder as a dictionary."""
        return {
            'type': 'Cylinder',
            'center': self.center.to_array(),
            'axis': self.axis.to_array(),
            'radius': self.radius
        }

    def _calculate_min_max(self):
        """Calculate maximum and minimum Point3D for this object."""
        base1, base2 = self.base_bottom, self.base_top
        b1mn, b1mx, b2mn, b2mx = base1.min, base1.max, base2.min, base2.max
        self._min = Point3D(
            min(b1mn.x, b2mn.x), min(b1mn.y, b2mn.y), min(b1mn.z, b2mn.z))
        self._max = Point3D(
            max(b1mx.x, b2mx.x), max(b1mx.y, b2mx.y), max(b1mx.z, b2mx.z))

    def __copy__(self):
        return Cylinder(self.center, self.axis, self.radius)

    def __key(self):
        """A tuple based on the object properties, useful for hashing."""
        return (self._center, self._axis, self._radius)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(other, Cylinder) and self.__key() == other.__key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __repr__(self):
        return 'Cylinder (center {}) (axis {}) (radius {})'.\
            format(self.center, self.axis, self.radius)
