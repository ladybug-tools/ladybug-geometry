# coding=utf-8
"""Cylinder3D"""
from __future__ import division

from .pointvector import Point3D, Vector3D

import math


class Cylinder3D(object):
    """Cylinder3D object.

    Args:
        center: A Point3D at the center of the bottom base of the cylinder.
        axis: A Vector3D representing the direction and height of the cylinder.
            The vector extends from the bottom base center to the top base center.
        radius: A number representing the radius of the cylinder.

    Properties:
        * center
        * axis
        * radius
        * diameter
        * height
        * area
        * volume
    """
    __slots__ = ('_center', '_axis', '_radius')

    def __init__(self, center, axis, radius):
        """Initilize Cylinder3D.
        """
        assert isinstance(center, Point3D), \
            "Expected Point3D. Got {}.".format(type(center))
        assert isinstance(axis, Vector3D), \
            "Expected Vector3D. Got {}.".format(type(axis))
        assert radius > 0, \
            'Cylinder radius must be greater than 0. Got {}.'.format(radius)
        self._center = center
        self._axis = axis
        self._radius = radius

    @classmethod
    def from_dict(cls, data):
        """Create a Cylinder3D from a dictionary.

        Args:
            data: A python dictionary in the following format

        .. code-block:: python

            {
            "type": "Cylinder3D"
            "center": (10, 0, 0),
            "axis": (0, 0, 1),
            "radius": 1.0,
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
    def diameter(self):
        """Diameter of Cylinder"""
        return self._radius * 2

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

    def move(self, moving_vec):
        """Get a Cylinder that has been moved along a vector.

        Args:
            moving_vec: A Vector3D with the direction and distance to move the Cylinder.
        """
        return Cylinder3D(self.center.move(moving_vec), self.axis, self.radius)

    def rotate(self, axis, angle, origin):
        """Rotate this Cylinder by a certain angle around an axis and origin.

        Right hand rule applies:
        If axis has a positive orientation, rotation will be clockwise.
        If axis has a negative orientation, rotation will be counterclockwise.

        Args:
            axis: A Vector3D axis representing the axis of rotation.
            angle: An angle for rotation in radians.
            origin: A Point3D for the origin around which the sphere will be rotated.
        """
        return Cylinder3D(self.center.rotate(axis, angle, origin),
                          self.axis.rotate(axis, angle),
                          self.radius)

    def rotate_xy(self, angle, origin):
        """Get a Cylinder that is rotated counterclockwise in the world XY plane by an angle.

        Args:
            angle: An angle for rotation in radians.
            origin: A Point3D for the origin around which the sphere will be rotated.
        """
        return Cylinder3D(self.center.rotate_xy(angle, origin),
                          self.axis.rotate_xy(angle),
                          self.radius)

    def reflect(self, normal, origin):
        """Get a Cylinder reflected across a plane with the input normal vector and origin.

        Args:
            normal: A Vector3D representing the normal vector for the plane across
                which the arc will be reflected. THIS VECTOR MUST BE NORMALIZED.
            origin: A Point3D representing the origin from which to reflect.
        """
        return Cylinder3D(self.center.reflect(normal, origin),
                          self.axis.reflect(normal),
                          self.radius)

    def scale(self, factor, origin=None):
        """Scale a Cylinder by a factor from an origin point.

        Args:
            factor: A number representing how much the Cylinder should be scaled.
            origin: A Point3D representing the origin from which to scale.
                If None, it will be scaled from the World origin (0, 0, 0).
        """
        return Cylinder3D(self.center.scale(factor, origin),
                          self.axis * factor,
                          self.radius)

    def duplicate(self):
        """Get a copy of this object."""
        return self.__copy__()

    def to_dict(self):
        """Get Cylinder as a dictionary."""
        return {'type': 'Cylinder3D', 'center': self.center.to_array(),
                'axis': self.axis.to_array(), 'radius': self.radius}

    def __copy__(self):
        return Cylinder3D(self.center, self.axis, self.radius)

    def __key(self):
        """A tuple based on the object properties, useful for hashing."""
        return (self._center, self._axis, self._radius)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(other, Cylinder3D) and self.__key() == other.__key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __repr__(self):
        return 'Cylinder3D (center {}) (axis {}) (radius {})'.\
            format(self.center, self.axis, self.radius)
