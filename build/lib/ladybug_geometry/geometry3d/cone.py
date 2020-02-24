# coding=utf-8
"""Cone"""
from __future__ import division

from .pointvector import Point3D, Vector3D

import math


class Cone(object):
    """Cone object.

    Args:
        vertex: A Point3D at the tip of the cone.
        axis: A Vector3D representing the direction and height of the cone.
            The vector extends from the vertex to the center of the base.
        angle: An angle in radians representing the half angle between
            the axis and the surface.

    Properties:
        * vertex
        * axis
        * angle
        * height
        * slant_height
        * radius
        * area
        * volume
    """
    __slots__ = ('_vertex', '_axis', '_angle')

    def __init__(self, vertex, axis, angle):
        """Initilize Cone.
        """
        assert isinstance(vertex, Point3D), \
            "Expected Point3D. Got {}.".format(type(vertex))
        assert isinstance(axis, Vector3D), \
            "Expected Vector3D. Got {}.".format(type(axis))
        assert angle > 0, 'Cone angle must be greater than 0. Got {}.'.format(angle)
        self._vertex = vertex
        self._axis = axis
        self._angle = angle

    @classmethod
    def from_dict(cls, data):
        """Create a Cone from a dictionary.

        Args:
            data: A python dictionary in the following format

        .. code-block:: python

            {
            "type": "Cone"
            "vertex": (10, 0, 0),
            "axis": (0, 0, 1),
            "angle": 1.0,
            }
        """
        return cls(Point3D.from_array(data['vertex']),
                   Vector3D.from_array(data['axis']),
                   data['angle'])

    @property
    def vertex(self):
        """Vertex of cone."""
        return self._vertex

    @property
    def axis(self):
        """Axis of cone."""
        return self._axis

    @property
    def angle(self):
        """Angle of cone"""
        return self._angle

    @property
    def height(self):
        """Height of cone"""
        return self.axis.magnitude

    @property
    def radius(self):
        """Radius of a cone"""
        return self.height * math.tan(self.angle)

    @property
    def slant_height(self):
        """Slant height of a cone"""
        return math.sqrt(self.radius ** 2 + self.height ** 2)

    @property
    def area(self):
        """Surface area of a cone"""
        return math.pi * self.radius ** 2 + math.pi * self.radius * self.slant_height

    @property
    def volume(self):
        """Volume of a cone"""
        return 1 / 3 * math.pi * self.radius ** 2 * self.height

    def move(self, moving_vec):
        """Get a cone that has been moved along a vector.

        Args:
            moving_vec: A Vector3D with the direction and distance to move the cone.
        """
        return Cone(self.vertex.move(moving_vec), self.axis, self.angle)

    def rotate(self, axis, angle, origin):
        """Rotate this cone by a certain angle around an axis and origin.

        Right hand rule applies:
        If axis has a positive orientation, rotation will be clockwise.
        If axis has a negative orientation, rotation will be counterclockwise.

        Args:
            axis: A Vector3D axis representing the axis of rotation.
            angle: An angle for rotation in radians.
            origin: A Point3D for the origin around which the cone will be rotated.
        """
        return Cone(self.vertex.rotate(axis, angle, origin),
                    self.axis.rotate(axis, angle),
                    self.angle)

    def rotate_xy(self, angle, origin):
        """Get a cone that is rotated counterclockwise in the world XY plane by an angle.

        Args:
            angle: An angle for rotation in radians.
            origin: A Point3D for the origin around which the cone will be rotated.
        """
        return Cone(self.vertex.rotate_xy(angle, origin),
                    self.axis.rotate_xy(angle),
                    self.angle)

    def reflect(self, normal, origin):
        """Get a cone reflected across a plane with the input normal vector and origin.

        Args:
            normal: A Vector3D representing the normal vector for the plane across
                which the arc will be reflected. THIS VECTOR MUST BE NORMALIZED.
            origin: A Point3D representing the origin from which to reflect.
        """
        return Cone(self.vertex.reflect(normal, origin),
                    self.axis.reflect(normal),
                    self.angle)

    def scale(self, factor, origin=None):
        """Scale a cone by a factor from an origin point.

        Args:
            factor: A number representing how much the cone should be scaled.
            origin: A Point3D representing the origin from which to scale.
                If None, it will be scaled from the World origin (0, 0, 0).
        """
        return Cone(self.vertex.scale(factor, origin),
                    self.axis * factor,
                    self.angle)

    def duplicate(self):
        """Get a copy of this object."""
        return self.__copy__()

    def to_dict(self):
        """Get Cone as a dictionary."""
        return {'type': 'Cone', 'vertex': self.vertex.to_array(),
                'axis': self.axis.to_array(), 'angle': self.angle}

    def __copy__(self):
        return Cone(self.vertex, self.axis, self.angle)

    def __key(self):
        """A tuple based on the object properties, useful for hashing."""
        return (self._vertex, self._axis, self._angle)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(other, Cone) and self.__key() == other.__key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __repr__(self):
        return 'Cone (vertex {}) (axis {}) (angle {}) (height {})'.\
            format(self.vertex, self.axis, self.angle, self.height)
