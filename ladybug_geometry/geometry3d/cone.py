# coding=utf-8
"""Cone3D"""
from __future__ import division

from .pointvector import Point3D, Vector3D

import math


class Cone3D(object):
    """Cone3D object.

    Args:
        vertex: A Point3D representing the vertex of the cone.
        axis: A Vector3D representing the direction of the cone.
        angle: A degree in radians representing the half angle between
            the axis and the surface.
        height: A number representing the height of the cone.

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
    __slots__ = ('_vertex', '_axis', '_angle', 'height')

    def __init__(self, vertex, axis, angle, height):
        """Initilize Cone3D.
        """
        assert isinstance(vertex, Point3D), \
            "Expected Point3D. Got {}.".format(type(vertex))
        assert isinstance(axis, Vector3D), \
            "Expected Vector3D. Got {}.".format(type(vertex))
        assert angle > 0, 'Cone angle must be greater than 0. Got {}.'.format(angle)
        assert height > 0, 'Cone height must be greater than 0. Got {}.'.format(height)
        self._vertex = vertex
        self._axis = axis
        self._angle = angle
        self._height = height

    @classmethod
    def from_dict(cls, data):
        """Create a Cone3D from a dictionary.

        Args:
            data: A python dictionary in the following format

        .. code-block:: python

            {
            "type": "Cone3D"
            "vertex": (10, 0, 0),
            "axis": (0, 0, 1),
            "angle": 1.0,
            "height": 1.0
            }
        """
        return cls(Point3D.from_array(data['vertex']),
                   Vector3D.from_array(data['axis']),
                   data['angle'],
                   data['height'])

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
        return self._height

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
        """Surface of a cone"""
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
        return Cone3D(self.vertex.move(moving_vec), self.axis, self.angle, self.height)

    def rotate(self, axis, angle, origin):
        """Rotate this cone by a certain angle around an axis and origin.

        Right hand rule applies:
        If axis has a positive orientation, rotation will be clockwise.
        If axis has a negative orientation, rotation will be counterclockwise.

        Args:
            axis: A Vector3D axis representing the axis of rotation.
            angle: An angle for rotation in radians.
            origin: A Point3D for the origin around which the sphere will be rotated.
        """
        return Cone3D(self.vertex.rotate(axis, angle, origin),
                      self.axis.rotate(axis, angle, origin),
                      self.angle,
                      self.height)

    def rotate_xy(self, angle, origin):
        """Get a cone that is rotated counterclockwise in the world XY plane by an angle.

        Args:
            angle: An angle for rotation in radians.
            origin: A Point3D for the origin around which the sphere will be rotated.
        """
        return Cone3D(self.vertex.rotate_xy(angle, origin),
                      self.axis.rotate_xy(angle, origin),
                      self.angle,
                      self.height)

    def reflect(self, normal, origin):
        """Get a cone reflected across a plane with the input normal vector and origin.

        Args:
            normal: A Vector3D representing the normal vector for the plane across
                which the arc will be reflected. THIS VECTOR MUST BE NORMALIZED.
            origin: A Point3D representing the origin from which to reflect.
        """
        return Cone3D(self.vertex.reflect(normal, origin),
                      self.axis.reflect(normal, origin),
                      self.angle,
                      self.height)

    def scale(self, factor, origin=None):
        """Scale a cone by a factor from an origin point.

        Args:
            factor: A number representing how much the cone should be scaled.
            origin: A Point3D representing the origin from which to scale.
                If None, it will be scaled from the World origin (0, 0, 0).
        """
        return Cone3D(self.origin.scale(factor, origin),
                      self.axis,
                      self.angle,
                      self.height)

    def duplicate(self):
        """Get a copy of this object."""
        return self.__copy__()

    def to_dict(self):
        """Get Cone as a dictionary."""
        return {'type': 'Cone3D', 'vertex': self.vertex.to_array(),
                'axis': self.axis.to_array(), 'angle': self.angle, 'height': self.height}

    def __copy__(self):
        return Cone3D(self.vertex, self.axis, self.angle, self.height)

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __repr__(self):
        return 'Cone3D (vertex {}) (axis {}) (angle {}) (height {}))'.format(self.vertex,
                                                                             self.axis,
                                                                             self.angle,
                                                                             self.height)
