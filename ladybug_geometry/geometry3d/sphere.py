# coding=utf-8
"""Sphere"""
from __future__ import division

from .pointvector import Point3D


class Sphere(object):
    """Sphere object.

    Args:
        center
        radius

    Properties:
        * center
        * radius
    """
    __slots__ = ('_center')

    def __init__(self, center, radius):
        """Initilize Arc3D.
        """
        assert isinstance(center, Point3D), \
            "Expected Point3D. Got {}.".format(type(center))
        assert radius > 0, 'Arc radius must be greater than 0. Got {}.'.format(radius)
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

    def move(self, moving_vec):
        """Get a sphere that has been moved along a vector.

        Args:
            moving_vec: A Vector3D with the direction and distance to move the shere.
        """
        return Sphere(self._center.move(moving_vec), self.radius)

    def scale(self, factor, origin=None):
        """Scale a arc by a factor from an origin point.

        Args:
            factor: A number representing how much the sphere should be scaled.
            origin: A Point2D representing the origin from which to scale.
                If None, it will be scaled from the World origin (0, 0, 0).
        """
        return Sphere(self._center.scale(factor, origin), self._radius * factor)

    def duplicate(self):
        """Get a copy of this object."""
        return self.__copy__()

    def to_dict(self):
        """Get Sphere as a dictionary."""
        return {'type': 'Sphere', 'center': self._center.to_dict(),
                'radius': self._radius}

    def __copy__(self):
        return self.__class__(self._center, self._radius)

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __repr__(self):
        return 'Sphere (center {}) (radius {}))'.format(self._center, self._radius)
