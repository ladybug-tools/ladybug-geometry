# coding=utf-8
"""2D Ray"""
from __future__ import division

from .pointvector import Vector2D, Point2D
from ._1d import Base1DIn2D


class Ray2D(Base1DIn2D):
    """2D Ray object.

    Args:
        p: A Point2D representing the base of the ray.
        v: A Vector2D representing the direction of the ray.

    Properties:
        * p
        * v
        * min
        * max
    """
    __slots__ = ()

    def __init__(self, p, v):
        """Initialize Ray2D."""
        Base1DIn2D.__init__(self, p, v)

    @classmethod
    def from_array(cls, ray_array):
        """ Create a Ray2D from a nested array with a point and a vector.

        Args:
            ray_array: Nested tuples ((p.x, p.y), (v.x, v.y)).
        """
        return Ray2D(Point2D(*ray_array[0]), Vector2D(*ray_array[1]))

    def reverse(self):
        """Get a copy of this ray that is reversed."""
        return Ray2D(self.p, self.v.reverse())

    def move(self, moving_vec):
        """Get a ray that has been moved along a vector.

        Args:
            moving_vec: A Vector2D with the direction and distance to move the ray.
        """
        return Ray2D(self.p.move(moving_vec), self.v)

    def rotate(self, angle, origin):
        """Get a ray that is rotated counterclockwise by a certain angle.

        Args:
            angle: An angle for rotation in radians.
            origin: A Point2D for the origin around which the ray will be rotated.
        """
        return Ray2D(self.p.rotate(angle, origin), self.v.rotate(angle))

    def reflect(self, normal, origin):
        """Get a ray reflected across a plane with the input normal vector and origin.

        Args:
            normal: A Vector2D representing the normal vector for the plane across
                which the ray will be reflected. THIS VECTOR MUST BE NORMALIZED.
            origin: A Point2D representing the origin from which to reflect.
        """
        return Ray2D(self.p.reflect(normal, origin), self.v.reflect(normal))

    def scale(self, factor, origin=None):
        """Scale a ray by a factor from an origin point.

        Args:
            factor: A number representing how much the ray should be scaled.
            origin: A Point2D representing the origin from which to scale.
                If None, it will be scaled from the World origin (0, 0).
        """
        return Ray2D(self.p.scale(factor, origin), self.v * factor)

    def to_dict(self):
        """Get Ray2D as a dictionary."""
        base = Base1DIn2D.to_dict(self)
        base['type'] = 'Ray2D'
        return base

    def to_array(self):
        """A nested array representing the start point and vector."""
        return (self.p.to_array(), self.v.to_array())

    def _u_in(self, u):
        return u >= 0.0

    def __key(self):
        """A tuple based on the object properties, useful for hashing."""
        return (hash(self.p), hash(self.v))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(other, Ray2D) and self.__key() == other.__key()

    def __repr__(self):
        return 'Ray2D (point <%.2f, %.2f>) (vector <%.2f, %.2f>)' % \
            (self.p.x, self.p.y, self.v.x, self.v.y)
