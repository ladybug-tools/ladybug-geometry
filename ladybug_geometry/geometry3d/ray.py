# coding=utf-8
"""3D Ray"""
from __future__ import division

from .pointvector import Point3D, Vector3D
from ._1d import Base1DIn3D


class Ray3D(Base1DIn3D):
    """3D Ray object.

    Args:
        p: A Point3D representing the base of the ray.
        v: A Vector3D representing the direction of the ray.

    Properties:
        * p
        * v
    """
    __slots__ = ()

    def __init__(self, p, v):
        """Initilize Ray3D.
        """
        assert isinstance(p, Point3D), "Expected Point3D. Got {}.".format(type(p))
        assert isinstance(v, Vector3D), "Expected Vector3D. Got {}.".format(type(v))
        self._p = p
        self._v = v

    def reverse(self):
        """Get a copy of this ray that is reversed."""
        return Ray3D(self.p, self.v.reverse())

    def move(self, moving_vec):
        """Get a ray that has been moved along a vector.

        Args:
            moving_vec: A Vector3D with the direction and distance to move the ray.
        """
        return Ray3D(self.p.move(moving_vec), self.v)

    def rotate(self, axis, angle, origin):
        """Rotate a ray by a certain angle around an axis and origin.

        Right hand rule applies:
        If axis has a positive orientation, rotation will be clockwise.
        If axis has a negative orientation, rotation will be counterclockwise.

        Args:
            axis: A Vector3D axis representing the axis of rotation.
            angle: An angle for rotation in radians.
            origin: A Point3D for the origin around which the object will be rotated.
        """
        return Ray3D(self.p.rotate(axis, angle, origin), self.v.rotate(axis, angle))

    def rotate_xy(self, angle, origin):
        """Get a ray rotated counterclockwise in the XY plane by a certain angle.

        Args:
            angle: An angle in radians.
            origin: A Point3D for the origin around which the object will be rotated.
        """
        return Ray3D(self.p.rotate_xy(angle, origin), self.v.rotate_xy(angle))

    def reflect(self, normal, origin):
        """Get a ray reflected across a plane with the input normal vector and origin.

        Args:
            normal: A Vector3D representing the normal vector for the plane across
                which the ray will be reflected. THIS VECTOR MUST BE NORMALIZED.
            origin: A Point3D representing the origin from which to reflect.
        """
        return Ray3D(self.p.reflect(normal, origin), self.v.reflect(normal))

    def scale(self, factor, origin=None):
        """Scale a ray by a factor from an origin point.

        Args:
            factor: A number representing how much the ray should be scaled.
            origin: A Point3D representing the origin from which to scale.
                If None, it will be scaled from the World origin (0, 0, 0).
        """
        return Ray3D(self.p.scale(factor, origin), self.v * factor)

    def scale_world_origin(self, factor):
        """Scale a ray by a factor from the world origin. Faster than Ray2D.scale.

        Args:
            factor: A number representing how much the ray should be scaled.
        """
        return Ray3D(self.p.scale_world_origin(factor), self.v * factor)

    def to_dict(self):
        """Get Ray3D as a dictionary."""
        base = Base1DIn3D.to_dict(self)
        base['type'] = 'Ray3D'
        return base

    def _u_in(self, u):
        return u >= 0.0
    
    def __copy__(self):
        return Ray3D(self.p, self.v)

    def __key(self):
        """A tuple based on the object properties, useful for hashing."""
        return (hash(self.p), hash(self.v))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(other, Ray3D) and self.__key() == other.__key()

    def __repr__(self):
        return 'Ray3D (point <%.2f, %.2f, %.2f>) (vector <%.2f, %.2f, %.2f>)' % \
            (self.p.x, self.p.y, self.p.z, self.v.x, self.v.y, self.v.z)
