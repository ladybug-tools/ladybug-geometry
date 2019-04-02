# coding=utf-8
"""3D Ray"""
from __future__ import division

from .._immutable import immutable
from ._1d import Base1DIn3D


class Ray3D(Base1DIn3D):
    """3D Ray object.

    Properties:
        p: Base point
        v: Direction vector
    """
    __slots__ = ('_p', '_v')

    def __init__(self, p, v):
        """Initilize Ray3D.

        Args:
            p: A Point3D representing the base of the ray.
            v: A Vector3D representing the direction of the ray.
        """
        self.p = p
        self.v = v

    def reverse(self):
        """Reverse the direction of the ray."""
        self.v.reverse()

    def reversed(self):
        """Get a copy of this ray that is reversed."""
        return Ray3D(self.p, self.v.reversed())

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

    def scale(self, factor, origin):
        """Scale a ray by a factor from an origin point.

        Args:
            factor: A number representing how much the ray should be scaled.
            origin: A Point3D representing the origin from which to scale.
        """
        return Ray3D(self.p.scale(factor, origin), self.v * factor)

    def scale_world_origin(self, factor):
        """Scale a ray by a factor from the world origin. Faster than Ray2D.scale.

        Args:
            factor: A number representing how much the ray should be scaled.
        """
        return Ray3D(self.p.scale_world_origin(factor), self.v * factor)

    def to_immutable(self):
        """Get an immutable version of this object."""
        return Ray3DImmutable(self.p, self.v)

    def _u_in(self, u):
        return u >= 0.0

    def __repr__(self):
        return 'Ray3D (<%.2f, %.2f, %.2f> point) (<%.2f, %.2f, %.2f> vector)' % \
            (self.p.x, self.p.y, self.p.z, self.v.x, self.v.y, self.v.z)


@immutable
class Ray3DImmutable(Ray3D):
    """Immutable 3D Ray object."""
    _mutable = False

    def __init__(self, p, v):
        """Initilize Ray2D.

        Args:
            p: A Point2D representing the base of the ray.
            v: A Vector2D representing the direction of the ray.
        """
        self.p = p.to_immutable()
        self.v = v.to_immutable()

    def to_mutable(self):
        """Get a mutable version of this vector."""
        return Ray3D(self.p.to_mutable(), self.v.to_mutable())

    def to_immutable(self):
        """Get an immutable version of this object."""
        return self
