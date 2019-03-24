# coding=utf-8
"""2D Ray"""
from __future__ import division

from .._immutable import immutable
from .pointvector import Point2D, Point2DImmutable, Vector2D, Vector2DImmutable
from ..intersection2d import intersect_line2d_line2d, closest_point2d_on_line2d


class Ray2D(object):
    """2D Ray object.

    Properties:
        p: Base point
        v: Direction vector
    """
    __slots__ = ('_p', '_v')
    _mutable = True

    def __init__(self, p, v):
        """Initilize Ray2D.

        Args:
            p: A Point2D representing the base of the ray.
            v: A Vector2D representing the direction of the ray.
        """
        self.p = p
        self.v = v

    @property
    def is_mutable(self):
        """Boolean to note whether the object is mutable."""
        return self._mutable

    @property
    def p(self):
        """Base point."""
        return self._p

    @p.setter
    def p(self, p):
        assert isinstance(p, (Point2D, Point2DImmutable)), \
            "Expected Point2D. Got {}.".format(type(p))
        self._p = p.duplicate()

    @property
    def v(self):
        """Direction vector."""
        return self._v

    @v.setter
    def v(self, v):
        assert isinstance(v, (Vector2D, Vector2DImmutable)), \
            "Expected Vector2D. Got {}.".format(type(v))
        self._v = v.duplicate()

    def reverse(self):
        """Reverse the direction of the ray."""
        self.v.reverse()

    def reversed(self):
        """Get a copy of this ray that is reversed."""
        return Ray2D(self.p, self.v.reversed())

    def move(self, moving_vec):
        """Get a ray that has been moved along a vector.

        Args:
            moving_vec: A Vector2D with the direction and distance to move the ray.
        """
        return Ray2D(self.p + moving_vec, self.v)

    def scale(self, factor, origin):
        """Scale a ray by a factor from an origin point.

        Args:
            factor: A number representing how much the point should be scaled.
            origin: A Point2D representing the origin from which to scale.
        """
        return Ray2D(self.p.scale(factor, origin), self.v * factor)

    def rotate(self, angle, origin):
        """Get a ray that is rotated counterclockwise by a certain angle.

        Args:
            angle: An angle for rotation in radians.
            origin: A Point2D for the origin around which the point will be rotated.
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

    def intersect_line2(self, other):
        """Get the intersection between this ray and another Ray2D or LineSegment2.

        Args:
            other: Another LineSegment2 or Ray2D or to intersect.

        Returns:
            Point2D of intersection if it exists. None if no intersection exists.
        """
        return intersect_line2d_line2d(self, other)

    def closest_point(self, point):
        """Get the closest Point2D on this ray to the input point.

        Args:
            point: A Point2D object to which the closest point on this ray
                will be computed.

        Returns:
            Point2D for the closest point on this ray to the input point.
        """
        return closest_point2d_on_line2d(point, self)

    def distance_to_point(self, point):
        """Get the minimum distance between this ray and the input point.

        Args:
            point: A Point2D object to which the minimum distance will be computed.

        Returns:
            The distance to the input point.
        """
        close_pt = self.closest_point(point)
        return point.distance_to_point(close_pt)

    def to_immutable(self):
        """Get an immutable version of this point."""
        return Ray2DImmutable(self.p, self.v)

    def duplicate(self):
        """Get a copy of this object."""
        return self.__copy__()

    def __copy__(self):
        return self.__class__(self.p, self.v)

    def _u_in(self, u):
        return u >= 0.0

    def __repr__(self):
        return 'Ray2D(<%.2f, %.2f> + u<%.2f, %.2f>)' % \
            (self.p.x, self.p.y, self.v.x, self.v.y)


@immutable
class Ray2DImmutable(Ray2D):
    """Immutable 2D Ray object."""
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
        return Ray2D(self.p.to_mutable(), self.v.to_mutable())
