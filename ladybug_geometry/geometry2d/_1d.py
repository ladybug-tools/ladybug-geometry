# coding=utf-8
"""Base class for all 1D geometries (Ray2D and LineSegment2D)."""
from __future__ import division

from .pointvector import Point2D, Point2DImmutable, Vector2D, Vector2DImmutable
from ..intersection2d import intersect_line2d, closest_point2d_on_line2d


class Base1D(object):
    """Base class for all 1D geometries (Ray2D and LineSegment2D).

    Properties:
        vertices
        min
        max
        center
    """
    __slots__ = ('_p', '_v')
    _mutable = True

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

    def closest_point(self, point):
        """Get the closest Point2D on this object to another Point2D.

        Args:
            point: A Point2D object to which the closest point on this object
                will be computed.

        Returns:
            Point2D for the closest point on this line to the input point.
        """
        return closest_point2d_on_line2d(point, self)

    def distance_to_point(self, point):
        """Get the minimum distance between this object and the input point.

        Args:
            point: A Point2D object to which the minimum distance will be computed.

        Returns:
            The distance to the input point.
        """
        close_pt = self.closest_point(point)
        return point.distance_to_point(close_pt)

    def intersect_line2(self, other):
        """Get the intersection between this object and another Ray2 or LineSegment2D.

        Args:
            other: Another LineSegment2D or Ray2 or to intersect.

        Returns:
            Point2D of intersection if it exists. None if no intersection exists.
        """
        return intersect_line2d(self, other)

    def duplicate(self):
        """Get a copy of this object."""
        return self.__copy__()

    def __copy__(self):
        return self.__class__(self.p, self.v)

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()
