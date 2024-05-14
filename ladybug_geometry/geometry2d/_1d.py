# coding=utf-8
"""Base class for all 1D geometries in 2D space (Ray2D and LineSegment2D)."""
from __future__ import division

from .pointvector import Vector2D, Point2D
from ..intersection2d import intersect_line2d, closest_point2d_on_line2d, \
    closest_point2d_on_line2d_infinite


class Base1DIn2D(object):
    """Base class for all 1D geometries in 2D space (Ray2D and LineSegment2D).

    Args:
        p: A Point2D representing the base.
        v: A Vector2D representing the direction.

    Properties:
        * p
        * v
        * min
        * max
        * center
    """
    __slots__ = ('_p', '_v')

    def __init__(self, p, v):
        """Initialize Base1DIn2D."""
        assert isinstance(p, Point2D), "Expected Point2D. Got {}.".format(type(p))
        assert isinstance(v, Vector2D), "Expected Vector2D. Got {}.".format(type(v))
        self._p = p
        self._v = v

    @classmethod
    def from_dict(cls, data):
        """Create a LineSegment2D/Ray2D from a dictionary.

        Args:
            data: A python dictionary in the following format

        .. code-block:: python

            {
            "p": (10, 0),
            "v": (10, 10)
            }
        """
        return cls(Point2D.from_array(data['p']),
                   Vector2D.from_array(data['v']))

    @property
    def p(self):
        """Base point."""
        return self._p

    @property
    def v(self):
        """Direction vector."""
        return self._v

    @property
    def min(self):
        """A Point2D for the minimum bounding rectangle vertex around this geometry."""
        p = self._p
        return Point2D(min(p.x, p.x + self.v.x), min(p.y, p.y + self.v.y))

    @property
    def max(self):
        """A Point2D for the maximum bounding rectangle vertex around this geometry."""
        p = self._p
        return Point2D(max(p.x, p.x + self.v.x), max(p.y, p.y + self.v.y))

    @property
    def center(self):
        """A Point2D for the center of the bounding rectangle around this geometry."""
        p = self._p
        return Point2D(p.x + (self.v.x / 2), p.y + (self.v.y / 2))

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

    def intersect_line_ray(self, line_ray):
        """Get the intersection between this object and another Ray2 or LineSegment2D.

        Args:
            line_ray: Another LineSegment2D or Ray2D or to intersect.

        Returns:
            Point2D of intersection if it exists. None if no intersection exists.
        """
        return intersect_line2d(self, line_ray)

    def is_parallel(self, line_ray, angle_tolerance):
        """Test whether this object is parallel to another LineSegment2D or Ray2D.

        Args:
            line_ray: Another LineSegment2D or Ray2D for which parallelization
                with this objects will be tested.
            angle_tolerance: The max angle in radians that the direction between
                this object and another can vary for them to be considered
                parallel.
        """
        if self.v.angle(line_ray.v) <= angle_tolerance:
            return True
        elif self.v.angle(line_ray.v.reverse()) <= angle_tolerance:
            return True
        return False

    def is_colinear(self, line_ray, tolerance, angle_tolerance=None):
        """Test whether this object is colinear to another LineSegment2D or Ray2D.

        Args:
            line_ray: Another LineSegment2D or Ray2D for which co-linearity
                with this object will be tested.
            tolerance: The maximum distance between the line_ray and the infinite
                extension of this object for them to be considered colinear.
            angle_tolerance: The max angle in radians that the direction between
                this object and another can vary for them to be considered
                parallel. If None, the angle tolerance will not be used to
                evaluate co-linearity and the lines will only be considered
                colinear if the endpoints of one line are within the tolerance
                distance of the other line. (Default: None).
        """
        if angle_tolerance is not None and \
                not self.is_parallel(line_ray, angle_tolerance):
            return False
        _close_pt = closest_point2d_on_line2d_infinite(self.p, line_ray)
        if self.p.distance_to_point(_close_pt) >= tolerance:
            return False
        return True

    def duplicate(self):
        """Get a copy of this object."""
        return self.__copy__()

    def to_dict(self):
        """Get LineSegment2D/Ray2D as a dictionary."""
        return {'p': self.p.to_array(),
                'v': self.v.to_array()}

    def __copy__(self):
        return self.__class__(self.p, self.v)

    def __key(self):
        """A tuple based on the object properties, useful for hashing."""
        return (hash(self.p), hash(self.v))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(other, Base1DIn2D) and self.__key() == other.__key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __repr__(self):
        """Base1Din2D representation."""
        return 'Base 1D Object (2D Space)'
