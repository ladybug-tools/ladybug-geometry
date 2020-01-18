# coding=utf-8
"""Base class for all 1D geometries in 3D space (Ray3D and LineSegment3D)."""
from __future__ import division

from .pointvector import Point3D, Vector3D
from ..intersection3d import closest_point3d_on_line3d, \
    closest_point3d_on_line3d_infinite


class Base1DIn3D(object):
    """Base class for all 1D geometries in 3D space (Ray3D and LineSegment3D).

    Args:
        p: A Point2D representing the base.
        v: A Vector2D representing the direction.

    Properties:
        * p
        * v
    """
    __slots__ = ('_p', '_v')

    def __init__(self, p, v):
        """Initilize Base1DIn3D.
        """
        assert isinstance(p, Point3D), "Expected Point3D. Got {}.".format(type(p))
        assert isinstance(v, Vector3D), "Expected Vector3D. Got {}.".format(type(v))
        self._p = p
        self._v = v

    @classmethod
    def from_dict(cls, data):
        """Create a LineSegment3D/Ray3D from a dictionary.

        Args:
            data: A python dictionary in the following format

        .. code-block:: python

            {
            "p": (10, 0, 0),
            "v": (10, 10, 0)
            }
        """
        return cls(Point3D.from_array(data['p']),
                   Vector3D.from_array(data['v']))

    @property
    def p(self):
        """Base point."""
        return self._p

    @property
    def v(self):
        """Direction vector."""
        return self._v

    def is_parallel(self, line_ray, angle_tolerance):
        """Test whether this object is parallel to another LineSegment3D or Ray3D.

        Args:
            line_ray: Another LineSegment3D or Ray3D for which parallelization
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

    def is_colinear(self, line_ray, tolerance, angle_tolerance):
        """Test whether this object is colinear to another LineSegment3D or Ray3D.

        Args:
            line_ray: Another LineSegment3D or Ray3D for which colinearity
                with this object will be tested.
            tolerance: The maximum distance between the line_ray and the infinite
                extension of this object for them to be cinsidered colinear.
            angle_tolerance: The max angle in radians that the direction between
                this object and another can vary for them to be considered
                parallel.
        """
        if not self.is_parallel(line_ray, angle_tolerance):
            return False
        _close_pt = closest_point3d_on_line3d_infinite(self.p, line_ray)
        if self.p.distance_to_point(_close_pt) >= tolerance:
            return False
        return True

    def closest_point(self, point):
        """Get the closest Point3D on this object to another Point3D.

        Args:
            point: A Point3D object to which the closest point on this object
                will be computed.

        Returns:
            Point3D for the closest point on this line/ray to the input point.
        """
        return closest_point3d_on_line3d(point, self)

    def distance_to_point(self, point):
        """Get the minimum distance between this object and the input point.

        Args:
            point: A Point3D object to which the minimum distance will be computed.

        Returns:
            The distance to the input point.
        """
        close_pt = self.closest_point(point)
        return point.distance_to_point(close_pt)

    def duplicate(self):
        """Get a copy of this object."""
        return self.__copy__()

    def to_dict(self):
        """Get LineSegment3D/Ray3D as a dictionary."""
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
        return isinstance(other, Base1DIn3D) and self.__key() == other.__key()
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __repr__(self):
        """Base1Din3D representation."""
        return 'Base 1D Object (3D Space)'
