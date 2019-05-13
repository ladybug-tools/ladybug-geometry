# coding=utf-8
"""Base class for all 1D geometries in 3D space (Ray3D and LineSegment3D)."""
from __future__ import division

from .pointvector import Point3D, Vector3D
from ..intersection3d import closest_point3d_on_line3d


class Base1DIn3D(object):
    """Base class for all 1D geometries in 3D space (Ray3D and LineSegment3D).

    Properties:
        p: End Point3D of object
        v: Vector3D along object
    """

    @property
    def p(self):
        """Base point."""
        return self._p

    @property
    def v(self):
        """Direction vector."""
        return self._v

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

    def __copy__(self):
        return self.__class__(self.p, self.v)

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()
