# coding=utf-8
"""Base class for all 1D geometries in 3D space (Ray3D and LineSegment3D)."""
from __future__ import division

from .pointvector import Point3D, Point3DImmutable, Vector3D, Vector3DImmutable
from ..intersection3d import closest_point3d_on_line3d, closest_point3d_between_line3d


class Base1DIn3D(object):
    """Base class for all 1D geometries in 3D space (Ray3D and LineSegment3D).

    Properties:
        p: End Point3D of object
        v: Vector3D along object
    """
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
        assert isinstance(p, (Point3D, Point3DImmutable)), \
            "Expected Point3D. Got {}.".format(type(p))
        self._p = p.duplicate()

    @property
    def v(self):
        """Direction vector."""
        return self._v

    @v.setter
    def v(self, v):
        assert isinstance(v, (Vector3D, Vector3DImmutable)), \
            "Expected Vector3D. Got {}.".format(type(v))
        self._v = v.duplicate()

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

    def closest_points_between_line(self, line_ray):
        """Get the two closest Point3D between this object to another Line3D or Ray3D.

        Args:
            line_ray: A Line3D or Ray3D object to which the closest points
                will be computed.

        Returns:
            Two Point3D objects representing:
                1) The closest point on this object to the input line_ray.
                2) The closest point on the input line_ray to this object.
        """
        return closest_point3d_between_line3d(self, line_ray)

    def distance_to_line(self, line_ray):
        """Get the minimum distance between this object and the input Line3D or Ray3D.

        Args:
            line_ray: A Line3D or Ray3D object to which the minimum distance
                will be computed.

        Returns:
            The minimum distance to the input line_ray.
        """
        close_pt_a, cloase_pt_b = self.closest_point_on_line(line_ray)
        return close_pt_a.distance_to_point(cloase_pt_b)

    def intersect_line(self, line_ray, tolerance):
        """Get the intersection between this object and the input Line3D or Ray3D.

        Args:
            line_ray: A Line3D or Ray3D object for which intersection will be computed.
            tolerance: The minumum distance between this object and the input
                line_ray at which the two objects are considered intersected.

        Returns:
            Point3D for the intersection. Will be None if no intersection exists.
        """
        close_pt_a, close_pt_b = self.closest_point_on_line(line_ray)
        if close_pt_a.distance_to_point(close_pt_b) <= tolerance:
            return close_pt_a
        return None

    def duplicate(self):
        """Get a copy of this object."""
        return self.__copy__()

    def __copy__(self):
        return self.__class__(self.p, self.v)

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()
