# coding=utf-8
"""Base class for 2D geometries in sD space (Surface3D and Mesh3D)."""
from __future__ import division

from .pointvector import Point3DImmutable


class Base2DIn3D(object):
    """Base class for 2D geometries in 3D space (Surface3D and Mesh3D).

    Properties:
        vertices
        min
        max
        center
    """
    _min = None
    _max = None
    _center = None

    @property
    def min(self):
        """A Point3D for the minimum bounding box vertex around this geometry."""
        if self._min is None:
            self._calculate_min_max()
        return self._min

    @property
    def max(self):
        """A Point3D for the maximum bounding box vertex around this geometry."""
        if self._max is None:
            self._calculate_min_max()
        return self._max

    @property
    def center(self):
        """A Point3D for the center of the bounding box around this geometry.

        For a Surface3D, this point will always lie within the plane of the surface.
        """
        if self._center is None:
            min, max = self.min, self.max
            self._center = Point3DImmutable(
                (min.x + max.x) / 2, (min.y + max.y) / 2, (min.z + max.z) / 2)
        return self._center

    def duplicate(self):
        """Get a copy of this object."""
        return self.__copy__()

    def _calculate_min_max(self):
        """Calculate maximum and minimum Point3D for this polygon."""
        min_pt = self.vertices[0].to_mutable()
        max_pt = self.vertices[0].to_mutable()

        for v in self.vertices[1:]:
            if v.x < min_pt.x:
                min_pt.x = v.x
            elif v.x > max_pt.x:
                max_pt.x = v.x
            if v.y < min_pt.y:
                min_pt.y = v.y
            elif v.y > max_pt.y:
                max_pt.y = v.y
            if v.z < min_pt.z:
                min_pt.z = v.z
            elif v.z > max_pt.z:
                max_pt.z = v.z

        self._min = min_pt.to_immutable()
        self._max = max_pt.to_immutable()

    def __len__(self):
        return len(self.vertices)

    def __getitem__(self, key):
        return self.vertices[key]

    def __iter__(self):
        return iter(self.vertices)

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()
