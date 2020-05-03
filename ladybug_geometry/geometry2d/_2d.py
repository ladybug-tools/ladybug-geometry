# coding=utf-8
"""Base class for 2D geometries in 2D space (Polygon2D and Mesh2D)."""
from __future__ import division

from .pointvector import Point2D


class Base2DIn2D(object):
    """Base class for 2D geometries in 2D space (Polygon2D and Mesh2D).

    Args:
        vertices: A list of Point2D objects representing the vertices.

    Properties:
        * vertices
        * min
        * max
        * center
    """
    __slots__ = ('_vertices', '_min', '_max', '_center')

    def __init__(self, vertices):
        """Initialize Base2DIn2D."""
        self._vertices = self._check_vertices_input(vertices)
        self._min = None
        self._max = None
        self._center = None

    @property
    def vertices(self):
        """Tuple of all vertices in this object."""
        return self._vertices

    @property
    def min(self):
        """A Point2D for the minimum bounding rectangle vertex around this geometry."""
        if self._min is None:
            self._calculate_min_max()
        return self._min

    @property
    def max(self):
        """A Point2D for the maximum bounding rectangle vertex around this geometry."""
        if self._max is None:
            self._calculate_min_max()
        return self._max

    @property
    def center(self):
        """A Point2D for the center of the bounding rectangle around this geometry."""
        if self._center is None:
            min, max = self.min, self.max
            self._center = Point2D((min.x + max.x) / 2, (min.y + max.y) / 2)
        return self._center

    def duplicate(self):
        """Get a copy of this object."""
        return self.__copy__()

    def _calculate_min_max(self):
        """Calculate maximum and minimum Point2D for this object."""
        min_pt = [self.vertices[0].x, self.vertices[0].y]
        max_pt = [self.vertices[0].x, self.vertices[0].y]

        for v in self.vertices[1:]:
            if v.x < min_pt[0]:
                min_pt[0] = v.x
            elif v.x > max_pt[0]:
                max_pt[0] = v.x
            if v.y < min_pt[1]:
                min_pt[1] = v.y
            elif v.y > max_pt[1]:
                max_pt[1] = v.y

        self._min = Point2D(min_pt[0], min_pt[1])
        self._max = Point2D(max_pt[0], max_pt[1])

    def _check_vertices_input(self, vertices):
        if not isinstance(vertices, tuple):
            vertices = tuple(vertices)
        assert len(vertices) >= 3, 'There must be at least 3 vertices for a {}.' \
            ' Got {}'.format(self.__class__.__name__, len(vertices))
        for vert in vertices:
            assert isinstance(vert, Point2D), \
                'Expected Point2D for {} vertex. Got {}.'.format(
                    self.__class__.__name__, type(vert))
        return vertices

    def __len__(self):
        return len(self._vertices)

    def __getitem__(self, key):
        return self._vertices[key]

    def __iter__(self):
        return iter(self._vertices)

    def __copy__(self):
        return Base2DIn2D(self._vertices)

    def __key(self):
        """A tuple based on the object properties, useful for hashing."""
        return tuple(hash(pt) for pt in self._vertices)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(other, Base2DIn2D) and self.__key() == other.__key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __repr__(self):
        """Base2Din2D representation."""
        return 'Base 2D Object (2D Space)'
