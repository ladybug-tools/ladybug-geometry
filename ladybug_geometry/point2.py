# coding=utf-8
"""2D Vector and 2D Point"""
from __future__ import division

from ._immutable import immutable

import math
import operator


class Vector2(object):
    """2D Vector object.

    Properties:
        x
        y
        magnitude
        magnitude_squared
    """
    __slots__ = ('x', 'y')
    __hash__ = None
    _mutable = True
    _type = 'Vector2'

    def __init__(self, x=0, y=0):
        """Initialize 2D Vector."""
        self.x = x
        self.y = y

    @property
    def is_mutable(self):
        """Boolean to note whether the object is mutable."""
        return self._mutable

    @property
    def magnitude(self):
        """Get the magnitude of the vector."""
        return self.__abs__()

    @property
    def magnitude_squared(self):
        """Get the magnitude squared of the vector."""
        return self.x ** 2 + self.y ** 2

    def normalize(self):
        """Convert this vector to a unit vector (magnitude=1) with the same direction."""
        d = self.magnitude
        if d:
            self.x /= d
            self.y /= d

    def normalized(self):
        """Get a copy of the vector that is a unit vector (magnitude=1)."""
        d = self.magnitude
        if d:
            return Vector2(self.x / d, self.y / d)
        return self.duplicate()

    def reverse(self):
        """Convert this vector into one with a reversed direction."""
        self.x = -self.x
        self.y = -self.y

    def reversed(self):
        """Get a copy of this vector that is reversed."""
        return self.__neg__()

    def dot(self, other):
        """Get the dot product of this vector with another."""
        assert isinstance(other, (Vector2, Vector2Immutable)), \
            'other must be a Vector2 to use `dot()` with {}'.format(self.__class__)
        return self.x * other.x + self.y * other.y

    def cross(self):
        """Get the cross product of this vector."""
        return Vector2(self.y, -self.x)

    def reflect(self, normal):
        """Get a vector that is reflected across the input normal vector.

        Note that the normal should be a Vector2 and should be normalized.
        """
        assert isinstance(normal, (Vector2, Vector2Immutable)), \
            'normal must be a Vector2 to use `reflect()` with {}'.format(self.__class__)
        d = 2 * (self.x * normal.x + self.y * normal.y)
        return Vector2(self.x - d * normal.x, self.y - d * normal.y)

    def angle(self, other):
        """Get the smallest angle between this vector and another."""
        return math.acos(self.dot(other) / (self.magnitude * other.magnitude))

    def to_immutable(self):
        """Get an immutable version of this vector."""
        return Vector2Immutable(self.x, self.y)

    def duplicate(self):
        """Get a copy of this vector."""
        return self.__copy__()

    def __copy__(self):
        return self.__class__(self.x, self.y)

    def __eq__(self, other):
        if isinstance(other, (Vector2, Vector2Immutable, Point2, Point2Immutable)):
            return self.x == other.x and self.y == other.y
        else:
            if hasattr(other, '__len__') and len(other) == 2:
                return self.x == other[0] and self.y == other[1]
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __nonzero__(self):
        return self.x != 0 or self.y != 0

    def __len__(self):
        return 2

    def __getitem__(self, key):
        return (self.x, self.y)[key]

    def __iter__(self):
        return iter((self.x, self.y))

    def __add__(self, other):
        if isinstance(other, (Vector2, Vector2Immutable, Point2, Point2Immutable)):
            # Vector + Vector -> Vector
            # Vector + Point -> Point
            # Point + Point -> Vector
            if self._type == other._type:
                _class = Vector2
            else:
                _class = Point2
            return _class(self.x + other.x, self.y + other.y)
        else:
            assert hasattr(other, '__len__') and len(other) == 2, \
                'Cannot add types {} and {}'.format(type(self), type(other))
            return Vector2(self.x + other[0], self.y + other[1])

    __radd__ = __add__

    def __iadd__(self, other):
        if isinstance(other, (Vector2, Vector2Immutable, Point2, Point2Immutable)):
            self.x += other.x
            self.y += other.y
        else:
            self.x += other[0]
            self.y += other[1]
        return self

    def __sub__(self, other):
        if isinstance(other, (Vector2, Vector2Immutable, Point2, Point2Immutable)):
            # Vector - Vector -> Vector
            # Vector - Point -> Point
            # Point - Point -> Vector
            if self._type == other._type:
                _class = Vector2
            else:
                _class = Point2
            return _class(self.x - other.x, self.y - other.y)
        else:
            assert hasattr(other, '__len__') and len(other) == 2, \
                'Cannot subtract types {} and {}'.format(type(self), type(other))
            return Vector2(self.x - other[0], self.y - other[1])

    def __rsub__(self, other):
        if isinstance(other, (Vector2, Vector2Immutable, Point2, Point2Immutable)):
            return Vector2(other.x - self.x, other.y - self.y)
        else:
            assert hasattr(other, '__len__') and len(other) == 2, \
                'Cannot subtract types {} and {}'.format(type(self), type(other))
            return Vector2(other.x - self[0], other.y - self[1])

    def __mul__(self, other):
        assert type(other) in (int, long, float), \
            'Cannot multiply types {} and {}'.format(type(self), type(other))
        return Vector2(self.x * other, self.y * other)

    __rmul__ = __mul__

    def __imul__(self, other):
        assert type(other) in (int, long, float), \
            'Cannot multiply types {} and {}'.format(type(self), type(other))
        self.x *= other
        self.y *= other
        return self

    def __div__(self, other):
        assert type(other) in (int, long, float), \
            'Cannot divide types {} and {}'.format(type(self), type(other))
        return Vector2(operator.div(self.x, other),
                       operator.div(self.y, other))

    def __rdiv__(self, other):
        assert type(other) in (int, long, float), \
            'Cannot divide types {} and {}'.format(type(self), type(other))
        return Vector2(operator.div(other, self.x),
                       operator.div(other, self.y))

    def __floordiv__(self, other):
        assert type(other) in (int, long, float), \
            'Cannot divide types {} and {}'.format(type(self), type(other))
        return Vector2(operator.floordiv(self.x, other),
                       operator.floordiv(self.y, other))

    def __rfloordiv__(self, other):
        assert type(other) in (int, long, float), \
            'Cannot divide types {} and {}'.format(type(self), type(other))
        return Vector2(operator.floordiv(other, self.x),
                       operator.floordiv(other, self.y))

    def __truediv__(self, other):
        assert type(other) in (int, long, float), \
            'Cannot divide types {} and {}'.format(type(self), type(other))
        return Vector2(operator.truediv(self.x, other),
                       operator.truediv(self.y, other))

    def __rtruediv__(self, other):
        assert type(other) in (int, long, float), \
            'Cannot divide types {} and {}'.format(type(self), type(other))
        return Vector2(operator.truediv(other, self.x),
                       operator.truediv(other, self.y))

    def __neg__(self):
        return Vector2(-self.x, -self.y)

    __pos__ = __copy__

    def __abs__(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __repr__(self):
        """Vector2 representation."""
        return 'Ladybug Vector2 (%.2f, %.2f)' % (self.x, self.y)


class Point2(Vector2):
    """2D Point object.

    Properties:
        x
        y
    """
    _type = 'Point2'

    def distance_to_point(self, point):
        """Get the distance from this point to another Point2."""
        vec = (self.x - point.x, self.y - point.y)
        return math.sqrt(vec[0] ** 2 + vec[1] ** 2)

    def distance_to_line_segment(self, line):
        """Get the minimum distance between this point and a LineSegment2."""
        close_pt = self.closest_point_line_segment(line)
        return self.distance_to_point(close_pt)

    def closest_point_line_segment(self, line):
        """Get the closest point on a LineSegment2 to this point."""
        d = line.v.magnitude_squared
        assert d != 0, 'Line segment length must not equal 0.'
        u = ((self.x - line.p.x) * line.v.x + (self.y - line.p.y) * line.v.y) / d
        if not line._u_in(u):
            u = max(min(u, 1.0), 0.0)
        return Point2(line.p.x + u * line.v.x, line.p.y + u * line.v.y)

    def to_immutable(self):
        """Get an immutable version of this point."""
        return Point2Immutable(self.x, self.y)

    def __repr__(self):
        """Point2 representation."""
        return 'Ladybug Point2 (%.2f, %.2f)' % (self.x, self.y)


@immutable
class Vector2Immutable(Vector2):
    """Immutable 2D Vector object."""
    _mutable = False

    def to_mutable(self):
        """Get a mutable version of this vector."""
        return Vector2(self.x, self.y)


@immutable
class Point2Immutable(Point2):
    """Immutable 2D Point object."""
    _mutable = False

    def to_mutable(self):
        """Get a mutable version of this point."""
        return Point2(self.x, self.y)
