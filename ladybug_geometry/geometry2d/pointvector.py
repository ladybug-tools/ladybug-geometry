# coding=utf-8
"""2D Vector and 2D Point"""
from __future__ import division

from .._immutable import immutable

import math
import operator


class Vector2D(object):
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
            return Vector2D(self.x / d, self.y / d)
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
        return self.x * other.x + self.y * other.y

    def determinant(self, other):
        """Get the determinant between this vector and another 2D vector."""
        return self.x * other.y - self.y * other.x

    def cross(self):
        """Get the cross product of this vector."""
        return Vector2D(self.y, -self.x)

    def angle(self, other):
        """Get the smallest angle between this vector and another."""
        return math.acos(self.dot(other) / (self.magnitude * other.magnitude))

    def angle_counterclockwise(self, other):
        """Get the counterclockwise angle between this vector and another."""
        inner = self.angle(other)
        det = self.determinant(other)
        if det >= 0:
            return inner  # if the det > 0 then self is immediately clockwise of other
        else:
            return 2 * math.pi - inner  # if the det < 0 then other is clockwise of self

    def angle_clockwise(self, other):
        """Get the clockwise angle between this vector and another."""
        inner = self.angle(other)
        det = self.determinant(other)
        if det <= 0:
            return inner  # if the det > 0 then self is immediately clockwise of other
        else:
            return 2 * math.pi - inner  # if the det < 0 then other is clockwise of self

    def rotate(self, angle):
        """Get a vector that is rotated counterclockwise by a certain angle.

        Args:
            angle: An angle for rotation in radians.
        """
        return Vector2D._rotate(self, angle)

    def reflect(self, normal):
        """Get a vector that is reflected across a plane with the input normal vector.

        Args:
            normal: A Vector2D representing the normal vector for the plane across
                which the vector will be reflected. THIS VECTOR MUST BE NORMALIZED.
        """
        return Vector2D._reflect(self, normal)

    def to_mutable(self):
        """Get a mutable version of this object."""
        return self

    def to_immutable(self):
        """Get an immutable version of this object."""
        return Vector2DImmutable(self.x, self.y)

    def duplicate(self):
        """Get a copy of this vector."""
        return self.__copy__()

    @staticmethod
    def _rotate(vec, angle):
        """Hidden rotation method used by both Point2D and Vector2D."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        qx = cos_a * vec.x - sin_a * vec.y
        qy = sin_a * vec.x + cos_a * vec.y
        return Vector2D(qx, qy)

    @staticmethod
    def _reflect(vec, normal):
        """Hidden reflection method used by both Point2D and Vector2D."""
        d = 2 * (vec.x * normal.x + vec.y * normal.y)
        return Vector2D(vec.x - d * normal.x, vec.y - d * normal.y)

    def __copy__(self):
        return self.__class__(self.x, self.y)

    def __eq__(self, other):
        if isinstance(other, (Vector2D, Vector2DImmutable, Point2D, Point2DImmutable)):
            return self.x == other.x and self.y == other.y
        else:
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
        # Vector + Point -> Point
        # Vector + Vector -> Vector
        if isinstance(other, (Point2D, Point2DImmutable)):
            return Point2D(self.x + other.x, self.y + other.y)
        elif isinstance(other, (Vector2D, Vector2DImmutable)):
            return Vector2D(self.x + other.x, self.y + other.y)
        else:
            raise TypeError('Cannot add Vector2D and {}'.format(type(other)))

    __radd__ = __add__

    def __iadd__(self, other):
        if isinstance(other, (Vector2D, Vector2DImmutable, Point2D, Point2DImmutable)):
            self.x += other.x
            self.y += other.y
        else:
            self.x += other[0]
            self.y += other[1]
        return self

    def __sub__(self, other):
        # Vector - Point -> Point
        # Vector - Vector -> Vector
        if isinstance(other, (Point2D, Point2DImmutable)):
            return Point2D(self.x - other.x, self.y - other.y)
        elif isinstance(other, (Vector2D, Vector2DImmutable)):
            return Vector2D(self.x - other.x, self.y - other.y)
        else:
            raise TypeError('Cannot subtract Vector2D and {}'.format(type(other)))

    def __rsub__(self, other):
        if isinstance(other, (Vector2D, Vector2DImmutable, Point2D, Point2DImmutable)):
            return Vector2D(other.x - self.x, other.y - self.y)
        else:
            assert hasattr(other, '__len__') and len(other) == 2, \
                'Cannot subtract types {} and {}'.format(type(self), type(other))
            return Vector2D(other.x - self[0], other.y - self[1])

    def __mul__(self, other):
        assert type(other) in (int, float), \
            'Cannot multiply types {} and {}'.format(type(self), type(other))
        return Vector2D(self.x * other, self.y * other)

    __rmul__ = __mul__

    def __imul__(self, other):
        assert type(other) in (int, float), \
            'Cannot multiply types {} and {}'.format(type(self), type(other))
        self.x *= other
        self.y *= other
        return self

    def __div__(self, other):
        assert type(other) in (int, float), \
            'Cannot divide types {} and {}'.format(type(self), type(other))
        return Vector2D(operator.div(self.x, other), operator.div(self.y, other))

    def __rdiv__(self, other):
        assert type(other) in (int, float), \
            'Cannot divide types {} and {}'.format(type(self), type(other))
        return Vector2D(operator.div(other, self.x), operator.div(other, self.y))

    def __floordiv__(self, other):
        assert type(other) in (int, float), \
            'Cannot divide types {} and {}'.format(type(self), type(other))
        return Vector2D(operator.floordiv(self.x, other),
                        operator.floordiv(self.y, other))

    def __rfloordiv__(self, other):
        assert type(other) in (int, float), \
            'Cannot divide types {} and {}'.format(type(self), type(other))
        return Vector2D(operator.floordiv(other, self.x),
                        operator.floordiv(other, self.y))

    def __truediv__(self, other):
        assert type(other) in (int, float), \
            'Cannot divide types {} and {}'.format(type(self), type(other))
        return Vector2D(operator.truediv(self.x, other),
                        operator.truediv(self.y, other))

    def __rtruediv__(self, other):
        assert type(other) in (int, float), \
            'Cannot divide types {} and {}'.format(type(self), type(other))
        return Vector2D(operator.truediv(other, self.x),
                        operator.truediv(other, self.y))

    def __neg__(self):
        return Vector2D(-self.x, -self.y)

    __pos__ = __copy__

    def __abs__(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __repr__(self):
        """Vector2D representation."""
        return 'Vector2D (%.2f, %.2f)' % (self.x, self.y)


class Point2D(Vector2D):
    """2D Point object.

    Properties:
        x
        y
    """

    def move(self, moving_vec):
        """Get a point that has been moved along a vector.

        Args:
            moving_vec: A Vector2D with the direction and distance to move the point.
        """
        return Point2D(self.x + moving_vec.x, self.y + moving_vec.y)

    def rotate(self, angle, origin):
        """Rotate a point counterclockwise by a certain angle around an origin.

        Args:
            angle: An angle for rotation in radians.
            origin: A Point2D for the origin around which the point will be rotated.
        """
        return Point2D._rotate(self - origin, angle) + origin

    def reflect(self, normal, origin):
        """Get a point reflected across a plane with the input normal vector and origin.

        Args:
            normal: A Vector2D representing the normal vector for the plane across
                which the point will be reflected. THIS VECTOR MUST BE NORMALIZED.
            origin: A Point2D representing the origin from which to reflect.
        """
        return Point2D._reflect(self - origin, normal) + origin

    def scale(self, factor, origin):
        """Scale a point by a factor from an origin point.

        Args:
            factor: A number representing how much the point should be scaled.
            origin: A Point2D representing the origin from which to scale.
        """
        return (factor * (self - origin)) + origin

    def scale_world_origin(self, factor):
        """Scale a point by a factor from the world origin. Faster than Point2D.scale.

        Args:
            factor: A number representing how much the point should be scaled.
        """
        return Point2D(self.x * factor, self.y * factor)

    def distance_to_point(self, point):
        """Get the distance from this point to another Point2D."""
        vec = (self.x - point.x, self.y - point.y)
        return math.sqrt(vec[0] ** 2 + vec[1] ** 2)

    def to_immutable(self):
        """Get an immutable version of this object."""
        return Point2DImmutable(self.x, self.y)

    def __add__(self, other):
        # Point + Vector -> Point
        # Point + Point -> Vector
        if isinstance(other, (Vector2D, Vector2DImmutable)):
            return Point2D(self.x + other.x, self.y + other.y)
        elif isinstance(other, (Point2D, Point2DImmutable)):
            return Vector2D(self.x + other.x, self.y + other.y)
        else:
            raise TypeError('Cannot add Point2D and {}'.format(type(other)))

    def __sub__(self, other):
        # Point - Vector -> Point
        # Point - Point -> Vector
        if isinstance(other, (Vector2D, Vector2DImmutable)):
            return Point2D(self.x - other.x, self.y - other.y)
        elif isinstance(other, (Point2D, Point2DImmutable)):
            return Vector2D(self.x - other.x, self.y - other.y)
        else:
            raise TypeError('Cannot subtract Point2D and {}'.format(type(other)))

    def __repr__(self):
        """Point2D representation."""
        return 'Point2D (%.2f, %.2f)' % (self.x, self.y)


@immutable
class Vector2DImmutable(Vector2D):
    """Immutable 2D Vector object."""
    _mutable = False

    def to_mutable(self):
        """Get a mutable version of this object."""
        return Vector2D(self.x, self.y)

    def to_immutable(self):
        """Get an immutable version of this object."""
        return self


@immutable
class Point2DImmutable(Point2D):
    """Immutable 2D Point object."""
    _mutable = False

    def to_mutable(self):
        """Get a mutable version of this object."""
        return Point2D(self.x, self.y)

    def to_immutable(self):
        """Get an immutable version of this object."""
        return self
