# coding=utf-8
"""3D Vector and 3D Point"""
from __future__ import division

from ._immutable import immutable

import math
import operator


class Vector3:
    """3D Vector object.

    Properties:
        x
        y
        z
        magnitude
        magnitude_squared
    """
    __slots__ = ('x', 'y', 'z')
    __hash__ = None
    _mutable = True
    _type = 'Vector3'

    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

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
        return self.x ** 2 + self.y ** 2 + self.z ** 2

    def normalize(self):
        """Convert this vector to a unit vector (magnitude=1) with the same direction."""
        d = self.magnitude
        if d:
            self.x /= d
            self.y /= d
            self.z /= d

    def normalized(self):
        """Get a copy of the vector that is a unit vector (magnitude=1)."""
        d = self.magnitude
        if d:
            return Vector3(self.x / d, self.y / d, self.z / d)
        return self.duplicate()

    def reverse(self):
        """Convert this vector into one with a reversed direction."""
        self.x = -self.x
        self.y = -self.y
        self.z = -self.z

    def reversed(self):
        """Get a copy of this vector that is reversed."""
        return self.__neg__()

    def dot(self, other):
        """Get the dot product of this vector with another."""
        assert isinstance(other, (Vector3, Vector3Immutable)), \
            'other must be a Vector3 to use `dot()` with {}'.format(self.__class__)
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        """Get the cross product of this vector and another vector."""
        assert isinstance(other, (Vector3, Vector3Immutable)), \
            'other must be a Vector3 to use `cross()` with {}'.format(self.__class__)
        return Vector3(self.y * other.z - self.z * other.y,
                       -self.x * other.z + self.z * other.x,
                       self.x * other.y - self.y * other.x)

    def reflect(self, normal):
        """Get a vector that is reflected across the input normal vector.

        Note that the normal should be a Vector3 and should be normalized.
        """
        assert isinstance(normal, (Vector3, Vector3Immutable)), \
            'normal must be a Vector3 to use `reflect()` with {}'.format(self.__class__)
        d = 2 * (self.x * normal.x + self.y * normal.y + self.z * normal.z)
        return Vector3(self.x - d * normal.x,
                       self.y - d * normal.y,
                       self.z - d * normal.z)

    def rotate_around(self, axis, theta):
        """Return the vector rotated around axis through angle theta.

        Right hand rule applies.

        Args:
            axis: A Vector3 axis representing the axis of rotation.
            theta: An angle in radians.
        """
        # Adapted from equations published by Glenn Murray.
        # http://inside.mines.edu/~gmurray/ArbitraryAxisRotation/ArbitraryAxisRotation.html
        x, y, z = self.x, self.y, self.z
        u, v, w = axis.x, axis.y, axis.z

        # Extracted common factors for simplicity and efficiency
        r2 = u**2 + v**2 + w**2
        r = math.sqrt(r2)
        ct = math.cos(theta)
        st = math.sin(theta) / r
        dt = (u * x + v * y + w * z) * (1 - ct) / r2
        return Vector3((u * dt + x * ct + (-w * y + v * z) * st),
                       (v * dt + y * ct + (w * x - u * z) * st),
                       (w * dt + z * ct + (-v * x + u * y) * st))

    def angle(self, other):
        """Get the smallest angle between this vector and another."""
        return math.acos(self.dot(other) / (self.magnitude * other.magnitude))

    def to_immutable(self):
        """Get an immutable version of this vector."""
        return Vector3Immutable(self.x, self.y)

    def duplicate(self):
        """Get a copy of this vector."""
        return self.__copy__()

    def __copy__(self):
        return self.__class__(self.x, self.y, self.z)

    def __eq__(self, other):
        if isinstance(other, (Vector3, Vector3Immutable, Point3, Point3Immutable)):
            return self.x == other.x and self.y == other.y and self.z == other.z
        else:
            if hasattr(other, '__len__') and len(other) == 3:
                return self.x == other[0] and self.y == other[1] and self.z == other[2]
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __nonzero__(self):
        return self.x != 0 or self.y != 0 or self.z != 0

    def __len__(self):
        return 3

    def __getitem__(self, key):
        return (self.x, self.y, self.z)[key]

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __add__(self, other):
        if isinstance(other, (Vector3, Vector3Immutable, Point3, Point3Immutable)):
            # Vector + Vector -> Vector
            # Vector + Point -> Point
            # Point + Point -> Vector
            if self._type == other._type:
                _class = Vector3
            else:
                _class = Point3
            return _class(self.x + other.x, self.y + other.y, self.z + other.z)
        else:
            assert hasattr(other, '__len__') and len(other) == 3, \
                'Cannot add types {} and {}'.format(type(self), type(other))
            return Vector3(self.x + other[0], self.y + other[1], self.z + other[2])

    __radd__ = __add__

    def __iadd__(self, other):
        if isinstance(other, (Vector3, Vector3Immutable, Point3, Point3Immutable)):
            self.x += other.x
            self.y += other.y
            self.z += other.z
        else:
            self.x += other[0]
            self.y += other[1]
            self.z += other[2]
        return self

    def __sub__(self, other):
        if isinstance(other, (Vector3, Vector3Immutable, Point3, Point3Immutable)):
            # Vector - Vector -> Vector
            # Vector - Point -> Point
            # Point - Point -> Vector
            if self._type == other._type:
                _class = Vector3
            else:
                _class = Point3
            return _class(self.x - other.x, self.y - other.y, self.z - other.z)
        else:
            assert hasattr(other, '__len__') and len(other) == 3, \
                'Cannot subtract types {} and {}'.format(type(self), type(other))
            return Vector3(self.x - other[0], self.y - other[1], self.z - other[2])

    def __rsub__(self, other):
        if isinstance(other, (Vector3, Vector3Immutable, Point3, Point3Immutable)):
            return Vector3(other.x - self.x, other.y - self.y, other.z - self.z)
        else:
            assert hasattr(other, '__len__') and len(other) == 3, \
                'Cannot subtract types {} and {}'.format(type(self), type(other))
            return Vector3(other.x - self[0], other.y - self[1], other.z - self[2])

    def __mul__(self, other):
        if isinstance(other, (Vector3, Vector3Immutable, Point3, Point3Immutable)):
            if self._type == 'Point3' or other._type == 'Point3':
                _class = Point3
            else:
                _class = Vector3
            return _class(self.x * other.x, self.y * other.y, self.z * other.z)
        else:
            assert type(other) in (int, long, float), \
                'Cannot multiply types {} and {}'.format(type(self), type(other))
            return Vector3(self.x * other, self.y * other, self.z * other)

    __rmul__ = __mul__

    def __imul__(self, other):
        assert type(other) in (int, long, float), \
            'Cannot multiply types {} and {}'.format(type(self), type(other))
        self.x *= other
        self.y *= other
        self.z *= other
        return self

    def __div__(self, other):
        assert type(other) in (int, long, float), \
            'Cannot divide types {} and {}'.format(type(self), type(other))
        return Vector3(operator.div(self.x, other),
                       operator.div(self.y, other),
                       operator.div(self.z, other))

    def __rdiv__(self, other):
        assert type(other) in (int, long, float), \
            'Cannot divide types {} and {}'.format(type(self), type(other))
        return Vector3(operator.div(other, self.x),
                       operator.div(other, self.y),
                       operator.div(other, self.z))

    def __floordiv__(self, other):
        assert type(other) in (int, long, float), \
            'Cannot divide types {} and {}'.format(type(self), type(other))
        return Vector3(operator.floordiv(self.x, other),
                       operator.floordiv(self.y, other),
                       operator.floordiv(self.z, other))

    def __rfloordiv__(self, other):
        assert type(other) in (int, long, float), \
            'Cannot divide types {} and {}'.format(type(self), type(other))
        return Vector3(operator.floordiv(other, self.x),
                       operator.floordiv(other, self.y),
                       operator.floordiv(other, self.z))

    def __truediv__(self, other):
        assert type(other) in (int, long, float), \
            'Cannot divide types {} and {}'.format(type(self), type(other))
        return Vector3(operator.truediv(self.x, other),
                       operator.truediv(self.y, other),
                       operator.truediv(self.z, other))

    def __rtruediv__(self, other):
        assert type(other) in (int, long, float), \
            'Cannot divide types {} and {}'.format(type(self), type(other))
        return Vector3(operator.truediv(other, self.x),
                       operator.truediv(other, self.y),
                       operator.truediv(other, self.z))

    def __neg__(self):
        return Vector3(-self.x, -self.y, -self.z)

    __pos__ = __copy__

    def __abs__(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __repr__(self):
        """Vector3 representation."""
        return 'Ladybug Vector3 (%.2f, %.2f, %.2f)' % (self.x, self.y, self.z)


class Point3(Vector3):
    """3D Point object.

    Properties:
        x
        y
        z
    """
    _type = 'Point3'

    def distance_to_point(self, point):
        """Get the distance from this point to another Point3."""
        vec = (self.x - point.x, self.y - point.y, self.z - point.z)
        return math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)

    def distance_to_line_segment(self, line):
        """Get the minimum distance between this point and a LineSegment3."""
        close_pt = self.closest_point_line_segment(line)
        return self.distance_to_point(close_pt)

    def closest_point_line_segment(self, line):
        """Get the closest point on a LineSegment3 to this point."""
        d = line.v.magnitude_squared
        assert d != 0, 'Line segment length must not equal 0.'
        u = ((self.x - line.p.x) * line.v.x +
             (self.y - line.p.y) * line.v.y +
             (self.z - line.p.z) * line.v.z) / d
        if not line._u_in(u):
            u = max(min(u, 1.0), 0.0)
        return Point3(line.p.x + u * line.v.x,
                      line.p.y + u * line.v.y,
                      line.p.z + u * line.v.z)

    def to_immutable(self):
        """Get an immutable version of this point."""
        return Point3Immutable(self.x, self.y)

    def __repr__(self):
        """Point3 representation."""
        return 'Ladybug Point3 (%.2f, %.2f)' % (self.x, self.y)


@immutable
class Vector3Immutable(Vector3):
    """Immutable 3D Vector object."""
    _mutable = False

    def to_mutable(self):
        """Get a mutable version of this vector."""
        return Vector3(self.x, self.y, self.z)


@immutable
class Point3Immutable(Point3):
    """Immutable 3D Point object."""
    _mutable = False

    def to_mutable(self):
        """Get a mutable version of this point."""
        return Point3(self.x, self.y, self.z)
