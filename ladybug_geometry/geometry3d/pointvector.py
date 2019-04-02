# coding=utf-8
"""3D Vector and 3D Point"""
from __future__ import division

from .._immutable import immutable
from ..geometry2d.pointvector import Vector2D

import math
import operator


class Vector3D(object):
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

    def __init__(self, x=0, y=0, z=0):
        """Initialize 3D Vector."""
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
            return Vector3D(self.x / d, self.y / d, self.z / d)
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
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        """Get the cross product of this vector and another vector."""
        return Vector3D(self.y * other.z - self.z * other.y,
                        -self.x * other.z + self.z * other.x,
                        self.x * other.y - self.y * other.x)

    def angle(self, other):
        """Get the smallest angle between this vector and another."""
        return math.acos(self.dot(other) / (self.magnitude * other.magnitude))

    def rotate(self, axis, angle):
        """Get a vector rotated around an axis through an angle.

        Right hand rule applies:
        If axis has a positive orientation, rotation will be clockwise.
        If axis has a negative orientation, rotation will be counterclockwise.

        Args:
            axis: A Vector3D axis representing the axis of rotation.
            angle: An angle in radians.
        """
        return Vector3D._rotate(self, axis, angle)

    def rotate_xy(self, angle):
        """Get a vector rotated counterclockwise in the XY plane by a certain angle.

        Args:
            angle: An angle in radians.
        """
        vec_2 = Vector2D._rotate(self, angle)
        return Vector3D(vec_2.x, vec_2.y, self.z)

    def reflect(self, normal):
        """Get a vector that is reflected across a plane with the input normal vector.

        Args:
            normal: A Vector3D representing the normal vector for the plane across
                which the vector will be reflected. THIS VECTOR MUST BE NORMALIZED.
        """
        return Vector3D._reflect(self, normal)

    def to_immutable(self):
        """Get an immutable version of this vector."""
        return Vector3DImmutable(self.x, self.y, self.z)

    def duplicate(self):
        """Get a copy of this vector."""
        return self.__copy__()

    @staticmethod
    def _reflect(vec, normal):
        """Hidden reflection method used by both Point3D and Vector3D."""
        d = 2 * (vec.x * normal.x + vec.y * normal.y + vec.z * normal.z)
        return Vector3D(vec.x - d * normal.x,
                        vec.y - d * normal.y,
                        vec.z - d * normal.z)

    @staticmethod
    def _rotate(vec, axis, angle):
        """Hidden rotation method used by both Point3D and Vector3D."""
        # Adapted from equations published by Glenn Murray.
        # http://inside.mines.edu/~gmurray/ArbitraryAxisRotation/ArbitraryAxisRotation.html
        x, y, z = vec.x, vec.y, vec.z
        u, v, w = axis.x, axis.y, axis.z

        # Extracted common factors for simplicity and efficiency
        r2 = u**2 + v**2 + w**2
        r = math.sqrt(r2)
        ct = math.cos(angle)
        st = math.sin(angle) / r
        dt = (u * x + v * y + w * z) * (1 - ct) / r2
        return Vector3D((u * dt + x * ct + (-w * y + v * z) * st),
                        (v * dt + y * ct + (w * x - u * z) * st),
                        (w * dt + z * ct + (-v * x + u * y) * st))

    def __copy__(self):
        return self.__class__(self.x, self.y, self.z)

    def __eq__(self, other):
        if isinstance(other, (Vector3D, Vector3DImmutable, Point3D, Point3DImmutable)):
            return self.x == other.x and self.y == other.y and self.z == other.z
        else:
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
        # Vector + Point -> Point
        # Vector + Vector -> Vector
        if isinstance(other, (Point3D, Point3DImmutable)):
            return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)
        elif isinstance(other, (Vector3D, Vector3DImmutable)):
            return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
        else:
            raise TypeError('Cannot add Vector3D and {}'.format(type(other)))

    __radd__ = __add__

    def __iadd__(self, other):
        if isinstance(other, (Vector3D, Vector3DImmutable, Point3D, Point3DImmutable)):
            self.x += other.x
            self.y += other.y
            self.z += other.z
        else:
            self.x += other[0]
            self.y += other[1]
            self.z += other[2]
        return self

    def __sub__(self, other):
        # Vector - Point -> Point
        # Vector - Vector -> Vector
        if isinstance(other, (Point3D, Point3DImmutable)):
            return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)
        elif isinstance(other, (Vector3D, Vector3DImmutable)):
            Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
        else:
            raise TypeError('Cannot subtract Vector3D and {}'.format(type(other)))

    def __rsub__(self, other):
        if isinstance(other, (Vector3D, Vector3DImmutable, Point3D, Point3DImmutable)):
            return Vector3D(other.x - self.x, other.y - self.y, other.z - self.z)
        else:
            assert hasattr(other, '__len__') and len(other) == 3, \
                'Cannot subtract types {} and {}'.format(type(self), type(other))
            return Vector3D(other.x - self[0], other.y - self[1], other.z - self[2])

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vector3D(self.x * other, self.y * other, self.z * other)
        elif isinstance(other, (Vector3D, Vector3DImmutable)):
            return Vector3D(self.x * other.x, self.y * other.y, self.z * other.z)
        elif isinstance(other, (Point3D, Point3DImmutable)):
            return Point3D(self.x * other.x, self.y * other.y, self.z * other.z)
        else:
            raise TypeError('Cannot multiply {} and {}'.format(type(self), type(other)))

    __rmul__ = __mul__

    def __imul__(self, other):
        assert type(other) in (int, float), \
            'Cannot multiply types {} and {}'.format(type(self), type(other))
        self.x *= other
        self.y *= other
        self.z *= other
        return self

    def __div__(self, other):
        assert type(other) in (int, float), \
            'Cannot divide types {} and {}'.format(type(self), type(other))
        return Vector3D(operator.div(self.x, other),
                        operator.div(self.y, other),
                        operator.div(self.z, other))

    def __rdiv__(self, other):
        assert type(other) in (int, float), \
            'Cannot divide types {} and {}'.format(type(self), type(other))
        return Vector3D(operator.div(other, self.x),
                        operator.div(other, self.y),
                        operator.div(other, self.z))

    def __floordiv__(self, other):
        assert type(other) in (int, float), \
            'Cannot divide types {} and {}'.format(type(self), type(other))
        return Vector3D(operator.floordiv(self.x, other),
                        operator.floordiv(self.y, other),
                        operator.floordiv(self.z, other))

    def __rfloordiv__(self, other):
        assert type(other) in (int, float), \
            'Cannot divide types {} and {}'.format(type(self), type(other))
        return Vector3D(operator.floordiv(other, self.x),
                        operator.floordiv(other, self.y),
                        operator.floordiv(other, self.z))

    def __truediv__(self, other):
        assert type(other) in (int, float), \
            'Cannot divide types {} and {}'.format(type(self), type(other))
        return Vector3D(operator.truediv(self.x, other),
                        operator.truediv(self.y, other),
                        operator.truediv(self.z, other))

    def __rtruediv__(self, other):
        assert type(other) in (int, float), \
            'Cannot divide types {} and {}'.format(type(self), type(other))
        return Vector3D(operator.truediv(other, self.x),
                        operator.truediv(other, self.y),
                        operator.truediv(other, self.z))

    def __neg__(self):
        return Vector3D(-self.x, -self.y, -self.z)

    __pos__ = __copy__

    def __abs__(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __repr__(self):
        """Vector3D representation."""
        return 'Vector3D (%.2f, %.2f, %.2f)' % (self.x, self.y, self.z)


class Point3D(Vector3D):
    """3D Point object.

    Properties:
        x
        y
        z
    """

    def move(self, moving_vec):
        """Get a point that has been moved along a vector.

        Args:
            moving_vec: A Vector3D with the direction and distance to move the point.
        """
        return Point3D(self.x + moving_vec.x,
                       self.y + moving_vec.y,
                       self.z + moving_vec.z)

    def rotate(self, axis, angle, origin):
        """Rotate a point by a certain angle around an axis and origin.

        Right hand rule applies:
        If axis has a positive orientation, rotation will be clockwise.
        If axis has a negative orientation, rotation will be counterclockwise.

        Args:
            axis: A Vector3D axis representing the axis of rotation.
            angle: An angle for rotation in radians.
            origin: A Point3D for the origin around which the point will be rotated.
        """
        return Point3D._rotate(self - origin, axis, angle) + origin

    def rotate_xy(self, angle, origin):
        """Get a point rotated counterclockwise in the XY plane by a certain angle.

        Args:
            angle: An angle in radians.
            origin: A Point3D for the origin around which the point will be rotated.
        """
        trans_self = self - origin
        vec_2 = Vector2D._rotate(trans_self, angle)
        return Point3D(vec_2.x, vec_2.y, trans_self.z) + origin

    def reflect(self, normal, origin):
        """Get a point reflected across a plane with the input normal vector and origin.

        Args:
            normal: A Vector3D representing the normal vector for the plane across
                which the point will be reflected. THIS VECTOR MUST BE NORMALIZED.
            origin: A Point3D representing the origin from which to reflect.
        """
        return Point3D._reflect(self - origin, normal) + origin

    def scale(self, factor, origin):
        """Scale a point by a factor from an origin point.

        Args:
            factor: A number representing how much the point should be scaled.
            origin: A Point3D representing the origin from which to scale.
        """
        return (factor * (self - origin)) + origin

    def scale_world_origin(self, factor):
        """Scale a point by a factor from the world origin. Faster than Point3D.scale.

        Args:
            factor: A number representing how much the point should be scaled.
        """
        return Point3D(self.x * factor, self.y * factor, self.z * factor)

    def project(self, normal, origin):
        """Get a point projected a point3d into a plane with a given normal and origin.

        Args:
            normal: A Vector3D representing the normal vector of the plane into wich
                the plane will be projected. THIS VECTOR MUST BE NORMALIZED.
            origin: A Point3D representing the origin the plane into which the
                point will be projected.
        """
        trans_self = self - origin
        return self - normal * trans_self.dot(normal)

    def distance_to_point(self, point):
        """Get the distance from this point to another Point3D."""
        vec = (self.x - point.x, self.y - point.y, self.z - point.z)
        return math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)

    def to_immutable(self):
        """Get an immutable version of this object."""
        return Point3DImmutable(self.x, self.y, self.z)

    def __add__(self, other):
        # Point + Vector -> Point
        # Point + Point -> Vector
        if isinstance(other, (Vector3D, Vector3DImmutable)):
            return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)
        elif isinstance(other, (Point3D, Point3DImmutable)):
            return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
        else:
            raise TypeError('Cannot add Point3D and {}'.format(type(other)))

    def __sub__(self, other):
        # Point - Vector -> Point
        # Point - Point -> Vector
        if isinstance(other, (Vector3D, Vector3DImmutable)):
            return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)
        elif isinstance(other, (Point3D, Point3DImmutable)):
            Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
        else:
            raise TypeError('Cannot subtract Point3D and {}'.format(type(other)))

    def __repr__(self):
        """Point3D representation."""
        return 'Point3D (%.2f, %.2f, %.2f)' % (self.x, self.y, self.z)


@immutable
class Vector3DImmutable(Vector3D):
    """Immutable 3D Vector object."""
    _mutable = False

    def to_mutable(self):
        """Get a mutable version of this vector."""
        return Vector3D(self.x, self.y, self.z)

    def to_immutable(self):
        """Get an immutable version of this object."""
        return self


@immutable
class Point3DImmutable(Point3D):
    """Immutable 3D Point object."""
    _mutable = False

    def to_mutable(self):
        """Get a mutable version of this point."""
        return Point3D(self.x, self.y, self.z)

    def to_immutable(self):
        """Get an immutable version of this object."""
        return self
