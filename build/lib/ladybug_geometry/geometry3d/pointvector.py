# coding=utf-8
"""3D Vector and 3D Point"""
from __future__ import division

from ..geometry2d.pointvector import Vector2D

import math
import operator


class Vector3D(object):
    """3D Vector object.

    Properties:
        * x
        * y
        * z
        * magnitude
        * magnitude_squared
        * is_zero
    """
    __slots__ = ('_x', '_y', '_z')

    def __init__(self, x=0, y=0, z=0):
        """Initialize 3D Vector."""
        self._x = self._cast_to_float(x)
        self._y = self._cast_to_float(y)
        self._z = self._cast_to_float(z)

    @classmethod
    def from_dict(cls, data):
        """Create a Vector3D/Point3D from a dictionary.

        Args:
            data: A python dictionary in the following format

        .. code-block:: python

            {
            "x": 10,
            "y": 0,
            "z": 0
            }
        """
        return cls(data['x'], data['y'], data['z'])

    @classmethod
    def from_array(cls, array):
        """Initialize a Vector3D/Point3D from an array.

        Args:
            array: A tuple or list with three numbers representing the x, y and z
                values of the point.
        """
        return cls(array[0], array[1], array[2])

    @property
    def x(self):
        """Get the X coordinate."""
        return self._x

    @property
    def y(self):
        """Get the Y coordinate."""
        return self._y

    @property
    def z(self):
        """Get the Z coordinate."""
        return self._z

    @property
    def magnitude(self):
        """Get the magnitude of the vector."""
        return self.__abs__()

    @property
    def magnitude_squared(self):
        """Get the magnitude squared of the vector."""
        return self.x ** 2 + self.y ** 2 + self.z ** 2

    def is_zero(self, tolerance=0):
        """Boolean to note whether the vector is within a given zero tolerance.

        Args:
            tolerance: The tolerance below which the vector is considered to
                be a zero vector.
        """
        return abs(self.x) <= tolerance and abs(self.y) <= tolerance and \
            abs(self.z) <= tolerance

    def normalize(self):
        """Get a copy of the vector that is a unit vector (magnitude=1)."""
        d = self.magnitude
        try:
            return Vector3D(self.x / d, self.y / d, self.z / d)
        except ZeroDivisionError:
            return self.duplicate()

    def reverse(self):
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
        try:
            return math.acos(self.dot(other) / (self.magnitude * other.magnitude))
        except ValueError:  # python floating tolerance can cause math domain error
            if self.dot(other) < 0:
                return math.acos(-1)
            return math.acos(1)

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

    def duplicate(self):
        """Get a copy of this vector."""
        return self.__copy__()

    def to_dict(self):
        """Get Vector3D as a dictionary."""
        return {'type': 'Vector3D',
                'x': self.x,
                'y': self.y,
                'z': self.z}

    def to_array(self):
        """Get Vector3D/Point3D as a tuple of three numbers"""
        return (self.x, self.y, self.z)

    def _cast_to_float(self, value):
        """Ensure that an input coordinate value is a float."""
        try:
            number = float(value)
        except (ValueError, TypeError):
            raise TypeError(
                'Coordinates must be numbers. Got {}: {}.'.format(type(value), value))
        return number

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
        r2 = u ** 2 + v ** 2 + w ** 2
        r = math.sqrt(r2)
        ct = math.cos(angle)
        st = math.sin(angle) / r
        dt = (u * x + v * y + w * z) * (1 - ct) / r2
        return Vector3D((u * dt + x * ct + (-w * y + v * z) * st),
                        (v * dt + y * ct + (w * x - u * z) * st),
                        (w * dt + z * ct + (-v * x + u * y) * st))

    def __copy__(self):
        return self.__class__(self.x, self.y, self.z)
    
    def __key(self):
        """A tuple based on the object properties, useful for hashing."""
        return (self.x, self.y, self.z)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(other, (Vector3D, Point3D)) and \
            self.__key() == other.__key()

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
        if isinstance(other, Point3D):
            return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)
        elif isinstance(other, Vector3D):
            return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
        else:
            raise TypeError('Cannot add {} and {}'.format(
                self.__class__.__name__, type(other)))

    __radd__ = __add__

    def __sub__(self, other):
        # Vector - Point -> Point
        # Vector - Vector -> Vector
        if isinstance(other, Point3D):
            return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)
        elif isinstance(other, Vector3D):
            return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
        else:
            raise TypeError('Cannot subtract {} and {}'.format(
                self.__class__.__name__, type(other)))

    def __rsub__(self, other):
        if isinstance(other, (Vector3D, Point3D)):
            return Vector3D(other.x - self.x, other.y - self.y, other.z - self.z)
        else:
            assert hasattr(other, '__len__') and len(other) == 3, \
                'Cannot subtract types {} and {}'.format(
                    self.__class__.__name__, type(other))
            return Vector3D(other.x - self[0], other.y - self[1], other.z - self[2])

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vector3D(self.x * other, self.y * other, self.z * other)
        elif isinstance(other, Vector3D):
            return Vector3D(self.x * other.x, self.y * other.y, self.z * other.z)
        elif isinstance(other, Point3D):
            return Point3D(self.x * other.x, self.y * other.y, self.z * other.z)
        else:
            raise TypeError('Cannot multiply {} and {}'.format(
                self.__class__.__name__, type(other)))

    __rmul__ = __mul__

    def __div__(self, other):
        assert type(other) in (int, float), \
            'Cannot divide types {} and {}'.format(self.__class__.__name__, type(other))
        return Vector3D(self.x / other, self.y / other, self.z / other)

    def __rdiv__(self, other):
        assert type(other) in (int, float), \
            'Cannot divide types {} and {}'.format(self.__class__.__name__, type(other))
        return Vector3D(other / self.x, other / self.y, other / self.z)

    def __floordiv__(self, other):
        assert type(other) in (int, float), \
            'Cannot divide types {} and {}'.format(self.__class__.__name__, type(other))
        return Vector3D(operator.floordiv(self.x, other),
                        operator.floordiv(self.y, other),
                        operator.floordiv(self.z, other))

    def __rfloordiv__(self, other):
        assert type(other) in (int, float), \
            'Cannot divide types {} and {}'.format(self.__class__.__name__, type(other))
        return Vector3D(operator.floordiv(other, self.x),
                        operator.floordiv(other, self.y),
                        operator.floordiv(other, self.z))

    def __truediv__(self, other):
        assert type(other) in (int, float), \
            'Cannot divide types {} and {}'.format(self.__class__.__name__, type(other))
        return Vector3D(operator.truediv(self.x, other),
                        operator.truediv(self.y, other),
                        operator.truediv(self.z, other))

    def __rtruediv__(self, other):
        assert type(other) in (int, float), \
            'Cannot divide types {} and {}'.format(self.__class__.__name__, type(other))
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
        * x
        * y
        * z
    """
    __slots__ = ()

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

    def scale(self, factor, origin=None):
        """Scale a point by a factor from an origin point.

        Args:
            factor: A number representing how much the point should be scaled.
            origin: A Point3D representing the origin from which to scale.
                If None, it will be scaled from the World origin (0, 0, 0).
        """
        if origin is None:
            return Point3D(self.x * factor, self.y * factor, self.z * factor)
        else:
            return (factor * (self - origin)) + origin

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

    def is_equivalent(self, point, tolerance):
        """Test whether this point is equivalent to another within a certain tolerance.

        Note that if you want to test whether the coordinate values are perfectly
        equal to one another, the == operator can be used.

        Args:
            point: Another Point3D for which geometric equivalency will be tested.
            tolerance: The minimum difference between the coordinate values of two
                points at which they can be considered geometrically equivalent.
        Returns:
            True if equivalent.  False if not equivalent.
        """
        return abs(self.x - point.x) <= tolerance and \
            abs(self.y - point.y) <= tolerance and \
            abs(self.z - point.z) <= tolerance

    def to_dict(self):
        """Get Point3D as a dictionary."""
        return {'type': 'Point3D',
                'x': self.x,
                'y': self.y,
                'z': self.z}

    def __add__(self, other):
        # Point + Vector -> Point
        # Point + Point -> Vector
        if isinstance(other, Vector3D):
            return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)
        elif isinstance(other, Point3D):
            return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
        else:
            raise TypeError('Cannot add Point3D and {}'.format(type(other)))

    def __sub__(self, other):
        # Point - Vector -> Point
        # Point - Point -> Vector
        if isinstance(other, Vector3D):
            return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)
        elif isinstance(other, Point3D):
            Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
        else:
            raise TypeError('Cannot subtract Point3D and {}'.format(type(other)))

    def __repr__(self):
        """Point3D representation."""
        return 'Point3D (%.2f, %.2f, %.2f)' % (self.x, self.y, self.z)
