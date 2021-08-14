# coding=utf-8
"""2D Vector and 2D Point"""
from __future__ import division

import math
import operator


class Vector2D(object):
    """2D Vector object.

    Args:
        x: Number for the X coordinate.
        y: Number for the Y coordinate.

    Properties:
        * x
        * y
        * magnitude
        * magnitude_squared
        * is_zero
    """
    __slots__ = ('_x', '_y')

    def __init__(self, x=0, y=0):
        """Initialize 2D Vector."""
        self._x = self._cast_to_float(x)
        self._y = self._cast_to_float(y)

    @classmethod
    def from_dict(cls, data):
        """Create a Vector2D/Point2D from a dictionary.

        Args:
            data: A python dictionary in the following format

        .. code-block:: python

            {
            "x": 10,
            "y": 0
            }
        """
        return cls(data['x'], data['y'])

    @classmethod
    def from_array(cls, array):
        """Initialize a Vector2D/Point2D from an array.

        Args:
            array: A tuple or list with two numbers representing the x and y
                values of the point.
        """
        return cls(array[0], array[1])

    def to_array(self):
        """Get Vector2D/Point2D as a tuple of two numbers"""
        return (self.x, self.y)

    @property
    def x(self):
        """Get the X coordinate."""
        return self._x

    @property
    def y(self):
        """Get the Y coordinate."""
        return self._y

    @property
    def magnitude(self):
        """Get the magnitude of the vector."""
        return self.__abs__()

    @property
    def magnitude_squared(self):
        """Get the magnitude squared of the vector."""
        return self.x ** 2 + self.y ** 2

    def is_zero(self, tolerance):
        """Boolean to note whether the vector is within a given zero tolerance.

        Args:
            tolerance: The tolerance below which the vector is considered to
                be a zero vector.
        """
        return abs(self.x) <= tolerance and abs(self.y) <= tolerance

    def is_equivalent(self, other, tolerance):
        """Test whether this object is equivalent to another within a certain tolerance.

        Note that if you want to test whether the coordinate values are perfectly
        equal to one another, the == operator can be used.

        Args:
            other: Another Point2D for which geometric equivalency will be tested.
            tolerance: The minimum difference between the coordinate values of two
                objects at which they can be considered geometrically equivalent.
        Returns:
            True if equivalent.  False if not equivalent.
        """
        return abs(self.x - other.x) <= tolerance and \
            abs(self.y - other.y) <= tolerance

    def normalize(self):
        """Get a copy of the vector that is a unit vector (magnitude=1)."""
        d = self.magnitude
        try:
            return Vector2D(self.x / d, self.y / d)
        except ZeroDivisionError:
            return self.duplicate()

    def reverse(self):
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
        try:
            return math.acos(self.dot(other) / (self.magnitude * other.magnitude))
        except ValueError:  # python floating tolerance can cause math domain error
            if self.dot(other) < 0:
                return math.acos(-1)
            return math.acos(1)

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

    def duplicate(self):
        """Get a copy of this vector."""
        return self.__copy__()

    def to_dict(self):
        """Get Vector2D as a dictionary."""
        return {'type': 'Vector2D',
                'x': self.x,
                'y': self.y}

    @staticmethod
    def circular_mean(angles):
        """Compute the circular mean across a list of angles in radians.

        If no circular mean exists, the normal mean will be returned.

        Args:
            angles: A list of angles in radians.
        """
        avg_x = sum(math.cos(ang) for ang in angles) / len(angles)
        avg_y = sum(math.sin(ang) for ang in angles) / len(angles)
        if (avg_x, avg_y) == (0, 0):  # just return the normal mean
            return sum(angles) / len(angles)
        return math.atan2(avg_y, avg_x)

    def _cast_to_float(self, value):
        """Ensure that an input coordinate value is a float."""
        try:
            number = float(value)
        except (ValueError, TypeError):
            raise TypeError(
                'Coordinates must be numbers. Got {}: {}.'.format(type(value), value))
        return number

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

    def __key(self):
        """A tuple based on the object properties, useful for hashing."""
        return (self.x, self.y)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(other, (Vector2D, Point2D)) and \
            self.__key() == other.__key()

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
        if isinstance(other, Point2D):
            return Point2D(self.x + other.x, self.y + other.y)
        elif isinstance(other, Vector2D):
            return Vector2D(self.x + other.x, self.y + other.y)
        else:
            raise TypeError('Cannot add {} and {}'.format(
                self.__class__.__name__, type(other)))

    __radd__ = __add__

    def __sub__(self, other):
        # Vector - Point -> Point
        # Vector - Vector -> Vector
        if isinstance(other, Point2D):
            return Point2D(self.x - other.x, self.y - other.y)
        elif isinstance(other, Vector2D):
            return Vector2D(self.x - other.x, self.y - other.y)
        else:
            raise TypeError('Cannot subtract {} and {}'.format(
                self.__class__.__name__, type(other)))

    def __rsub__(self, other):
        if isinstance(other, (Vector2D, Point2D)):
            return Vector2D(other.x - self.x, other.y - self.y)
        else:
            assert hasattr(other, '__len__') and len(other) == 2, \
                'Cannot subtract {} and {}'.format(
                    self.__class__.__name__, type(other))
            return Vector2D(other.x - self[0], other.y - self[1])

    def __mul__(self, other):
        assert type(other) in (int, float), \
            'Cannot multiply types {} and {}'.format(
                self.__class__.__name__, type(other))
        return Vector2D(self.x * other, self.y * other)

    __rmul__ = __mul__

    def __div__(self, other):
        assert type(other) in (int, float), \
            'Cannot divide types {} and {}'.format(self.__class__.__name__, type(other))
        return Vector2D(self.x / other, self.y / other)

    def __rdiv__(self, other):
        assert type(other) in (int, float), \
            'Cannot divide types {} and {}'.format(self.__class__.__name__, type(other))
        return Vector2D(other / self.x, other / self.y)

    def __floordiv__(self, other):
        assert type(other) in (int, float), \
            'Cannot divide types {} and {}'.format(self.__class__.__name__, type(other))
        return Vector2D(operator.floordiv(self.x, other),
                        operator.floordiv(self.y, other))

    def __rfloordiv__(self, other):
        assert type(other) in (int, float), \
            'Cannot divide types {} and {}'.format(self.__class__.__name__, type(other))
        return Vector2D(operator.floordiv(other, self.x),
                        operator.floordiv(other, self.y))

    def __truediv__(self, other):
        assert type(other) in (int, float), \
            'Cannot divide types {} and {}'.format(self.__class__.__name__, type(other))
        return Vector2D(operator.truediv(self.x, other),
                        operator.truediv(self.y, other))

    def __rtruediv__(self, other):
        assert type(other) in (int, float), \
            'Cannot divide types {} and {}'.format(self.__class__.__name__, type(other))
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

    Args:
        x: Number for the X coordinate.
        y: Number for the Y coordinate.

    Properties:
        * x
        * y
    """
    __slots__ = ()

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
        return Vector2D._rotate(self - origin, angle) + origin

    def reflect(self, normal, origin):
        """Get a point reflected across a plane with the input normal vector and origin.

        Args:
            normal: A Vector2D representing the normal vector for the plane across
                which the point will be reflected. THIS VECTOR MUST BE NORMALIZED.
            origin: A Point2D representing the origin from which to reflect.
        """
        return Vector2D._reflect(self - origin, normal) + origin

    def scale(self, factor, origin=None):
        """Scale a point by a factor from an origin point.

        Args:
            factor: A number representing how much the point should be scaled.
            origin: A Point2D representing the origin from which to scale.
                If None, it will be scaled from the World origin (0, 0).
        """
        if origin is None:
            return Point2D(self.x * factor, self.y * factor)
        else:
            return (factor * (self - origin)) + origin

    def distance_to_point(self, point):
        """Get the distance from this point to another Point2D."""
        vec = (self.x - point.x, self.y - point.y)
        return math.sqrt(vec[0] ** 2 + vec[1] ** 2)

    def to_dict(self):
        """Get Point2D as a dictionary."""
        return {'type': 'Point2D',
                'x': self.x,
                'y': self.y}

    def __add__(self, other):
        # Point + Vector -> Point
        # Point + Point -> Vector
        if isinstance(other, Point2D):
            return Vector2D(self.x + other.x, self.y + other.y)
        elif isinstance(other, Vector2D):
            return Point2D(self.x + other.x, self.y + other.y)
        else:
            raise TypeError('Cannot add Point2D and {}'.format(type(other)))

    def __sub__(self, other):
        # Point - Vector -> Point
        # Point - Point -> Vector
        if isinstance(other, Point2D):
            return Vector2D(self.x - other.x, self.y - other.y)
        elif isinstance(other, Vector2D):
            return Point2D(self.x - other.x, self.y - other.y)
        else:
            raise TypeError('Cannot subtract Point2D and {}'.format(type(other)))

    def __repr__(self):
        """Point2D representation."""
        return 'Point2D (%.2f, %.2f)' % (self.x, self.y)

    def __lt__(self, other):
        """ Lesser then inequality method. This is used by certain external
        data structure libraries to efficiently store and retrieve point data.
        """
        if isinstance(other, Vector2D):
            return self.x < other.x

    def __gt__(self, other):
        """ Greater then inequality method. This is used by certain external
        data structure libraries to efficiently store and retrieve point data.
        """
        if isinstance(other, Vector2D):
            return self.x > other.x
