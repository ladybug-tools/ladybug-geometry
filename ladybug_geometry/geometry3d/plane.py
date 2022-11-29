# coding=utf-8
"""Plane"""
from __future__ import division

from .pointvector import Point3D, Vector3D
from .ray import Ray3D
from ..intersection3d import intersect_line3d_plane, intersect_plane_plane, \
    closest_point3d_on_plane, closest_point3d_between_line3d_plane
from ..geometry2d.pointvector import Point2D, Vector2D
from ..geometry2d.ray import Ray2D

import math


class Plane(object):
    """Plane object.

    Args:
        n: A Vector3D representing the normal of the plane.
        o: A Point3D representing the origin point of the plane.
        x: An optional Vector3D for the X-Axis of the Plane.
            Note that this vector must be orthogonal to the input normal vector.
            If None, the default will find an X-Axis in the world XY plane.

    Properties:
        * n
        * o
        * k
        * x
        * y
        * altitude
        * azimuth
        * min
        * max
    """
    __slots__ = ('_n', '_o', '_k', '_x', '_y', '_altitude', '_azimuth')

    def __init__(self, n=Vector3D(0, 0, 1), o=Point3D(0, 0, 0), x=None):
        """Initialize Plane."""
        assert isinstance(n, Vector3D), \
            "Expected Vector3D for plane normal. Got {}.".format(type(n))
        assert isinstance(o, Point3D), \
            "Expected Point3D for plane origin. Got {}.".format(type(o))
        self._n = n.normalize()
        self._o = o
        self._k = self._n.dot(self._o)

        if x is None:
            if self._n.x == 0 and self._n.y == 0:
                self._x = Vector3D(1, 0, 0)
            else:
                x = Vector3D(self._n.y, -self._n.x, 0)
                x = x.normalize()
                self._x = x
        else:
            assert isinstance(x, Vector3D), \
                "Expected Vector3D for plane X-axis. Got {}.".format(type(x))
            x = x.normalize()
            assert abs(self._n.x * x.x + self._n.y * x.y + self._n.z * x.z) < 1e-2, \
                'Plane X-axis and normal vector are not orthogonal. Got angle of {} ' \
                'degrees between them.'.format(math.degrees(self._n.angle(x)))
            self._x = x
        self._y = self._n.cross(self._x)
        self._altitude = None
        self._azimuth = None

    @classmethod
    def from_dict(cls, data):
        """Create a Plane from a dictionary.

        .. code-block:: python

            {
                "type": "Plane"
                "n": (0, 0, 1),
                "o": (0, 10, 0),
                "x": (1, 0, 0)
            }
        """
        x = None
        if 'x' in data and data['x'] is not None:
            x = Vector3D.from_array(data['x'])
        return cls(Vector3D.from_array(data['n']),
                   Point3D.from_array(data['o']), x)

    @classmethod
    def from_three_points(cls, o, p2, p3):
        """Initialize a Plane from three Point3D objects that are not co-linear.

        Args:
            o: A Point3D representing the origin point of the plane.
            p2: A Point3D representing a point the plane.
            p3: A Point3D representing a point the plane.
        """
        return cls((p2 - o).cross(p3 - o), o)

    @classmethod
    def from_normal_k(cls, n, k):
        """Initialize a Plane from a normal vector and a scalar constant.

        Args:
            o: A Point3D representing the origin point of the plane.
            k: Scalar constant relating origin point to normal vector
        """
        # get an arbitrary point on the plane for the origin
        if n.z:
            o = Point3D(0., 0., k / n.z)
        elif n.y:
            o = Point3D(0., k / n.y, 0.)
        else:
            o = Point3D(k / n.x, 0., 0.)
        return cls(n, o)

    @property
    def n(self):
        """Normal vector. This vector will always be normalized (magnitude = 1)."""
        return self._n

    @property
    def o(self):
        """Origin point."""
        return self._o

    @property
    def k(self):
        """Scalar constant relating origin point to normal vector."""
        return self._k

    @property
    def x(self):
        """Plane X-Axis. This vector will always be normalized (magnitude = 1)."""
        return self._x

    @property
    def y(self):
        """Plane Y-Axis. This vector will always be normalized (magnitude = 1)."""
        return self._y

    @property
    def azimuth(self):
        """Get the azimuth of the plane (between 0 and 2 * Pi).

        This will be zero if the plane is perfectly horizontal.
        """
        if self._azimuth is None:
            try:
                n_vec = Vector2D(0, 1)
                self._azimuth = n_vec.angle_clockwise(Vector2D(self.n.x, self.n.y))
            except ZeroDivisionError:  # plane is perfectly horizontal
                self._azimuth = 0
        return self._azimuth

    @property
    def altitude(self):
        """Get the altitude of the plane (between Pi/2 and -Pi/2)."""
        if self._altitude is None:
            self._altitude = self.n.angle(Vector3D(0, 0, -1)) - math.pi / 2
        return self._altitude

    @property
    def min(self):
        """Returns the Plane origin."""
        return self._o

    @property
    def max(self):
        """Returns the Plane origin."""
        return self._o

    def flip(self):
        """Get a flipped version of this plane (facing the opposite direction)."""
        return Plane(self.n.reverse(), self.o, self.x)

    def move(self, moving_vec):
        """Get a plane that has been moved along a vector.

        Args:
            moving_vec: A Vector3D with the direction and distance to move the plane.
        """
        return Plane(self.n, self.o.move(moving_vec), self.x)

    def rotate(self, axis, angle, origin):
        """Rotate a plane by a certain angle around an axis and origin.

        Right hand rule applies:
        If axis has a positive orientation, rotation will be clockwise.
        If axis has a negative orientation, rotation will be counterclockwise.

        Args:
            axis: A Vector3D axis representing the axis of rotation.
            angle: An angle for rotation in radians.
            origin: A Point3D for the origin around which the object will be rotated.
        """
        return Plane(self.n.rotate(axis, angle),
                     self.o.rotate(axis, angle, origin),
                     self.x.rotate(axis, angle))

    def rotate_xy(self, angle, origin):
        """Get a plane rotated counterclockwise in the world XY plane by a certain angle.

        Args:
            angle: An angle in radians.
            origin: A Point3D for the origin around which the object will be rotated.
        """
        return Plane(self.n.rotate_xy(angle),
                     self.o.rotate_xy(angle, origin),
                     self.x.rotate_xy(angle))

    def reflect(self, normal, origin):
        """Get a plane reflected across a plane with the input normal vector and origin.

        Args:
            normal: A Vector3D representing the normal vector for the plane across
                which the plane will be reflected. THIS VECTOR MUST BE NORMALIZED.
            origin: A Point3D representing the origin from which to reflect.
        """
        return Plane(self.n.reflect(normal),
                     self.o.reflect(normal, origin),
                     self.x.reflect(normal))

    def scale(self, factor, origin=None):
        """Scale a plane by a factor from an origin point.

        Args:
            factor: A number representing how much the plane should be scaled.
            origin: A Point3D representing the origin from which to scale.
                If None, it will be scaled from the World origin (0, 0, 0).
        """
        return Plane(self.n, self.o.scale(factor, origin), self.x)

    def xyz_to_xy(self, point):
        """Get a Point2D in the coordinate system of this plane from a Point3D.

        Note that the input Point3D should lie within this plane object in order
        for the result to be valid.
        """
        _diff = Vector3D(point.x - self.o.x, point.y - self.o.y, point.z - self.o.z)
        return Point2D(self.x.dot(_diff), self.y.dot(_diff))

    def xy_to_xyz(self, point):
        """Get a Point3D from a Point2D in the coordinate system of this plane."""
        # This method returns the same result as the following code:
        # self.o + (self.x * point.x) + (self.y * point.y)
        # It has been written explicitly to cut out the isinstance() checks for speed
        _u = (self.x.x * point.x, self.x.y * point.x, self.x.z * point.x)
        _v = (self.y.x * point.y, self.y.y * point.y, self.y.z * point.y)
        return Point3D(
            self.o.x + _u[0] + _v[0], self.o.y + _u[1] + _v[1], self.o.z + _u[2] + _v[2])

    def is_point_above(self, point):
        """Test if a given point is above or below this plane.

        Above is defined as being on the side of the plane that the plane normal
        is pointing towards.

        Args:
            point: A Point3D object to test.

        Returns:
            True is point is above; False if below.
        """
        vec = Vector3D(point.x - self.o.x, point.y - self.o.y, point.z - self.o.z)
        return self.n.dot(vec) > 0

    def closest_point(self, point):
        """Get the closest Point3D on this plane to another Point3D.

        Args:
            point: A Point3D object to which the closest point on this plane
                will be computed.

        Returns:
            Point3D for the closest point on this plane to the input point.
        """
        return closest_point3d_on_plane(point, self)

    def distance_to_point(self, point):
        """Get the minimum distance between this plane and the input point.

        Args:
            point: A Point3D object to which the minimum distance will be computed.

        Returns:
            The distance to the input point.
        """
        close_pt = self.closest_point(point)
        return point.distance_to_point(close_pt)

    def closest_points_between_line(self, line_ray):
        """Get the two closest Point3D between this plane and a Line3D or Ray3D.

        Args:
            line_ray: A Line3D or Ray3D object to which the closest points
                will be computed.

        Returns:
            Two Point3D objects representing

            1) The closest point on the input line_ray to this plane.
            2) The closest point on this plane to the input line_ray.

            Will be None if the line_ray intersects this plant
        """
        return closest_point3d_between_line3d_plane(line_ray, self)

    def distance_to_line(self, line_ray):
        """Get the minimum distance between this plane and the input Line3D or Ray3D.

        Args:
            line_ray: A Line3D or Ray3D object to which the minimum distance
                will be computed.

        Returns:
            The minimum distance to the input line_ray.
        """
        result = self.closest_points_between_line(line_ray)
        if result is None:  # intersection
            return 0
        else:
            return result[0].distance_to_point(result[1])

    def intersect_line_ray(self, line_ray):
        """Get the intersection between this plane and the input Line3D or Ray3D.

        Args:
            line_ray: A Line3D or Ray3D object for which intersection will be computed.

        Returns:
            Point3D for the intersection. Will be None if no intersection exists.
        """
        return intersect_line3d_plane(line_ray, self)

    def intersect_arc(self, arc):
        """Get the intersection between this Plane and an Arc3D.

        Args:
            plane: A Plane object for which intersection will be computed.

        Returns:
            A list of 2 Point3D objects if a full intersection exists.
            A list with a single Point3D object if the line is tangent or intersects
            only once. None if no intersection exists.
        """
        _plane_int_ray = self.intersect_plane(arc.plane)
        if _plane_int_ray is not None:
            _p12d = arc.plane.xyz_to_xy(_plane_int_ray.p)
            _p22d = arc.plane.xyz_to_xy(_plane_int_ray.p + _plane_int_ray.v)
            _v2d = _p22d - _p12d
            _int_ray2d = Ray2D(_p12d, _v2d)
            _int_pt2d = arc.arc2d.intersect_line_infinite(_int_ray2d)
            if _int_pt2d is not None:
                return [arc.plane.xy_to_xyz(pt) for pt in _int_pt2d]
        return None

    def intersect_plane(self, plane):
        """Get the intersection between this Plane and another Plane.

        Args:
            plane: A Plane object for which intersection will be computed.

        Returns:
            Ray3D for the intersection. Will be None if planes are parallel.
        """
        result = intersect_plane_plane(self, plane)
        if result is not None:
            return Ray3D(result[0], result[1])
        return None

    def is_coplanar(self, plane):
        """Test if another Plane object is perfectly coplanar with this Plane.

        Args:
            plane: A Plane object for which co-planarity will be tested.

        Returns:
            True if plane is coplanar. False if it is not coplanar.
        """
        if self.n == plane.n:
            return self.k == plane.k
        elif self.n == plane.n.reverse():
            return self.k == -plane.k
        return False

    def is_coplanar_tolerance(self, plane, tolerance, angle_tolerance):
        """Test if another Plane object is coplanar within a certain tolerance.

        Args:
            plane: A Plane object for which co-planarity will be tested.
            tolerance: The distance between the two planes at which point they can
                be considered coplanar.
            angle_tolerance: The angle in radians that the plane normals can
                differ from one another in order for the planes to be considered
                coplanar.

        Returns:
            True if plane is coplanar. False if it is not coplanar.
        """
        if self.n.angle(plane.n) <= angle_tolerance or \
                self.n.angle(plane.n.reverse()) <= angle_tolerance:
            return self.distance_to_point(plane.o) <= tolerance
        return False

    def duplicate(self):
        """Get a copy of this object."""
        return self.__copy__()

    def to_dict(self):
        """Get Plane as a dictionary."""
        return {'type': 'Plane',
                'n': self.n.to_array(),
                'o': self.o.to_array(),
                'x': self.x.to_array()}

    def __copy__(self):
        return Plane(self.n, self.o, self.x)

    def __key(self):
        """A tuple based on the object properties, useful for hashing."""
        return (self.n, self.o, self.x)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(other, Plane) and self.__key() == other.__key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __repr__(self):
        return 'Plane (<%.2f, %.2f, %.2f> normal) (<%.2f, %.2f, %.2f> origin)' % \
            (self.n.x, self.n.y, self.n.z, self.o.x, self.o.y, self.o.z)
