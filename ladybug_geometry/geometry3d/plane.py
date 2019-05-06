# coding=utf-8
"""Plane"""
from __future__ import division

from .pointvector import Point3D, Point3DImmutable, Vector3D, Vector3DImmutable
from .ray import Ray3D
from ..intersection3d import intersect_line3d_plane, intersect_plane_plane, \
    closest_point3d_on_plane, closest_point3d_between_line3d_plane
from ..geometry2d.pointvector import Point2D, Point2DImmutable

import math


class Plane(object):
    """Plane object.

    Properties:
        n: Normal vector
        o: Origin point
        k: Scalar constant relating origin point to normal vector
        x: Plane X-Axis
        y: Plane Y-Axis
    """
    __slots__ = ('_n', '_o', '_k', '_x', '_y')

    def __init__(self, n=Vector3D(0, 0, 1), o=Point3D(0, 0, 0), x=None):
        """Initilize Plane.

        Args:
            n: A Vector3D representing the normal of the plane.
            o: A Point3D representing the origin point of the plane.
            x: An optional Vector3D for the X-Axis of the Plane.
                Note that this vector must be orthagonal to the input normal vector.
                If None, the default will find an X-Axis in the world XY plane.
        """
        assert isinstance(n, (Vector3D, Vector3DImmutable)), \
            "Expected Vector3D for plane normal. Got {}.".format(type(n))
        assert isinstance(o, (Point3D, Point3DImmutable)), \
            "Expected Point3D for plane origin. Got {}.".format(type(o))
        n = n.normalized()
        self._n = n.to_immutable()
        self._o = o.to_immutable()
        self._k = n.dot(o)

        if x is None:
            if n.x == 0 and n.y == 0:
                self._x = Vector3DImmutable(1, 0, 0)
            else:
                self._x = Vector3DImmutable(n.y, -n.x, 0)
        else:
            assert isinstance(x, (Vector3D, Vector3DImmutable)), \
                "Expected Vector3D for plane X-axis. Got {}.".format(type(x))
            x = x.normalized()
            assert abs(n.x * x.x + n.y * x.y + n.z * x.z) < 1e-9, \
                'Plane X-axis and normal vector are not orthagonal. Got angle of {} ' \
                'degrees between them.'.format(math.degrees(n.angle(x)))
            self._x = x.to_immutable()
        y = n.cross(self._x)
        self._y = y.to_immutable()

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

    def flip(self):
        """Get a flipped version of this plane (facing the opposite direction)."""
        return Plane(self.n.reversed(), self.o, self.x.reversed())

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

    def scale(self, factor, origin):
        """Scale a plane by a factor from an origin point.

        Args:
            factor: A number representing how much the plane should be scaled.
            origin: A Point3D representing the origin from which to scale.
        """
        return Plane(self.n, self.o.scale(factor, origin), self.x)

    def scale_world_origin(self, factor):
        """Scale a Plane by a factor from the world origin. Faster than Plane.scale.

        Args:
            factor: A number representing how much the line segment should be scaled.
        """
        return Plane(self.n, self.o.scale_world_origin(factor), self.x)

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
        # It has been wirtten explicitly to cut out the isinstance() calls for speed
        _u = (self.x.x * point.x, self.x.y * point.x, self.x.z * point.x)
        _v = (self.y.x * point.y, self.y.y * point.y, self.y.z * point.y)
        return Point3D(
            self.o.x + _u[0] + _v[0], self.o.y + _u[1] + _v[1], self.o.z + _u[2] + _v[2])

    def xyz_to_xy_immutable(self, point):
        """Get a Point2DImmutable in the coordinate system of this plane from a Point3D.
        """
        _diff = Vector3D(point.x - self.o.x, point.y - self.o.y, point.z - self.o.z)
        return Point2DImmutable(self.x.dot(_diff), self.y.dot(_diff))

    def xy_to_xyz_immutable(self, point):
        """Get a Point3DImmutable from a Point2D in the coordinate system of this plane.
        """
        _u = (self.x.x * point.x, self.x.y * point.x, self.x.z * point.x)
        _v = (self.y.x * point.y, self.y.y * point.y, self.y.z * point.y)
        return Point3DImmutable(
            self.o.x + _u[0] + _v[0], self.o.y + _u[1] + _v[1], self.o.z + _u[2] + _v[2])

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
            Two Point3D objects representing:
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
            plane: A Plane object for which coplanartiy will be tested.

        Returns:
            True if plane is coplanar. False if it is not coplanar.
        """
        if self.n == plane.n:
            return self.k == plane.k
        elif self.n == plane.n.reversed():
            return self.k == -plane.k
        return False

    def is_coplanar_tolerance(self, plane, tolerance, angle_tolerance):
        """Test if another Plane object is coplanar within a certain tolerance.

        Args:
            plane: A Plane object for which coplanartiy will be tested.
            tolerance: The distance between the two planes at which point they can
                be considered coplanar.
            angle_tolerance: The angle in radians that the plane normals can
                differ from one another in order for the planes to be considered
                coplanar.

        Returns:
            True if plane is coplanar. False if it is not coplanar.
        """
        if self.n.angle(plane.n) < angle_tolerance or \
                self.n.angle(plane.n.reversed()) < angle_tolerance:
            return self.distance_to_point(plane.o) < tolerance
        return False

    def duplicate(self):
        """Get a copy of this object."""
        return self.__copy__()

    def __copy__(self):
        return self.__class__(self.n, self.o)

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __repr__(self):
        return 'Plane (<%.2f, %.2f, %.2f> normal) (<%.2f, %.2f, %.2f> origin)' % \
            (self.n.x, self.n.y, self.n.z, self.o.x, self.o.y, self.o.z)
