# coding=utf-8
"""3D Arc"""
from __future__ import division

from .plane import Plane

from ..geometry2d.pointvector import Point2D, Vector2D
from ..geometry2d.ray import Ray2D
from ..geometry2d.arc import Arc2D

import math


class Arc3D(object):
    """3D arc object.

    Args:
        plane: A Plane in which the arc lies with an origin representing the
            center of the circle for the arc.
        radius: A number representing the radius of the arc.
        a1: A number between 0 and 2 * pi for the start angle of the arc.
        a2: A number between 0 and 2 * pi for the end angle of the arc.

    Properties:
        * plane
        * radius
        * a1
        * a2
        * p1
        * p2
        * midpoint
        * c
        * length
        * angle
        * is_circle
        * arc2d
    """
    __slots__ = ('_plane', '_arc2d')

    def __init__(self, plane, radius, a1=0, a2=2*math.pi):
        """Initilize Arc3D.
        """
        assert isinstance(plane, Plane), "Expected Plane. Got {}.".format(type(plane))
        self._plane = plane
        self._arc2d = Arc2D(Point2D(0, 0), radius, a1, a2)

    @classmethod
    def from_dict(cls, data):
        """Create a Arc3D from a dictionary.

        Args:
            data: A python dictionary in the following format

        .. code-block:: python

            {
            "type": "Arc3D"
            "plane": {"n": (0, 0, 1), "o": (0, 10, 0), "x": (1, 0, 0)},
            "radius": 5,
            "a1": 0,
            "a2": 3.14159
            }
        """
        return cls(Plane.from_dict(data['plane']), data['radius'],
                   data['a1'], data['a2'])

    @classmethod
    def from_start_mid_end(cls, p1, m, p2, circle=False):
        """Initialize a new arc from start, middle, and end points.

        Note that input points will be assumed to be in counterclockwise order.

        Args:
            p1: The start point of the arc.
            m: Any point along the length of the arc that is not the start or end.
            p2: The end point of the arc.
            circle: Set to True if you would like the output to be a full circle
                defined by the three points instead of an arc with a start and end.
                Default is False.
        """
        plane = cls._plane_from_vertices(p1, m, p2)
        p1_2d, m_2d, p2_2d = plane.xyz_to_xy(p1), plane.xyz_to_xy(m), plane.xyz_to_xy(p2)
        arc_2d = Arc2D.from_start_mid_end(p1_2d, m_2d, p2_2d, circle)
        return cls(Plane(plane.n, plane.xy_to_xyz(arc_2d.c), plane.x),
                   arc_2d.r, arc_2d.a1, arc_2d.a2)

    @property
    def plane(self):
        """A Plane in which the arc lies with an origin for the center of the arc."""
        return self._plane

    @property
    def radius(self):
        """Radius of arc."""
        return self._arc2d.r

    @property
    def a1(self):
        """Start angle of the arc in radians."""
        return self._arc2d.a1

    @property
    def a2(self):
        """End angle of the arc in radians."""
        return self._arc2d.a2

    @property
    def p1(self):
        """Start point."""
        return self.plane.xy_to_xyz(self.arc2d.p1)

    @property
    def p2(self):
        """End point."""
        return self.plane.xy_to_xyz(self.arc2d.p2)

    @property
    def midpoint(self):
        """Midpoint."""
        return self.point_at(0.5)

    @property
    def c(self):
        """Center point of the circle on which the arc lies."""
        return self.plane.o

    @property
    def length(self):
        """The length of the arc."""
        return self.angle * self.radius

    @property
    def angle(self):
        """The total angle over the domain of the arc in radians."""
        _diff = self.a2 - self.a1
        return _diff if not self.is_inverted else 2 * math.pi + _diff

    @property
    def area(self):
        """Area of the circle to which the arc belongs."""
        assert self.is_circle, 'Arc must be a closed circle to access "area" property.'
        return math.pi * self.radius ** 2

    @property
    def is_circle(self):
        """Boolean for whether the arc is a full circle (True) or not (False)."""
        return self.a1 == 0 and self.a2 == 2 * math.pi

    @property
    def is_inverted(self):
        """Boolean noting whether the end angle a2 is smaller than the start angle a1."""
        return self.a2 < self.a1

    @property
    def arc2d(self):
        """An Arc2D within the plane of the Arc3D."""
        return self._arc2d

    def move(self, moving_vec):
        """Get an arc that has been moved along a vector.

        Args:
            moving_vec: A Vector3D with the direction and distance to move the arc.
        """
        return Arc3D(self.plane.move(moving_vec), self.radius, self.a1, self.a2)

    def rotate(self, axis, angle, origin):
        """Rotate this arc by a certain angle around an axis and origin.

        Right hand rule applies:
        If axis has a positive orientation, rotation will be clockwise.
        If axis has a negative orientation, rotation will be counterclockwise.

        Args:
            axis: A Vector3D axis representing the axis of rotation.
            angle: An angle for rotation in radians.
            origin: A Point3D for the origin around which the object will be rotated.
        """
        return Arc3D(self.plane.rotate(axis, angle, origin),
                     self.radius, self.a1, self.a2)

    def rotate_xy(self, angle, origin):
        """Get a arc that is rotated counterclockwise in the world XY plane by a certain angle.

        Args:
            angle: An angle for rotation in radians.
            origin: A Point3D for the origin around which the arc will be rotated.
        """
        return Arc3D(self.plane.rotate_xy(angle, origin), self.radius, self.a1, self.a2)

    def reflect(self, normal, origin):
        """Get a arc reflected across a plane with the input normal vector and origin.

        Args:
            normal: A Vector3D representing the normal vector for the plane across
                which the arc will be reflected. THIS VECTOR MUST BE NORMALIZED.
            origin: A Point3D representing the origin from which to reflect.
        """
        arc2d = self.arc2d.reflect(Vector2D(0, 1), Point2D(0, 0))
        return Arc3D(self.plane.reflect(normal, origin), self.radius, arc2d.a1, arc2d.a2)

    def scale(self, factor, origin=None):
        """Scale a arc by a factor from an origin point.

        Args:
            factor: A number representing how much the arc should be scaled.
            origin: A Point2D representing the origin from which to scale.
                If None, it will be scaled from the World origin (0, 0, 0).
        """
        return Arc3D(self.plane.scale(factor, origin), self.radius * factor,
                     self.a1, self.a2)

    def subdivide(self, distances):
        """Get Point3D values along the arc that subdivide it based on input distances.

        Args:
            distances: A list of distances along the arc at which to subdivide it.
                This can also be a single number that will be repeated unitl the end of
                the arc.
        """
        return [self.plane.xy_to_xyz(pt) for pt in self.arc2d.subdivide(distances)]

    def subdivide_evenly(self, number):
        """Get Point3D values along the arc that divide it into evenly-spaced segments.

        Args:
            number: The number of segments into which the arc will be divided.
        """
        return [self.plane.xy_to_xyz(pt) for pt in self.arc2d.subdivide_evenly(number)]

    def point_at(self, parameter):
        """Get a point at a given fraction along the arc.

        Args:
            parameter: The fraction between the start and end point where the
                desired point lies. For example, 0.5 will yield the midpoint.
        """
        return self.plane.xy_to_xyz(self.arc2d.point_at(parameter))

    def point_at_angle(self, angle):
        """Get a point at a given angle along the arc.

        Args:
            angle: The angle in radians from the start point along the arc
                to get the Point3D.
        """
        return self.plane.xy_to_xyz(self.arc2d.point_at_angle(angle))

    def point_at_length(self, length):
        """Get a point at a given distance along the arc segment.

        Args:
            length: The distance along the arc from the start point where the
                desired point lies.
        """
        return self.point_at(length / self.length)

    def closest_point(self, point):
        """Get the closest Point3D on this object to another Point3D.

        Args:
            point: A Point3D object to which the closest point on this object
                will be computed.

        Returns:
            Point3D for the closest point on this line to the input point.
        """
        plane_pt = self.plane.closest_point(point)
        return self.plane.xy_to_xyz(
            self.arc2d.closest_point(self.plane.xyz_to_xy(plane_pt)))

    def distance_to_point(self, point):
        """Get the minimum distance between this object and the input point.

        Args:
            point: A Point3D object to which the minimum distance will be computed.

        Returns:
            The distance to the input point.
        """
        close_pt = self.closest_point(point)
        return point.distance_to_point(close_pt)

    def split_with_plane(self, plane):
        """Split this Arc3D in 2 or 3 smaller arcs using a Plane.

        Args:
            plane: A Plane that will be used to split this arc.

        Returns:
            A list with two Arc3D objects if the split was successful.
            None if no intersection exists.
        """
        _plane_int_ray = plane.intersect_plane(self.plane)
        if _plane_int_ray is not None:
            _p12d = self.plane.xyz_to_xy(_plane_int_ray.p)
            _p22d = self.plane.xyz_to_xy(_plane_int_ray.p + _plane_int_ray.v)
            _v2d = _p22d - _p12d
            _int_ray2d = Ray2D(_p12d, _v2d)
            _int_pt2d = self.arc2d.split_line_infinite(_int_ray2d)
            if _int_pt2d is not None:
                return [Arc3D(self.plane, self.radius, arc.a1, arc.a2)
                        for arc in _int_pt2d]
        return None

    def duplicate(self):
        """Get a copy of this object."""
        return self.__copy__()

    def to_dict(self):
        """Get Arc3D as a dictionary."""
        return {'type': 'Arc3D', 'plane': self.plane.to_dict(),
                'radius': self.radius, 'a1': self.a1, 'a2': self.a2}

    @staticmethod
    def _plane_from_vertices(pt1, pt2, pt3):
        """Get a plane from three vertices."""
        try:
            v1 = pt2 - pt1
            v2 = pt3 - pt1
            n = v1.cross(v2)
        except Exception as e:
            raise ValueError('Incorrect Point3D input for Arc3D:\n\t{}'.format(e))
        return Plane(n, pt1)

    def __copy__(self):
        return Arc3D(self.plane, self.radius, self.a1, self.a2)

    def __key(self):
        """A tuple based on the object properties, useful for hashing."""
        return (hash(self.plane), self.radius, self.a1, self.a2)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(other, Arc3D) and self.__key() == other.__key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __repr__(self):
        return 'Arc3D (center {}) (radius {}) (length {})'.format(
            self.plane.o, self.radius, self.length)
