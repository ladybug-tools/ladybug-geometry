# coding=utf-8
"""2D Arc"""
from __future__ import division

from .pointvector import Point2D, Vector2D
from ..intersection2d import closest_point2d_on_arc2d, intersect_line2d_arc2d, \
    intersect_line2d_infinite_arc2d

import math


class Arc2D(object):
    """2D arc object.

    Args:
        c: A Point2D representing the center of the arc.
        r: A number representing the radius of the arc.
        a1: A number between 0 and 2 * pi for the start angle of the arc.
        a2: A number between 0 and 2 * pi for the end angle of the arc.

    Properties:
        * c
        * r
        * a1
        * a2
        * p1
        * p2
        * midpoint
        * length
        * angle
        * is_circle
    """
    __slots__ = ('_c', '_r', '_a1', '_a2', '_cos_a1', '_sin_a1', '_cos_a2', '_sin_a2')

    def __init__(self, c, r, a1=0, a2=2*math.pi):
        """Initilize Arc2D.
        """
        assert isinstance(c, Point2D), "Expected Point2D. Got {}.".format(type(c))
        assert r > 0, 'Arc radius must be greater than 0. Got {}.'.format(r)
        assert 0 <= a1 <= 2 * math.pi, 'Arc start angle must be between 0 and 2*pi. ' \
            'Got {}.'.format(a1)
        assert 0 <= a2 <= 2 * math.pi, 'Arc start angle must be between 0 and 2*pi. ' \
            'Got {}.'.format(a2)
        self._c = c
        self._r = r
        self._a1 = a1
        self._a2 = a2
        self._cos_a1 = math.cos(a1)
        self._sin_a1 = math.sin(a1)
        self._cos_a2 = math.cos(a2)
        self._sin_a2 = math.sin(a2)

    @classmethod
    def from_dict(cls, data):
        """Create a Arc2D from a dictionary.

        Args:
            data: A python dictionary in the following format

        .. code-block:: python

            {
            "type": "Arc2D"
            "c": (10, 0),
            "r": 5,
            "a1": 0,
            "a2": 3.14159
            }
        """
        return cls(Point2D.from_array(data['c']),
                   data['r'], data['a1'], data['a2'])

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
        for pt in (p1, m, p2):
            assert isinstance(pt, Point2D), "Expected Point2D. Got {}.".format(type(pt))
        e1 = (p1.x ** 2 + p1.y ** 2)
        e2 = (m.x ** 2 + m.y ** 2)
        e3 = (p2.x ** 2 + p2.y ** 2)
        den = 2 * (p1.x * (m.y - p2.y) - p1.y * (m.x - p2.x) + m.x * p2.y - p2.x * m.y)
        try:
            x = -(e1 * (p2.y - m.y) + e2 * (p1.y - p2.y) + e3 * (m.y - p1.y)) / den
            y = -(e1 * (m.x - p2.x) + e2 * (p2.x - p1.x) + e3 * (p1.x - m.x)) / den
        except ZeroDivisionError:
            raise ValueError('Input points {}, {}, {} are colinear and '
                             'cannot define an arc.'.format(p1, m, p2))
        r = math.sqrt((x - p1.x) ** 2 + (y - p1.y) ** 2)
        if circle is True:
            return cls(Point2D(x, y), r)
        else:
            a1 = Vector2D(1, 0).angle_counterclockwise(Vector2D(p1.x - x, p1.y - y))
            a2 = Vector2D(1, 0).angle_counterclockwise(Vector2D(p2.x - x, p2.y - y))
            return cls(Point2D(x, y), r, a1, a2)

    @property
    def c(self):
        """Center point of the circle on which the arc lies."""
        return self._c

    @property
    def r(self):
        """Radius of arc."""
        return self._r

    @property
    def a1(self):
        """Start angle of the arc in radians."""
        return self._a1

    @property
    def a2(self):
        """End angle of the arc in radians."""
        return self._a2

    @property
    def p1(self):
        """Start point."""
        return Point2D(
            self.c.x + self._cos_a1 * self.r, self.c.y + self._sin_a1 * self.r)

    @property
    def p2(self):
        """End point."""
        return Point2D(
            self.c.x + self._cos_a2 * self.r, self.c.y + self._sin_a2 * self.r)

    @property
    def midpoint(self):
        """Midpoint."""
        return self.point_at(0.5)

    @property
    def length(self):
        """The length of the arc."""
        return self.angle * self.r

    @property
    def angle(self):
        """The total angle over the domain of the arc in radians."""
        _diff = self._a2 - self._a1
        return _diff if not self.is_inverted else 2 * math.pi + _diff

    @property
    def area(self):
        """Area of the circle to which the arc belongs."""
        assert self.is_circle, 'Arc must be a closed circle to access "area" property.'
        return math.pi * self.r ** 2

    @property
    def is_circle(self):
        """Boolean for whether the arc is a full circle (True) or not (False)."""
        return self.a1 == 0 and self.a2 == 2 * math.pi

    @property
    def is_inverted(self):
        """Boolean noting whether the end angle a2 is smaller than the start angle a1."""
        return self._a2 < self._a1

    def move(self, moving_vec):
        """Get an arc that has been moved along a vector.

        Args:
            moving_vec: A Vector2D with the direction and distance to move the arc.
        """
        return Arc2D(self.c.move(moving_vec), self.r, self.a1, self.a2)

    def rotate(self, angle, origin):
        """Get a arc that is rotated counterclockwise by a certain angle.

        Args:
            angle: An angle for rotation in radians.
            origin: A Point2D for the origin around which the arc will
                be rotated.
        """
        _a1 = self.a1 + angle
        _a2 = self.a2 + angle
        _a1 = _a1 - 2 * math.pi if _a1 > 2 * math.pi else _a1
        _a2 = _a2 - 2 * math.pi if _a2 > 2 * math.pi else _a2
        return Arc2D(self.c.rotate(angle, origin), self.r, _a1, _a2)

    def reflect(self, normal, origin):
        """Get a arc reflected across a plane with the input normal vector and origin.

        Args:
            normal: A Vector2D representing the normal vector for the plane across
                which the arc will be reflected. THIS VECTOR MUST BE NORMALIZED.
            origin: A Point2D representing the origin from which to reflect.
        """
        return Arc2D.from_start_mid_end(self.p2.reflect(normal, origin),
                                        self.midpoint.reflect(normal, origin),
                                        self.p1.reflect(normal, origin))

    def scale(self, factor, origin=None):
        """Scale a arc by a factor from an origin point.

        Args:
            factor: A number representing how much the arc should be scaled.
            origin: A Point2D representing the origin from which to scale.
                If None, it will be scaled from the World origin (0, 0).
        """
        return Arc2D(self.c.scale(factor, origin), self.r * factor, self.a1, self.a2)

    def subdivide(self, distances):
        """Get Point2D values along the arc that subdivide it based on input distances.

        Args:
            distances: A list of distances along the arc at which to subdivide it.
                This can also be a single number that will be repeated unitl the end of
                the arc.
        """
        if isinstance(distances, (float, int)):
            distances = [distances]
        arc_length = self.length
        dist = distances[0]
        index = 0
        sub_pts = [self.p1]
        while dist < arc_length:
            sub_pts.append(self.point_at_length(dist))
            if index < len(distances) - 1:
                index += 1
            dist += distances[index]
        sub_pts.append(self.p2)
        return sub_pts

    def subdivide_evenly(self, number):
        """Get Point2D values along the arc that divide it into evenly-spaced segments.

        Args:
            number: The number of segments into which the arc will be divided.
        """
        interval = 1 / number
        parameter = interval
        sub_pts = [self.p1]
        while parameter < 1:
            sub_pts.append(self.point_at(parameter))
            parameter += interval
        sub_pts.append(self.p2)
        return sub_pts

    def point_at(self, parameter):
        """Get a point at a given fraction along the arc.

        Args:
            parameter: The fraction between the start and end point where the
                desired point lies. For example, 0.5 will yield the midpoint.
        """
        _ang = self._a1 + self.angle * parameter
        _ang = _ang if _ang <= math.pi * 2 else _ang - math.pi * 2
        return Point2D(
            self.c.x + math.cos(_ang) * self.r, self.c.y + math.sin(_ang) * self.r)

    def point_at_angle(self, angle):
        """Get a point at a given angle along the arc.

        Args:
            angle: The angle in radians from the start point along the arc
                to get the Point2D.
        """
        _ang = self._a1 + angle
        _ang = _ang if _ang <= math.pi * 2 else _ang - math.pi * 2
        return Point2D(
            self.c.x + math.cos(_ang) * self.r, self.c.y + math.sin(_ang) * self.r)

    def point_at_length(self, length):
        """Get a point at a given distance along the arc segment.

        Args:
            length: The distance along the arc from the start point where the
                desired point lies.
        """
        return self.point_at(length / self.length)

    def closest_point(self, point):
        """Get the closest Point2D on this object to another Point2D.

        Args:
            point: A Point2D object to which the closest point on this object
                will be computed.

        Returns:
            Point2D for the closest point on this line to the input point.
        """
        return closest_point2d_on_arc2d(point, self)

    def distance_to_point(self, point):
        """Get the minimum distance between this object and the input point.

        Args:
            point: A Point2D object to which the minimum distance will be computed.

        Returns:
            The distance to the input point.
        """
        close_pt = self.closest_point(point)
        return point.distance_to_point(close_pt)

    def intersect_line_ray(self, line_ray):
        """Get the intersection between this Arc2D and another Ray2 or LineSegment2D.

        Args:
            line_ray: Another LineSegment2D or Ray2D or to intersect.

        Returns:
            A list of 2 Point2D objects if a full intersection exists.
            A list with a single Point2D object if the line is tangent or intersects
            only once. None if no intersection exists.
        """
        return intersect_line2d_arc2d(line_ray, self)

    def intersect_line_infinite(self, line_ray):
        """Get the intersection between this Arc2D and an infinitely extending Ray2D .

        Args:
            line_ray: Another LineSegment2D or Ray2D or to intersect.

        Returns:
            A list of 2 Point2D objects if a full intersection exists.
            A list with a single Point2D object if the line is tangent or intersects
            only once. None if no intersection exists.
        """
        return intersect_line2d_infinite_arc2d(line_ray, self)

    def split_line_infinite(self, line_ray):
        """Split this Arc2D in 2-3 using an infinitely extending Ray2D or LineSegment2D.

        Args:
            line_ray: A LineSegment2D or Ray2D that will be extended infinitely for
                intersection.

        Returns:
            A list with 2 or 3 Arc2D objects if the split was successful.
            None if no intersection exists.
        """
        inters = intersect_line2d_infinite_arc2d(line_ray, self)
        if inters is None:
            return None
        elif self.is_circle:
            if len(inters) != 2:
                return None
            a1 = self._a_from_pt(inters[0])
            a2 = self._a_from_pt(inters[1])
            return [Arc2D(self.c, self.r, a1, a2), Arc2D(self.c, self.r, a2, a1)]
        elif len(inters) == 1:
            am = self._a_from_pt(inters[0])
            return [Arc2D(self.c, self.r, self.a1, am),
                    Arc2D(self.c, self.r, am, self.a2)]
        elif len(inters) == 2:
            am1 = self._a_from_pt(inters[0])
            am2 = self._a_from_pt(inters[1])
            if self._cc_difference(am1) < self._cc_difference(am2):
                return [Arc2D(self.c, self.r, self.a1, am1),
                        Arc2D(self.c, self.r, am1, am2),
                        Arc2D(self.c, self.r, am2, self.a2)]
            else:
                return [Arc2D(self.c, self.r, self.a1, am2),
                        Arc2D(self.c, self.r, am2, am1),
                        Arc2D(self.c, self.r, am1, self.a2)]

    def duplicate(self):
        """Get a copy of this object."""
        return self.__copy__()

    def to_dict(self):
        """Get Arc2D as a dictionary."""
        return {'type': 'Arc2D', 'c': self.c.to_array(),
                'r': self.r, 'a1': self.a1, 'a2': self.a2}

    def _pt_in(self, point):
        if self.is_circle:
            return True
        else:
            v = Vector2D(point.x - self.c.x, point.y - self.c.y)
            a = Vector2D(1, 0).angle_counterclockwise(v)
            return (not self.is_inverted and self.a1 < a < self.a2) or \
                (self.is_inverted and self.a1 > a > self.a2)

    def _a_from_pt(self, point):
        """Get the angle along the arc given a point along the arc."""
        v = Vector2D(point.x - self.c.x, point.y - self.c.y)
        return Vector2D(1, 0).angle_counterclockwise(v)

    def _cc_difference(self, angle):
        """Get counterclockwise different between an angle and the start of this arc."""
        _diff = angle - self.a1
        return _diff if not angle < self.a1 else 2 * math.pi + _diff

    def __copy__(self):
        return Arc2D(self.c, self.r, self.a1, self.a2)

    def __key(self):
        """A tuple based on the object properties, useful for hashing."""
        return (self.c, self.r, self.a1, self.a2)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(other, Arc2D) and self.__key() == other.__key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __repr__(self):
        return 'Arc2D (center <%.2f, %.2f>) (radius <%.2f>) (length <%.2f>)' % \
            (self.c.x, self.c.y, self.r, self.length)
