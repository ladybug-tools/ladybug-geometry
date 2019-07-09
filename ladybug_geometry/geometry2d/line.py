# coding=utf-8
"""2D Line Segment"""
from __future__ import division

from .pointvector import Vector2D, Point2D
from ._1d import Base1DIn2D
from ..intersection2d import closest_point2d_between_line2d


class LineSegment2D(Base1DIn2D):
    """2D line segment object.

    Properties:
        p: Base point
        v: Direction vector
        p1: First point (same as p)
        p2: Second point
        length: The length of the line segement
    """
    __slots__ = ()

    def __init__(self, p, v):
        """Initilize LineSegment2D.

        Args:
            p: A Point2D representing the first point of the line segment.
            v: A Vector2D representing the vector to the second point.
        """
        assert isinstance(p, Point2D), "Expected Point2D. Got {}.".format(type(p))
        assert isinstance(v, Vector2D), "Expected Vector2D. Got {}.".format(type(v))
        self._p = p
        self._v = v

    @classmethod
    def from_end_points(cls, p1, p2):
        """Initialize a line segment from a start point and and end point.

        Args:
            p1: A Point2D representing the first point of the line segment.
            p2: A Point2D representing the second point of the line segment.
        """
        return cls(p1, p2 - p1)

    @classmethod
    def from_sdl(cls, s, d, length):
        """Initialize a line segment from a start point, direction, and length.

        Args:
            s: A Point2D representing the start point of the line segment.
            d: A Vector2D representing the direction of the line segment.
            length: A number representing the length of the line segment.
        """
        return cls(s, d * length / d.magnitude)

    @property
    def p1(self):
        """First point (same as p)."""
        return self.p

    @property
    def p2(self):
        """Second point."""
        return Point2D(self.p.x + self.v.x, self.p.y + self.v.y)

    @property
    def midpoint(self):
        """Midpoint."""
        return self.point_at(0.5)

    @property
    def length(self):
        """The length of the line segment."""
        return self.v.magnitude

    def flip(self):
        """Get a copy of this line segment that is flipped."""
        return LineSegment2D(self.p2, self.v.reverse())

    def move(self, moving_vec):
        """Get a line segment that has been moved along a vector.

        Args:
            moving_vec: A Vector2D with the direction and distance to move the ray.
        """
        return LineSegment2D(self.p.move(moving_vec), self.v)

    def rotate(self, angle, origin):
        """Get a line segment that is rotated counterclockwise by a certain angle.

        Args:
            angle: An angle for rotation in radians.
            origin: A Point2D for the origin around which the line segment will
                be rotated.
        """
        return LineSegment2D(self.p.rotate(angle, origin), self.v.rotate(angle))

    def reflect(self, normal, origin):
        """Get a line segment reflected across a plane with the input normal vector and origin.

        Args:
            normal: A Vector2D representing the normal vector for the plane across
                which the line segment will be reflected. THIS VECTOR MUST BE NORMALIZED.
            origin: A Point2D representing the origin from which to reflect.
        """
        return LineSegment2D(self.p.reflect(normal, origin), self.v.reflect(normal))

    def scale(self, factor, origin=None):
        """Scale a line segment by a factor from an origin point.

        Args:
            factor: A number representing how much the line segment should be scaled.
            origin: A Point2D representing the origin from which to scale.
                If None, it will be scaled from the World origin (0, 0).
        """
        return LineSegment2D(self.p.scale(factor, origin), self.v * factor)

    def subdivide(self, distances):
        """Get Point2D values along the line that subdivide it based on input distances.

        Args:
            distances: A list of distances along the line at which to subdivide it.
                This can also be a single number that will be repeated unitl the end of
                the line.
        """
        if isinstance(distances, (float, int)):
            distances = [distances]
        line_length = self.length
        dist = distances[0]
        index = 0
        sub_pts = [self.p]
        while dist < line_length:
            sub_pts.append(self.point_at_length(dist))
            if index < len(distances) - 1:
                index += 1
            dist += distances[index]
        sub_pts.append(self.p2)
        return sub_pts

    def subdivide_evenly(self, number):
        """Get Point2D values along the line that divide it into evenly-spaced segments.

        Args:
            number: The number of segments into which the line will be divided.
        """
        interval = 1 / number
        parameter = interval
        sub_pts = [self.p]
        while parameter < 1:
            sub_pts.append(self.point_at(parameter))
            parameter += interval
        sub_pts.append(self.p2)
        return sub_pts

    def point_at(self, parameter):
        """Get a point at a given fraction along the line segment.

        Args:
            parameter: The fraction between the start and end point where the
                desired point lies. For example, 0.5 will yield the midpoint.
        """
        return self.p + self.v * parameter

    def point_at_length(self, length):
        """Get a point at a given distance along the line segment.

        Args:
            length: The distance along the line from the start point where the
                desired point lies.
        """
        return self.p + self.v * (length / self.length)

    def closest_points_between_line(self, line):
        """Get the two closest Point2D between this object to another LineSegment2D.

        Note that the line segments should not intersect for the result to be valid.

        Args:
            line: A LineSegment2D object to which the closest points
                will be computed.

        Returns:
            Two Point2D objects representing:
                1) The closest point on this object to the input line.
                2) The closest point on the input line to this object.
        """
        dist, pts = closest_point2d_between_line2d(self, line)
        return pts

    def distance_to_line(self, line):
        """Get the minimum distance between this object and the input LineSegment2D.

        Note that the line segments should not intersect for the result to be valid.

        Args:
            line: A LineSegment2D object to which the minimum distance will be computed.

        Returns:
            The minimum distance to the input line.
        """
        dist, pts = closest_point2d_between_line2d(self, line)
        return dist

    def _u_in(self, u):
        return u >= 0.0 and u <= 1.0

    def __abs__(self):
        return abs(self.v)

    def __eq__(self, other):
        if isinstance(other, LineSegment2D):
            return self.p == other.p and self.v == other.v
        else:
            return False

    def __repr__(self):
        return 'LineSegment2D (<%.2f, %.2f> to <%.2f, %.2f>)' % \
            (self.p.x, self.p.y, self.p.x + self.v.x, self.p.y + self.v.y)
