# coding=utf-8
"""3D Line Segment"""
from __future__ import division

from .._immutable import immutable
from .pointvector import Point3D
from ._1d import Base1DIn3D


class LineSegment3D(Base1DIn3D):
    """3D line segment object.

    Properties:
        p: Base point
        v: Direction vector
        p1: First point (same as p)
        p2: Second point
        length: The length of the line segement
    """

    def __init__(self, p, v):
        """Initilize LineSegment3D.

        Args:
            p: A Point3D representing the first point of the line segment.
            v: A Vector3D representing the vector to the second point.
        """
        self.p = p
        self.v = v

    @classmethod
    def from_end_points(cls, p1, p2):
        """Initialize a line segment from a start point and and end point.

        Args:
            p1: A Point3D representing the first point of the line segment.
            p2: A Point3D representing the second point of the line segment.
        """
        return cls(p1, p2 - p1)

    @classmethod
    def from_sdl(cls, s, d, length):
        """Initialize a line segment from a start point, direction, and length.

        Args:
            s: A Point3D representing the start point of the line segment.
            d: A Vector3D representing the direction of the line segment.
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
        return Point3D(self.p.x + self.v.x, self.p.y + self.v.y, self.p.z + self.v.z)

    @property
    def midpoint(self):
        """Midpoint."""
        return self.point_at(0.5)

    @property
    def length(self):
        """The length of the line segment."""
        return self.v.magnitude

    def flip(self):
        """Flip the direction of the line segment."""
        self.p = self.p2
        self.v.reverse()

    def flipped(self):
        """Get a copy of this line segment that is flipped."""
        return LineSegment3D(self.p2, self.v.reversed())

    def move(self, moving_vec):
        """Get a line segment that has been moved along a vector.

        Args:
            moving_vec: A Vector3D with the direction and distance to move the ray.
        """
        return LineSegment3D(self.p.move(moving_vec), self.v)

    def rotate(self, axis, angle, origin):
        """Rotate a line segment by a certain angle around an axis and origin.

        Right hand rule applies:
        If axis has a positive orientation, rotation will be clockwise.
        If axis has a negative orientation, rotation will be counterclockwise.

        Args:
            axis: A Vector3D axis representing the axis of rotation.
            angle: An angle for rotation in radians.
            origin: A Point3D for the origin around which the object will be rotated.
        """
        return LineSegment3D(self.p.rotate(axis, angle, origin),
                             self.v.rotate(axis, angle))

    def rotate_xy(self, angle, origin):
        """Get a line segment rotated counterclockwise in the XY plane by a certain angle.

        Args:
            angle: An angle in radians.
            origin: A Point3D for the origin around which the object will be rotated.
        """
        return LineSegment3D(self.p.rotate_xy(angle, origin),
                             self.v.rotate_xy(angle))

    def reflect(self, normal, origin):
        """Get a line segment reflected across a plane with the input normal vector and origin.

        Args:
            normal: A Vector3D representing the normal vector for the plane across
                which the line segment will be reflected. THIS VECTOR MUST BE NORMALIZED.
            origin: A Point3D representing the origin from which to reflect.
        """
        return LineSegment3D(self.p.reflect(normal, origin), self.v.reflect(normal))

    def scale(self, factor, origin):
        """Scale a line segment by a factor from an origin point.

        Args:
            factor: A number representing how much the line segment should be scaled.
            origin: A Point3D representing the origin from which to scale.
        """
        return LineSegment3D(self.p.scale(factor, origin), self.v * factor)

    def scale_world_origin(self, factor):
        """Scale a line segment by a factor from the world origin. Faster than scale.

        Args:
            factor: A number representing how much the line segment should be scaled.
        """
        return LineSegment3D(self.p.scale_world_origin(factor), self.v * factor)

    def subdivide(self, distances):
        """Get Point3D values along the line that subdivide it based on input distances.

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
        """Get Point3D values along the line that divide it into evenly-spaced segments.

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
        """Get a Point3D at a given fraction along the line segment.

        Args:
            parameter: The fraction between the start and end point where the
                desired point lies. For example, 0.5 will yield the midpoint.
        """
        return self.p + self.v * parameter

    def point_at_length(self, length):
        """Get a Point3D at a given distance along the line segment.

        Args:
            length: The distance along the line from the start point where the
                desired point lies.
        """
        return self.p + self.v * (length / self.length)

    def to_immutable(self):
        """Get an immutable version of this object."""
        return LineSegment3DImmutable(self.p, self.v)

    def __abs__(self):
        return abs(self.v)

    def _u_in(self, u):
        return u >= 0.0 and u <= 1.0

    def __repr__(self):
        return 'Ladybug LineSegment3D (<%.2f, %.2f, %.2f> to <%.2f, %.2f, %.2f>)' % \
            (self.p.x, self.p.y, self.p.z,
             self.p.x + self.v.x, self.p.y + self.v.y, self.p.z + self.v.z)


@immutable
class LineSegment3DImmutable(LineSegment3D):
    """Immutable 3D Line Segment object."""
    _mutable = False

    def __init__(self, p, v):
        """Initilize LineSegment2D.

        Args:
            p: A Point3D representing the base of the ray.
            v: A Vector3D representing the direction of the ray.
        """
        self.p = p.to_immutable()
        self.v = v.to_immutable()

    def to_mutable(self):
        """Get a mutable version of this object."""
        return LineSegment3D(self.p.to_mutable(), self.v.to_mutable())

    def to_immutable(self):
        """Get an immutable version of this object."""
        return self
