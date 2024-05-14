# coding=utf-8
"""3D Line Segment"""
from __future__ import division

from .pointvector import Point3D, Vector3D
from ._1d import Base1DIn3D


class LineSegment3D(Base1DIn3D):
    """3D line segment object.

    Args:
        p: A Point3D representing the first point of the line segment.
        v: A Vector3D representing the vector to the second point.

    Properties:
        * p
        * v
        * p1
        * p2
        * min
        * max
        * center
        * midpoint
        * endpoints
        * length
        * vertices
    """
    __slots__ = ()

    def __init__(self, p, v):
        """Initialize LineSegment3D."""
        Base1DIn3D.__init__(self, p, v)

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

    @classmethod
    def from_array(cls, line_array):
        """ Create a LineSegment3D from a nested array of two endpoint coordinates.

        Args:
            line_array: Nested tuples ((pt1.x, pt1.y, pt.z), (pt2.x, pt2.y, pt.z)),
                where pt1 and pt2 represent the endpoints of the line segment.
        """
        return LineSegment3D.from_end_points(*tuple(Point3D(*pt) for pt in line_array))

    @classmethod
    def from_line_segment2d(cls, line2d, z=0):
        """Initialize a new LineSegment3D from an LineSegment2D and a z value.

        Args:
            line2d: A LineSegment2D to be used to generate the LineSegment3D.
            z: A number for the Z coordinate value of the line.
        """
        base_p = Point3D(line2d.p.x, line2d.p.y, z)
        base_v = Vector3D(line2d.v.x, line2d.v.y, 0)
        return cls(base_p, base_v)

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
    def endpoints(self):
        """Tuple of endpoints """
        return (self.p1, self.p2)

    @property
    def length(self):
        """The length of the line segment."""
        return self.v.magnitude

    @property
    def vertices(self):
        """Tuple of both vertices in this object."""
        return (self.p1, self.p2)

    def is_horizontal(self, tolerance):
        """Test whether this line segment is horizontal within a certain tolerance.

        Args:
            tolerance: The maximum difference between the z values of the start and
                end coordinates at which the line segment is considered horizontal.
        """
        return abs(self.v.z) <= tolerance

    def is_vertical(self, tolerance):
        """Test whether this line segment is vertical within a certain tolerance.

        Args:
            tolerance: The maximum difference between the x and y values of the start
                and end coordinates at which the line segment is considered horizontal.
        """
        return abs(self.v.x) <= tolerance and abs(self.v.y) <= tolerance

    def flip(self):
        """Get a copy of this line segment that is flipped."""
        return LineSegment3D(self.p2, self.v.reverse())

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

    def scale(self, factor, origin=None):
        """Scale a line segment by a factor from an origin point.

        Args:
            factor: A number representing how much the line segment should be scaled.
            origin: A Point3D representing the origin from which to scale.
                If None, it will be scaled from the World origin (0, 0, 0).
        """
        return LineSegment3D(self.p.scale(factor, origin), self.v * factor)

    def subdivide(self, distances):
        """Get Point3D values along the line that subdivide it based on input distances.

        Args:
            distances: A list of distances along the line at which to subdivide it.
                This can also be a single number that will be repeated until the
                end of the line.
        """
        if isinstance(distances, (float, int)):
            distances = [distances]
        # this assert prevents the while loop from being infinite
        assert sum(distances) > 0, 'Segment subdivisions must be greater than 0'
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
            number: Integer for the number of segments into which the line will
                be divided.
        """
        # this assert prevents the while loop from being infinite
        assert number > 0, 'Segment subdivisions must be greater than 0'
        interval = 1 / number
        parameter = interval
        sub_pts = [self.p]
        while parameter <= 1:
            sub_pts.append(self.point_at(parameter))
            parameter += interval
        if len(sub_pts) != number + 1:  # tolerance issue with last point
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

    def split_with_plane(self, plane):
        """Split this LineSegment3D in 2 smaller LineSegment3Ds using a Plane.

        Args:
            plane: A Plane that will be used to split this line segment.

        Returns:
            A list of two LineSegment3D objects if the split was successful.
            Will be a list with 1 LineSegment3D if no intersection exists.
        """
        _plane_int = self.intersect_plane(plane)
        if _plane_int is not None:
            return [LineSegment3D.from_end_points(self.p1, _plane_int),
                    LineSegment3D.from_end_points(_plane_int, self.p2)]
        return [self]

    def to_dict(self):
        """Get LineSegment3D as a dictionary."""
        base = Base1DIn3D.to_dict(self)
        base['type'] = 'LineSegment3D'
        return base

    def to_array(self):
        """ A nested list representing the two line endpoint coordinates."""
        return (self.p1.to_array(), self.p2.to_array())

    def _u_in(self, u):
        return u >= 0.0 and u <= 1.0

    def __abs__(self):
        return abs(self.v)

    def __copy__(self):
        return LineSegment3D(self.p, self.v)

    def __key(self):
        """A tuple based on the object properties, useful for hashing."""
        return (hash(self.p), hash(self.v))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(other, LineSegment3D) and self.__key() == other.__key()

    def __repr__(self):
        return 'LineSegment3D (<%.2f, %.2f, %.2f> to <%.2f, %.2f, %.2f>)' % \
            (self.p.x, self.p.y, self.p.z,
             self.p.x + self.v.x, self.p.y + self.v.y, self.p.z + self.v.z)
