# coding=utf-8
"""3D Polyline"""
from __future__ import division

from ..geometry2d.pointvector import Point2D
from ..geometry2d.polyline import Polyline2D

from ._2d import Base2DIn3D
from .pointvector import Point3D
from .line import LineSegment3D
from .plane import Plane
from ..intersection3d import intersect_line3d_plane
from .._polyline import _group_vertices


class Polyline3D(Base2DIn3D):
    """3D polyline object.

    Args:
        vertices: A list of Point3D objects representing the vertices of the polyline.
        interpolated: Boolean to note whether the polyline should be interpolated
            between the input vertices when it is translated to other interfaces.
            Note that this property has no bearing on the geometric calculations
            performed by this library and is only present in order to assist with
            display/translation.

    Properties:
        * vertices
        * segments
        * min
        * max
        * center
        * p1
        * p2
        * length
        * interpolated
    """
    __slots__ = ('_interpolated', '_segments', '_length')

    def __init__(self, vertices, interpolated=False):
        """Initialize Polyline3D."""
        Base2DIn3D.__init__(self, vertices)
        self._interpolated = interpolated
        self._segments = None
        self._length = None

    @classmethod
    def from_dict(cls, data):
        """Create a Polyline3D from a dictionary.

        Args:
            data: A python dictionary in the following format.

        .. code-block:: python

            {
                "type": "Polyline3D",
                "vertices": [(0, 0, 0), (10, 0, 2), (0, 10, 4)]
            }
        """
        interp = data['interpolated'] if 'interpolated' in data else False
        return cls(tuple(Point3D.from_array(pt) for pt in data['vertices']), interp)

    @classmethod
    def from_array(cls, point_array):
        """Create a Polyline3D from a nested array of vertex coordinates.

        Args:
            point_array: nested array of point arrays.
        """
        return Polyline3D(Point3D(*point) for point in point_array)

    @classmethod
    def from_polyline2d(cls, polyline2d, plane=Plane()):
        """Create a closed Polyline3D from a Polyline2D and a plane.

        Args:
            polyline2d: A Polyline2D object to be converted to a Polyline3D.
            plane: A Plane in which the Polyline2D sits.
        """
        return Polyline3D((plane.xy_to_xyz(pt) for pt in polyline2d.vertices),
                          polyline2d.interpolated)

    @property
    def segments(self):
        """Tuple of all line segments in the polyline."""
        if self._segments is None:
            self._segments = \
                tuple(LineSegment3D.from_end_points(vert, self._vertices[i + 1])
                      for i, vert in enumerate(self._vertices[:-1]))
        return self._segments

    @property
    def p1(self):
        """Starting point of the Polyline3D."""
        return self._vertices[0]

    @property
    def p2(self):
        """End point of the Polyline3D."""
        return self._vertices[-1]

    @property
    def length(self):
        """The length of the polyline."""
        if self._length is None:
            self._length = sum([seg.length for seg in self.segments])
        return self._length

    @property
    def interpolated(self):
        """Boolean noting whether the polyline should be interpolated upon translation.

        Note that this property has no bearing on the geometric calculations
        performed by this library and is only present in order to assist with
        display/translation.
        """
        return self._interpolated

    def is_closed(self, tolerance):
        """Test whether this polyline is closed to within the tolerance.

        Args:
            tolerance: The minimum difference between vertices below which vertices
                are considered the same.
        """
        return self._vertices[0].is_equivalent(self._vertices[-1], tolerance)

    def remove_colinear_vertices(self, tolerance):
        """Get a version of this polyline without colinear or duplicate vertices.

        Args:
            tolerance: The minimum distance that a vertex can be from a line
                before it is considered colinear.
        """
        if len(self.vertices) == 3:
            return self  # Polyline3D cannot have fewer than 3 vertices
        new_vertices = [self.vertices[0]]  # first vertex is always ok
        for i, _v in enumerate(self.vertices[1:-1]):
            if (self[i] - _v).cross(self[i + 2] - _v).magnitude >= tolerance:
                new_vertices.append(_v)
        new_vertices.append(self[-1])  # last vertex is always ok
        _new_poly = Polyline3D(new_vertices)
        self._transfer_properties(_new_poly)
        return _new_poly

    def reverse(self):
        """Get a copy of this polyline where the vertices are reversed."""
        _new_poly = Polyline3D(tuple(pt for pt in reversed(self.vertices)))
        self._transfer_properties(_new_poly)
        return _new_poly

    def move(self, moving_vec):
        """Get a polyline that has been moved along a vector.

        Args:
            moving_vec: A Vector3D with the direction and distance to move the polyline.
        """
        _new_poly = Polyline3D(tuple(pt.move(moving_vec) for pt in self.vertices))
        self._transfer_properties(_new_poly)
        return _new_poly

    def rotate(self, axis, angle, origin):
        """Rotate a polyline by a certain angle around an axis and origin.

        Right hand rule applies:
        If axis has a positive orientation, rotation will be clockwise.
        If axis has a negative orientation, rotation will be counterclockwise.

        Args:
            axis: A Vector3D axis representing the axis of rotation.
            angle: An angle for rotation in radians.
            origin: A Point3D for the origin around which the point will be rotated.
        """
        _new_poly = Polyline3D(tuple(pt.rotate(axis, angle, origin)
                                     for pt in self.vertices))
        self._transfer_properties(_new_poly)
        return _new_poly

    def rotate_xy(self, angle, origin):
        """Get a polyline rotated counterclockwise in the XY plane by a certain angle.

        Args:
            angle: An angle in radians.
            origin: A Point3D for the origin around which the object will be rotated.
        """
        _new_p = Polyline3D(tuple(pt.rotate_xy(angle, origin) for pt in self.vertices))
        self._transfer_properties(_new_p)
        return _new_p

    def reflect(self, normal, origin):
        """Get a polyline reflected across a plane with the input normal and origin.

        Args:
            normal: A Vector3D representing the normal vector for the plane across
                which the polyline will be reflected. THIS VECTOR MUST BE NORMALIZED.
            origin: A Point3D representing the origin from which to reflect.
        """
        _new_poly = Polyline3D(tuple(pt.reflect(normal, origin) for pt in self.vertices))
        self._transfer_properties(_new_poly)
        return _new_poly

    def scale(self, factor, origin=None):
        """Scale a polyline by a factor from an origin point.

        Args:
            factor: A number representing how much the polyline should be scaled.
            origin: A Point3D representing the origin from which to scale.
                If None, it will be scaled from the World origin (0, 0, 0).
        """
        if origin is None:
            _new_poly = Polyline3D(tuple(
                Point3D(pt.x * factor, pt.y * factor, pt.z * factor)
                for pt in self.vertices))
        else:
            _new_poly = Polyline3D(tuple(
                pt.scale(factor, origin) for pt in self.vertices))
        _new_poly._interpolated = self._interpolated
        return _new_poly

    def intersect_plane(self, plane):
        """Get the intersections between this polyline and a Plane.

        Args:
            plane: A Plane that will be intersected with this object.

        Returns:
            A list with Point3D objects for the intersections.
            List will be empty if no intersection exists.
        """
        intersections = []
        for _s in self.segments:
            inters = intersect_line3d_plane(_s, plane)
            if inters is not None:
                intersections.append(inters)
        return intersections

    def split_with_plane(self, plane):
        """Split this Polyline3D into Polyline3Ds and LineSegment3Ds using a Plane.

        Args:
            plane: A Plane that will be used to split this polyline.

        Returns:
            A list of Polyline3D and LineSegment3D objects if the split was successful.
            Will be a list with 1 Polyline3D if no intersection exists.
        """
        # group the vertices based on when they cross the plane
        grouped_verts = [[self._vertices[0]]]
        for _s in self.segments:
            inters = intersect_line3d_plane(_s, plane)
            if inters is None:
                grouped_verts[-1].append(_s.p2)
            else:  # intersection; start a new group
                grouped_verts[-1].append(inters)
                grouped_verts.append([inters, _s.p2])

        # make new Polyline3D and LineSegment3D objects based on the groups
        return self._grouped_verts_to_objs(grouped_verts, self._interpolated)

    def to_dict(self):
        """Get Polyline3D as a dictionary."""
        base = {'type': 'Polyline3D',
                'vertices': [pt.to_array() for pt in self.vertices]}
        if self.interpolated:
            base['interpolated'] = self.interpolated
        return base

    def to_array(self):
        """Get a list of lists where each sub-list represents a Point3D vertex."""
        return tuple(pt.to_array() for pt in self.vertices)

    def to_polyline2d(self):
        """Get a Polyline2D in the XY plane derived from this 3D polyline."""
        return Polyline2D((Point2D(pt.x, pt.y) for pt in self.vertices), self.interpolated)

    @staticmethod
    def join_segments(segments, tolerance):
        """Get an array of Polyline3Ds from a list of LineSegment3Ds.

        Args:
            segments: An array of LineSegment3D objects.
            tolerance: The minimum difference in X, Y, and Z values at which Point2Ds
                are considred equivalent. Segments with points that match within the
                tolerance will be joined.

        Returns:
            An array of Polyline3D and LineSegment3D objects assembled from the
            joined segments.
        """
        # group the vertices that make up polylines
        grouped_verts = _group_vertices(segments, tolerance)
        # create the Polyline3D and LineSegment3D objects
        return Polyline3D._grouped_verts_to_objs(grouped_verts)

    def _transfer_properties(self, new_polyline):
        """Transfer properties from this polyline to a new polyline."""
        new_polyline._interpolated = self._interpolated
        new_polyline._length = self._length

    @staticmethod
    def _grouped_verts_to_objs(grouped_verts, interpolated=False):
        joined_lines = []
        for v_list in grouped_verts:
            if len(v_list) == 2:
                joined_lines.append(LineSegment3D.from_end_points(v_list[0], v_list[1]))
            else:
                joined_lines.append(Polyline3D(v_list, interpolated))
        return joined_lines

    def __copy__(self):
        return Polyline3D(self._vertices, self._interpolated)

    def __key(self):
        """A tuple based on the object properties, useful for hashing."""
        return tuple(hash(pt) for pt in self._vertices) + (self._interpolated,)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(other, Polyline3D) and self.__key() == other.__key()

    def __repr__(self):
        return 'Polyline3D ({} vertices)'.format(len(self))
