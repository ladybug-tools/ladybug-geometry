# coding=utf-8
"""2D Polyline"""
from __future__ import division

from .pointvector import Point2D
from .line import LineSegment2D
from .polygon import Polygon2D
from ..intersection2d import intersect_line2d, intersect_line2d_infinite
from ._2d import Base2DIn2D


class Polyline2D(Base2DIn2D):
    """2D polyline object.

    Args:
        vertices: A list of Point2D objects representing the vertices of the polyline.
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
        * length
        * is_self_intersecting
        * interpolated
    """
    __slots__ = ('_interpolated', '_segments', '_length', '_is_self_intersecting')

    def __init__(self, vertices, interpolated=False):
        """Initilize Polyline2D."""
        Base2DIn2D.__init__(self, vertices)
        self._interpolated = interpolated
        self._segments = None
        self._length = None
        self._is_self_intersecting = None

    @classmethod
    def from_dict(cls, data):
        """Create a Polyline2D from a dictionary.

        Args:
            data: A python dictionary in the following format.

        .. code-block:: python

            {
            "type": "Polyline2D",
            "vertices": [(0, 0), (10, 0), (0, 10)]
            }
        """
        interp = data['interpolated'] if 'interpolated' in data else False
        return cls(tuple(Point2D.from_array(pt) for pt in data['vertices']), interp)

    @property
    def vertices(self):
        """Tuple of all vertices in this geometry."""
        return self._vertices

    @property
    def segments(self):
        """Tuple of all line segments in the polyline."""
        if self._segments is None:
            self._segments = \
                tuple(LineSegment2D.from_end_points(vert, self._vertices[i + 1])
                      for i, vert in enumerate(self._vertices[:-1]))
        return self._segments

    @property
    def length(self):
        """The length of the polyline."""
        if self._length is None:
            self._length = sum([seg.length for seg in self.segments])
        return self._length

    @property
    def is_self_intersecting(self):
        """Boolean noting whether the polyline has self-intersecting segments."""
        if self._is_self_intersecting is None:
            self._is_self_intersecting = False
            _segs = self.segments
            for i, _s in enumerate(_segs[1: len(_segs) - 1]):
                _skip = (i, i + 1, i + 2)
                _other_segs = [x for j, x in enumerate(_segs) if j not in _skip]
                for _oth_s in _other_segs:
                    if _s.intersect_line_ray(_oth_s) is not None:  # intersection!
                        self._is_self_intersecting = True
                        break
                if self._is_self_intersecting is True:
                    break
        return self._is_self_intersecting

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
            tolerance: The minimum differnce between vertices below which vertices
                are considered the same.
        """
        return self._vertices[0].is_equivalent(self._vertices[-1], tolerance)

    def to_array(self):
        """Get a list of lists whenre each sub-list represents a Point2D vetex."""
        return tuple(pt.to_array() for pt in self.vertices)

    def to_polygon(self, tolerance):
        """Get a Polygon2D derived from this object.
        
        If the polyline is closed to within the tolerance, the segments of this
        polyline and the resulting polygon will match. Otherwise, an extra
        LineSegment2D will be added to connect the start and end of the polyline.

        Args:
            tolerance: The minimum differnce between vertices below which vertices
                are considered the same.
        """
        if self.is_closed(tolerance):
            return Polygon2D(self._vertices[:-1])
        return Polygon2D(self._vertices)

    def remove_colinear_vertices(self, tolerance):
        """Get a version of this polyline without colinear or duplicate vertices.

        Args:
            tolerance: The minimum distance that a vertex can be from a line
                before it is considered colinear.
        """
        if len(self.vertices) == 3:
            return self  # Polyline2D cannot have fewer than 3 vertices
        new_vertices = [self.vertices[0]]  # first vertex is always ok
        for i, _v in enumerate(self.vertices[1:-1]):
            _a = self[i].determinant(_v) + _v.determinant(self[i + 2]) + \
                self[i + 2].determinant(self[i])
            if abs(_a) >= tolerance:
                new_vertices.append(_v)
        new_vertices.append(self[-1])  # last vertex is always ok
        _new_poly = Polyline2D(new_vertices)
        self._transfer_properties(_new_poly)
        return _new_poly

    def reverse(self):
        """Get a copy of this polyline where the vertices are reversed."""
        _new_poly = Polyline2D(tuple(pt for pt in reversed(self.vertices)))
        self._transfer_properties(_new_poly)
        return _new_poly

    def move(self, moving_vec):
        """Get a polyline that has been moved along a vector.

        Args:
            moving_vec: A Vector2D with the direction and distance to move the polyline.
        """
        _new_poly = Polyline2D(tuple(pt.move(moving_vec) for pt in self.vertices))
        self._transfer_properties(_new_poly)
        return _new_poly

    def rotate(self, angle, origin):
        """Get a polyline that is rotated counterclockwise by a certain angle.

        Args:
            angle: An angle for rotation in radians.
            origin: A Point2D for the origin around which the point will be rotated.
        """
        _new_poly = Polyline2D(tuple(pt.rotate(angle, origin) for pt in self.vertices))
        self._transfer_properties(_new_poly)
        return _new_poly

    def reflect(self, normal, origin):
        """Get a polyline reflected across a plane with the input normal and origin.

        Args:
            normal: A Vector2D representing the normal vector for the plane across
                which the polyline will be reflected. THIS VECTOR MUST BE NORMALIZED.
            origin: A Point2D representing the origin from which to reflect.
        """
        _new_poly = Polyline2D(tuple(pt.reflect(normal, origin) for pt in self.vertices))
        self._transfer_properties(_new_poly)
        return _new_poly

    def scale(self, factor, origin=None):
        """Scale a polyline by a factor from an origin point.

        Args:
            factor: A number representing how much the polyline should be scaled.
            origin: A Point2D representing the origin from which to scale.
                If None, it will be scaled from the World origin (0, 0).
        """
        if origin is None:
            _new_poly = Polyline2D(tuple(
                Point2D(pt.x * factor, pt.y * factor) for pt in self.vertices))
        else:
            _new_poly = Polyline2D(tuple(
                pt.scale(factor, origin) for pt in self.vertices))
        _new_poly._interpolated = self._interpolated
        return _new_poly

    def intersect_line_ray(self, line_ray):
        """Get the intersections between this polyline and a Ray2D or LineSegment2D.

        Args:
            line_ray: A LineSegment2D or Ray2D or to intersect.

        Returns:
            A list with Point2D objects for the intersections.
            List will be empty if no intersection exists.
        """
        intersections = []
        for _s in self.segments:
            inters = intersect_line2d(_s, line_ray)
            if inters is not None:
                intersections.append(inters)
        return intersections

    def intersect_line_infinite(self, ray):
        """Get the intersections between this polyline and a Ray2D extended infintiely.

        Args:
            ray: A Ray2D or to intersect. This will be extended in both
                directions infinetly for the intersection.

        Returns:
            A list with Point2D objects for the intersections.
            List will be empty if no intersection exists.
        """
        intersections = []
        for _s in self.segments:
            inters = intersect_line2d_infinite(_s, ray)
            if inters is not None:
                intersections.append(inters)
        return intersections

    def to_dict(self):
        """Get Polyline2D as a dictionary."""
        base = {'type': 'Polyline2D',
                'vertices': [pt.to_array() for pt in self.vertices]}
        if self.interpolated:
            base['interpolated'] = self.interpolated
        return base

    @staticmethod
    def join_segments(segments, tolerance):
        """Get an array of Polyline2Ds from a list of LineSegment2Ds.

        Args:
            segments: An array of LineSegment2D objects.
            tolerance: The minimum difference in X, Y, and Z values at which Point2Ds
                are considred equivalent. Segments with points that match within the
                tolerance will be joined.

        Returns:
            An array of Polyline2D and LineSegment2D objects assembled from the
            joined segments.
        """
        # group the vertices that make up polylines
        grouped_verts = []
        base_seg = segments[0]
        remain_segs = list(segments[1:])
        while len(remain_segs) > 0:
            grouped_verts.append(
                Polyline2D._build_polyline(base_seg, remain_segs, tolerance))
            if len(remain_segs) > 1:
                base_seg = remain_segs[0]
                del remain_segs[0]
            elif len(remain_segs) == 1:  # lone last segment
                grouped_verts.append([segments[0].p1, segments[0].p2])
                del remain_segs[0]
        
        # create the Polyline2D and LineSegment2D objects
        joined_lines = []
        for v_list in grouped_verts:
            if len(v_list) == 2:
                joined_lines.append(LineSegment2D.from_end_points(v_list[0], v_list[1]))
            else:
                joined_lines.append(Polyline2D(v_list))
        return joined_lines

    def _transfer_properties(self, new_polyline):
        """Transfer properties from this polyline to a new polyline.

        This is used by the transform methods that don't alter the relationship of
        face vertices to one another (move, rotate, reflect).
        """
        new_polyline._interpolated = self._interpolated
        new_polyline._length = self._length
        new_polyline._is_self_intersecting = self._is_self_intersecting
    
    @staticmethod
    def _build_polyline(base_seg, other_segs, tol):
        """Attempt to build a list of polyline vertices from a base segment.
        
        Args:
            base_seg: A LineSegment2D to serve as the base of the Polyline.
            other_segs: A list of other LineSegment2D objects to attempt to
                connect to the base_seg. This method will delete any segments
                that are successfully connected to the output from this list. 
            tol: The tolerance to be used for connecting the line.
        
        Returns:
            A list of vertices that represent the longest Polyline to which the
            base_seg can be a part of given the other_segs as connections.
        """
        poly_verts = [base_seg.p1, base_seg.p2]
        more_to_check = True
        while more_to_check:
            for i, r_seg in enumerate(other_segs):
                if Polyline2D._connect_seg_to_poly(poly_verts, r_seg, tol):
                    del other_segs[i]
                    break
            else:
                more_to_check = False
        return poly_verts

    @staticmethod
    def _connect_seg_to_poly(poly_verts, seg, tol):
        """Connect a LineSegment2D to a list of polyline vertices.

        If successful, a Point2D will be appended to the poly_verts list and True
        will be returned. If not successful, the poly_verts list will remain unchanged
        and False will be returned.

        Args:
            poly_verts: An ordered list of Poin2Ds to which the segment should
                be connected.
            seg: A LineSegment2D to connect to the poly_verts.
            tol: The tolerance to be used for connecting the line.
        """
        p1, p2 = seg.p1, seg.p2
        if poly_verts[-1].is_equivalent(p1, tol):
            poly_verts.append(p2)
            return True
        elif poly_verts[0].is_equivalent(p2, tol):
            poly_verts.insert(0, p1)
            return True
        elif poly_verts[-1].is_equivalent(p2, tol):
            poly_verts.append(p1)
            return True
        elif poly_verts[0].is_equivalent(p1, tol):
            poly_verts.insert(0, p2)
            return True
        return False

    def __copy__(self):
        return Polyline2D(self._vertices, self._interpolated)

    def __key(self):
        """A tuple based on the object properties, useful for hashing."""
        return tuple(hash(pt) for pt in self._vertices) + (self._interpolated,)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(other, Polyline2D) and self.__key() == other.__key()

    def __repr__(self):
        return 'Polyline2D ({} vertices)'.format(len(self))
