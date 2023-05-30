# coding=utf-8
"""2D Polygon"""
from __future__ import division
import math
import time
from collections import deque

try:  # Python3
    from queue import PriorityQueue
except ImportError:  # Python2
    from Queue import PriorityQueue

from .pointvector import Point2D, Vector2D
from .line import LineSegment2D
from .ray import Ray2D
from ..triangulation import _linked_list, _eliminate_holes
from ..intersection2d import intersect_line2d, intersect_line2d_infinite, \
    does_intersection_exist_line2d, closest_point2d_on_line2d
from ._2d import Base2DIn2D
import ladybug_geometry.boolean as pb

inf = float("inf")


class Polygon2D(Base2DIn2D):
    """2D polygon object.

    Args:
        vertices: A list of Point2D objects representing the vertices of the polygon.

    Properties:
        * vertices
        * segments
        * min
        * max
        * center
        * perimeter
        * area
        * is_clockwise
        * is_convex
        * is_self_intersecting
        * is_valid
    """
    __slots__ = ('_segments', '_perimeter', '_area',
                 '_is_clockwise', '_is_convex', '_is_self_intersecting')

    def __init__(self, vertices):
        """Initialize Polygon2D."""
        Base2DIn2D.__init__(self, vertices)
        self._segments = None
        self._perimeter = None
        self._area = None
        self._is_clockwise = None
        self._is_convex = None
        self._is_self_intersecting = None

    @classmethod
    def from_dict(cls, data):
        """Create a Polygon2D from a dictionary.

        Args:
            data: A python dictionary in the following format

        .. code-block:: python

            {
                "type": "Polygon2D",
                "vertices": [(0, 0), (10, 0), (0, 10)]
            }
        """
        return cls(tuple(Point2D.from_array(pt) for pt in data['vertices']))

    @classmethod
    def from_array(cls, point_array):
        """Create a Polygon2D from a nested array of vertex coordinates.

        Args:
            point_array: nested array of point arrays.
        """
        return Polygon2D(Point2D(*point) for point in point_array)

    @classmethod
    def from_rectangle(cls, base_point, height_vector, base, height):
        """Initialize Polygon2D from rectangle parameters.

        Initializing a polygon this way has the added benefit of having its properties
        quickly computed.

        Args:
            base_point: A Point2D for the lower left vertex of the polygon.
            height_vector: A vector denoting the direction of the rectangle height.
            base: A number indicating the length of the base of the rectangle.
            height: A number indicating the length of the height of the rectangle.
        """
        assert isinstance(base_point, Point2D), \
            'base_point must be Point2D. Got {}.'.format(type(base_point))
        assert isinstance(height_vector, Vector2D), \
            'height_vector must be Vector2D. Got {}.'.format(type(height_vector))
        assert isinstance(base, (float, int)), 'base must be a number.'
        assert isinstance(height, (float, int)), 'height must be a number.'
        _hv_norm = height_vector.normalize()
        _bv = Vector2D(_hv_norm.y, -_hv_norm.x) * base
        _hv = _hv_norm * height
        _verts = (base_point, base_point + _bv, base_point + _hv + _bv, base_point + _hv)
        polygon = cls(_verts)
        polygon._perimeter = base * 2 + height * 2
        polygon._area = base * height
        polygon._is_clockwise = False
        polygon._is_convex = True
        polygon._is_self_intersecting = False
        return polygon

    @classmethod
    def from_regular_polygon(cls, number_of_sides, radius=1, base_point=Point2D()):
        """Initialize Polygon2D from regular polygon parameters.

        Args:
            number_of_sides: An integer for the number of sides on the regular
                polygon. This number must be greater than 2.
            radius: A number indicating the distance from the polygon's center
                where the vertices of the polygon will lie.
                The default is set to 1.
            base_point: A Point2D for the center of the regular polygon.
                The default is the Origin at (0, 0).
        """
        assert isinstance(number_of_sides, int), 'number_of_sides must be an ' \
            'integer. Got {}.'.format(type(number_of_sides))
        assert number_of_sides > 2, 'number_of_sides must be greater than 2. ' \
            'Got {}.'.format(number_of_sides)
        assert isinstance(base_point, Point2D), \
            'base_point must be Point2D. Got {}.'.format(type(base_point))
        assert isinstance(radius, (float, int)), 'height must be a number.'

        # calculate angle at which each vertex is rotated from the previous one
        angle = (math.pi * 2) / number_of_sides
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # pick a starting vertex that makes sense for the number of sides
        if number_of_sides % 2 == 0:
            start_vert = Point2D(base_point.x - radius, base_point.y)
            start_vert = start_vert.rotate(angle / 2, base_point)
        else:
            start_vert = Point2D(base_point.x, base_point.y + radius)
        _vertices = [start_vert]

        # generate the vertices
        for i in range(number_of_sides - 1):
            last_pt = _vertices[-1]
            qx = cos_a * (last_pt.x - base_point.x) - sin_a * (last_pt.y - base_point.y)
            qy = sin_a * (last_pt.x - base_point.x) + cos_a * (last_pt.y - base_point.y)
            _vertices.append(Point2D(qx + base_point.x, qy + base_point.y))

        # build the new polygon and set the properties that we know.
        _new_poly = cls(_vertices)
        _new_poly._is_clockwise = False
        _new_poly._is_convex = True
        _new_poly._is_self_intersecting = False
        return _new_poly

    @classmethod
    def from_shape_with_hole(cls, boundary, hole):
        """Initialize a Polygon2D from a boundary shape with a hole inside of it.

        This method will convert the shape into a single concave polygon by drawing
        a line from the hole to the outer boundary.

        Args:
            boundary: A list of Point2D objects for the outer boundary of the polygon
                inside of which the hole is contained.
            hole: A list of Point2D objects for the hole.
        """
        # check that the inputs are in the correct format
        assert isinstance(boundary, list), \
            'boundary should be a list. Got {}'.format(type(boundary))
        assert isinstance(hole, list), \
            'hole should be a list. Got {}'.format(type(hole))

        # check that the direction of vertices for the hole is opposite the boundary
        bound_direction = Polygon2D._are_clockwise(boundary)
        if cls._are_clockwise(hole) is bound_direction:
            hole.reverse()

        # join the hole with the boundary at the closest point
        dist_dict = {}
        for i, b_pt in enumerate(boundary):
            for j, h_pt in enumerate(hole):
                dist_dict[b_pt.distance_to_point(h_pt)] = (i, j)
        boundary = cls._merge_boundary_and_hole(boundary, hole, dist_dict)

        # return the polygon with some properties set based on what we know
        _new_poly = cls(boundary)
        _new_poly._is_clockwise = bound_direction
        _new_poly._is_convex = False
        _new_poly._is_self_intersecting = False
        return _new_poly

    @classmethod
    def from_shape_with_holes(cls, boundary, holes):
        """Initialize a Polygon2D from a boundary shape with holes inside of it.

        This method will convert the shape into a single concave polygon by drawing
        lines from the holes to the outer boundary.

        Args:
            boundary: A list of Point2D objects for the outer boundary of the polygon
                inside of which all of the holes are contained.
            holes: A list of lists with one list for each hole in the shape. Each hole
                should be a list of at least 3 Point2D objects.
        """
        # check that the inputs are in the correct format.
        assert isinstance(boundary, list), \
            'boundary should be a list. Got {}'.format(type(boundary))
        assert isinstance(holes, list), \
            'holes should be a list. Got {}'.format(type(holes))
        for hole in holes:
            assert isinstance(hole, list), \
                'hole should be a list. Got {}'.format(type(hole))
            assert len(hole) >= 3, \
                'hole should have at least 3 vertices. Got {}'.format(len(hole))

        # check that the direction of vertices for the hole is opposite the boundary
        bound_direction = cls._are_clockwise(boundary)
        for hole in holes:
            if cls._are_clockwise(hole) is bound_direction:
                hole.reverse()

        # recursively add the nearest hole to the boundary until there are none left.
        boundary = cls._merge_boundary_and_holes(boundary, holes)

        # return the polygon with some properties set based on what we know
        _new_poly = cls(boundary)
        _new_poly._is_clockwise = bound_direction
        _new_poly._is_convex = False
        _new_poly._is_self_intersecting = False
        return _new_poly

    @classmethod
    def from_shape_with_holes_fast(cls, boundary, holes):
        """Initialize a Polygon2D from a boundary shape with holes using a fast method.

        This method is similar in principle to the from_shape_with_holes method
        but it uses David Eberly's algorithm for finding a bridge between the holes
        and outer polygon. This is extremely fast in comparison to the methods used
        by from_shape_with_holes but is not always the prettiest or the shortest
        pathway through the holes. Granted, it is very practical for shapes with
        lots of holes (eg. 100 holes) and will run in a fraction of the time for
        this case.

        Args:
            boundary: A list of Point2D objects for the outer boundary of the polygon
                inside of which all of the holes are contained.
            holes: A list of lists with one list for each hole in the shape. Each hole
                should be a list of at least 3 Point2D objects.
        """
        # check the initial direction of the boundary vertices
        bound_direction = cls._are_clockwise(boundary)
        # format the coordinates for input to the earcut methods
        vert_coords, hole_indices = [], None
        for pt in boundary:
            vert_coords.append(pt.x)
            vert_coords.append(pt.y)
        hole_indices = []
        for hole in holes:
            hole_indices.append(int(len(vert_coords) / 2))
            for pt in hole:
                vert_coords.append(pt.x)
                vert_coords.append(pt.y)

        # eliminate the holes within the list
        outer_len = hole_indices[0] * 2
        outer_node = _linked_list(vert_coords, 0, outer_len, 2, True)
        outer_node = _eliminate_holes(vert_coords, hole_indices, outer_node, 2)

        # loop through the chain of nodes and translate them to Point2D
        start_i = outer_node.i
        vertices = [Point2D(outer_node.x, outer_node.y)]
        node = outer_node.next
        node_counter, orig_start_i = 0, 0
        while node.i != start_i:
            vertices.append(Point2D(node.x, node.y))
            node_counter += 1
            if node.i == 0:
                orig_start_i = node_counter
            node = node.next

        # ensure that the starting vertex is the same as the input boundary
        vertices = vertices[orig_start_i:] + vertices[:orig_start_i]
        vertices[0] = boundary[0]  # this avoids issues of floating point tolerance

        # return the polygon with some properties set based on what we know
        _new_poly = cls(vertices)
        _new_poly._is_clockwise = bound_direction
        _new_poly._is_convex = False
        _new_poly._is_self_intersecting = False
        return _new_poly

    @property
    def vertices(self):
        """Tuple of all vertices in this geometry."""
        return self._vertices

    @property
    def segments(self):
        """Tuple of all line segments in the polygon."""
        if self._segments is None:
            _segs = self._segments_from_vertices(self.vertices)
            self._segments = tuple(_segs)
        return self._segments

    @property
    def perimeter(self):
        """The perimeter of the polygon."""
        if self._perimeter is None:
            self._perimeter = sum([seg.length for seg in self.segments])
        return self._perimeter

    @property
    def area(self):
        """The area of the polygon."""
        if self._area is None:
            _a = 0
            for i, pt in enumerate(self.vertices):
                _a += self.vertices[i - 1].x * pt.y - self.vertices[i - 1].y * pt.x
            self._area = _a / 2
        return abs(self._area)

    @property
    def is_clockwise(self):
        """Boolean for whether the polygon vertices are in clockwise order."""
        if self._is_clockwise is None:
            if self._area is None:
                self.area
            self._is_clockwise = self._area < 0
        return self._is_clockwise

    @property
    def is_convex(self):
        """Boolean noting whether the polygon is convex (True) or non-convex (False)."""
        if self._is_convex is None:
            self._is_convex = True
            if len(self.vertices) == 3:
                pass
            else:
                _segs = self.segments
                if self.is_clockwise:
                    for i, _s in enumerate(_segs):
                        if _segs[i - 1].v.determinant(_s.v) > 0:  # counterclockwise turn
                            self._is_convex = False
                            break
                else:
                    for i, _s in enumerate(_segs):
                        if _segs[i - 1].v.determinant(_s.v) < 0:  # clockwise turn
                            self._is_convex = False
                            break
        return self._is_convex

    @property
    def is_self_intersecting(self):
        """Boolean noting whether the polygon has self-intersecting edges.

        Note that this property is relatively computationally intense to obtain and
        most CAD programs forbid surfaces with self-intersecting edges.
        So this property should only be used in quality control scripts where the
        origin of the geometry is unknown.
        """
        if self._is_self_intersecting is None:
            self._is_self_intersecting = False
            if self.is_convex is False:
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
    def is_valid(self):
        """Boolean noting whether the polygon is valid (having a non-zero area).

        Note that polygons are still considered valid if they have self-intersecting
        edges, or duplicate/colinear vertices. The s_self_intersecting property
        identifies self-intersecting edges, and the remove_colinear_vertices method
        will remove duplicate/colinear vertices.
        """
        return not self.area == 0

    def is_equivalent(self, other, tolerance):
        """Boolean for equivalence between this polygon and another (within tolerance).

        The order of the polygon vertices do not have to start from the
        same vertex for equivalence to be true, but must be in the same counterclockwise
        or clockwise order.

        Args:
            other: Polygon2D for comparison.
            tolerance: float representing point equivalence.

        Returns:
            True if equivalent else False
        """

        # Check number of points
        if len(self.vertices) != len(other.vertices):
            return False

        vertices = self.vertices

        # Check order
        if not vertices[0].is_equivalent(other.vertices[0], tolerance):
            self_idx = None
            other_pt = other.vertices[0]
            for i, pt in enumerate(self.vertices):
                if pt.is_equivalent(other_pt, tolerance):
                    self_idx = i
                    break

            if self_idx is None:
                return False

            # Re-order polygon vertices to match other
            vertices = vertices[self_idx:] + vertices[:self_idx]

        is_equivalent = True
        for pt, other_pt in zip(vertices[1:], other.vertices[1:]):
            is_equivalent = is_equivalent and pt.is_equivalent(other_pt, tolerance)
        return is_equivalent

    def pole_of_inaccessibility(self, tolerance):
        """Get the pole of inaccessibility for the polygon.

        The pole of inaccessibility is the most distant internal point from the
        polygon outline. It is not to be confused with the centroid, which
        represents the "center of mass" of the polygon and may be outside of
        the polygon if the shape is concave. The poly of inaccessibility is
        useful for optimal placement of a text label on a polygon.

        The algorithm here is a port of the polylabel library from MapBox
        assembled by Michal Hatak. (https://github.com/Twista/python-polylabel).

        Args:
            tolerance: The precision to which the pole of inaccessibility
                will be computed.
        """
        # compute the cell size from the bounding rectangle
        min_x, min_y = self.min.x, self.min.y
        max_x, max_y = self.max.x, self.max.y
        width = max_x - min_x
        height = max_y - min_y
        cell_size = min(width, height)
        h = cell_size / 2.0
        if cell_size == 0:  # degenerate polygon; just return the minimum
            return self.min

        # get an array representation of the polygon and set up the priority queue
        _polygon = tuple(pt.to_array() for pt in self.vertices)
        cell_queue = PriorityQueue()

        # cover polygon with initial cells
        x = min_x
        while x < max_x:
            y = min_y
            while y < max_y:
                c = _Cell(x + h, y + h, h, _polygon)
                y += cell_size
                cell_queue.put((-c.max, time.time(), c))
            x += cell_size

        best_cell = self._get_centroid_cell(_polygon)

        bbox_cell = _Cell(min_x + width / 2, min_y + height / 2, 0, _polygon)
        if bbox_cell.d > best_cell.d:
            best_cell = bbox_cell

        # recursively iterate until we find the pole
        num_of_probes = cell_queue.qsize()
        while not cell_queue.empty():
            _, __, cell = cell_queue.get()

            if cell.d > best_cell.d:
                best_cell = cell

            if cell.max - best_cell.d <= tolerance:
                continue

            h = cell.h / 2
            c = _Cell(cell.x - h, cell.y - h, h, _polygon)
            cell_queue.put((-c.max, time.time(), c))
            c = _Cell(cell.x + h, cell.y - h, h, _polygon)
            cell_queue.put((-c.max, time.time(), c))
            c = _Cell(cell.x - h, cell.y + h, h, _polygon)
            cell_queue.put((-c.max, time.time(), c))
            c = _Cell(cell.x + h, cell.y + h, h, _polygon)
            cell_queue.put((-c.max, time.time(), c))
            num_of_probes += 4
        return Point2D(best_cell.x, best_cell.y)

    def remove_colinear_vertices(self, tolerance):
        """Get a version of this polygon without colinear or duplicate vertices.

        Args:
            tolerance: The minimum distance that a vertex can be from a line
                before it is considered colinear.
        """
        if len(self.vertices) == 3:
            return self  # Polygon2D cannot have fewer than 3 vertices
        new_vertices = []
        skip = 0
        for i, _v in enumerate(self.vertices):
            _a = self[i - 2].determinant(self[i - 1]) + self[i - 1].determinant(_v) + \
                _v.determinant(self[i - 2])
            if abs(_a) >= tolerance:
                new_vertices.append(self[i - 1])
                skip = 0
            else:
                skip += 1
        if skip != 0 and self.vertices[-2].is_equivalent(self.vertices[-1], tolerance):
            pts_2d = self.vertices
            _a = pts_2d[-3].determinant(pts_2d[-1]) + \
                pts_2d[-1].determinant(pts_2d[0]) + pts_2d[0].determinant(pts_2d[-3])
            if abs(_a) >= tolerance:
                new_vertices.append(pts_2d[-1])
        return Polygon2D(new_vertices)

    def reverse(self):
        """Get a copy of this polygon where the vertices are reversed."""
        _new_poly = Polygon2D(tuple(pt for pt in reversed(self.vertices)))
        self._transfer_properties(_new_poly)
        if self._is_clockwise is not None:
            _new_poly._is_clockwise = not self._is_clockwise
        return _new_poly

    def move(self, moving_vec):
        """Get a polygon that has been moved along a vector.

        Args:
            moving_vec: A Vector2D with the direction and distance to move the polygon.
        """
        _new_poly = Polygon2D(tuple(pt.move(moving_vec) for pt in self.vertices))
        self._transfer_properties(_new_poly)
        return _new_poly

    def rotate(self, angle, origin):
        """Get a polygon that is rotated counterclockwise by a certain angle.

        Args:
            angle: An angle for rotation in radians.
            origin: A Point2D for the origin around which the point will be rotated.
        """
        _new_poly = Polygon2D(tuple(pt.rotate(angle, origin) for pt in self.vertices))
        self._transfer_properties(_new_poly)
        return _new_poly

    def reflect(self, normal, origin):
        """Get a polygon reflected across a plane with the input normal and origin.

        Args:
            normal: A Vector2D representing the normal vector for the plane across
                which the polygon will be reflected. THIS VECTOR MUST BE NORMALIZED.
            origin: A Point2D representing the origin from which to reflect.
        """
        _new_poly = Polygon2D(tuple(pt.reflect(normal, origin) for pt in self.vertices))
        self._transfer_properties(_new_poly)
        if self._is_clockwise is not None:
            _new_poly._is_clockwise = not self._is_clockwise
        return _new_poly

    def scale(self, factor, origin=None):
        """Scale a polygon by a factor from an origin point.

        Args:
            factor: A number representing how much the polygon should be scaled.
            origin: A Point2D representing the origin from which to scale.
                If None, it will be scaled from the World origin (0, 0).
        """
        if origin is None:
            return Polygon2D(tuple(
                Point2D(pt.x * factor, pt.y * factor) for pt in self.vertices))
        else:
            return Polygon2D(tuple(pt.scale(factor, origin) for pt in self.vertices))

    def offset(self, distance, check_intersection=False):
        """Offset the polygon by a given distance inwards or outwards.

        Note that the resulting shape may be self-intersecting if the distance
        is large enough and the is_self_intersecting property may be used to identify
        these shapes.

        Args:
            distance: The distance inwards that the polygon will be offset.
                Positive values will always be offset inwards while negative ones
                will be offset outwards.
            check_intersection: A boolean to note whether the resulting operation
                should be checked for self intersection and, if so, None will be
                returned instead of the mis-shaped polygon.
        """
        # make sure the offset is not zero
        if distance == 0:
            return self

        # loop through the vertices and get the new offset vectors
        init_verts = self._vertices if not self.is_clockwise \
            else list(reversed(self._vertices))
        init_verts = [pt for i, pt in enumerate(init_verts) if pt != init_verts[i - 1]]
        move_vecs, max_i = [], len(init_verts) - 1
        for i, pt in enumerate(init_verts):
            v1 = init_verts[i - 1] - pt
            end_i = i + 1 if i != max_i else 0
            v2 = init_verts[end_i] - pt
            if not self.is_clockwise:
                ang = v1.angle_clockwise(v2) / 2
                if ang == 0:
                    ang = math.pi / 2
                m_vec = v1.rotate(-ang).normalize()
                m_dist = distance / math.sin(ang)
            else:
                ang = v1.angle_counterclockwise(v2) / 2
                if ang == 0:
                    ang = math.pi / 2
                m_vec = v1.rotate(ang).normalize()
                m_dist = -distance / math.sin(ang)
            m_vec = m_vec * m_dist
            move_vecs.append(m_vec)

        # move the vertices by the offset to create the new Polygon2D
        new_pts = tuple(pt.move(m_vec) for pt, m_vec in zip(init_verts, move_vecs))
        if self.is_clockwise:
            new_pts = tuple(reversed(new_pts))
        new_poly = Polygon2D(new_pts)

        # check for self intersection between the moving vectors if requested
        if check_intersection:
            poly_segs = new_poly.segments
            _segs = [LineSegment2D(p, v) for p, v in zip(init_verts, move_vecs)]
            _skip = (0, len(_segs) - 1)
            _other_segs = [x for j, x in enumerate(poly_segs) if j not in _skip]
            for _oth_s in _other_segs:
                if _segs[0].intersect_line_ray(_oth_s) is not None:  # intersection!
                    return None
            for i, _s in enumerate(_segs[1: len(_segs)]):
                _skip = (i, i + 1)
                _other_segs = [x for j, x in enumerate(poly_segs) if j not in _skip]
                for _oth_s in _other_segs:
                    if _s.intersect_line_ray(_oth_s) is not None:  # intersection!
                        return None
        return new_poly

    def intersect_line_ray(self, line_ray):
        """Get the intersections between this polygon and a Ray2D or LineSegment2D.

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
        """Get the intersections between this polygon and a Ray2D extended infinitely.

        Args:
            ray: A Ray2D or to intersect. This will be extended in both
                directions infinitely for the intersection.

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

    def point_relationship(self, point, tolerance):
        """Test whether a Point2D lies inside, outside or on the boundary of the polygon.

        Compared to other methods like is_point_inside this method is slow. However,
        it covers all edge cases, including the literal edge of the polygon.

        Args:
            point: A Point2D for which the relationship to the polygon will be tested.
            tolerance: The minimum distance from the edge at which a point is
                considered to lie on the edge.

        Returns:
            An integer denoting the relationship of the point.

            This will be one of the following:

            * -1 = Outside polygon
            *  0 = On the edge of the polygon
            * +1 = Inside polygon
        """
        if self.is_point_on_edge(point, tolerance):
            return 0
        if self.is_point_inside_bound_rect(point):
            return 1
        return -1

    def is_point_on_edge(self, point, tolerance):
        """Test whether a Point2D lies on the boundary edges of the polygon.

        Args:
            point: A Point2D for which the edge relationship will be tested.
            tolerance: The minimum distance from the edge at which a point is
                considered to lie on the edge.

        Returns:
            A boolean denoting whether the point lies on the polygon edges (True)
            or not on the edges (False).
        """
        for _s in self.segments:
            close_pt = closest_point2d_on_line2d(point, _s)
            if point.distance_to_point(close_pt) <= tolerance:
                return True
        return False

    def is_point_inside_check(self, point):
        """Test whether a Point2D lies inside the polygon with checks for fringe cases.

        This method uses the same calculation as the the `is_point_inside` method
        but it includes additional checks for the fringe cases noted in the
        `is_point_inside` description. Using this method means that it will always
        yield the right result for all convex polygons and concave polygons with
        one concave turn (provided that they do not have colinear vertices).
        This is suitable for nearly all practical purposes and the only cases
        that could yield an incorrect result are when a point is co-linear with
        two or more polygon edges along the X vector like so:

        .. code-block:: shell

             _____     _____     _____
            |  .  |___|     |___|     |
            |_________________________|

        While this method covers most fringe cases, it will not test for whether
        a point lies perfectly on the edge of the polygon so it assesses whether
        a point lies inside the polygon up to Python floating point tolerance
        (16 digits). If distinguishing edge conditions from inside/ outside is
        important, the `point_relationship` method should be used.

        Args:
            point: A Point2D for which the inside/ outside relationship will be tested.
        Returns:
            A boolean denoting whether the point lies inside (True) or outside (False).
        """
        def non_duplicate_intersect(test_ray):
            inters = []
            n_int = 0
            for _s in self.segments:
                inter = intersect_line2d(_s, test_ray)
                if inter is not None:
                    try:
                        if inter != inters[-1]:  # ensure intersection is not duplicated
                            n_int += 1
                            inters.append(inter)
                    except IndexError:
                        n_int += 1
                        inters.append(inter)
            return n_int, inters

        n_int, inters = non_duplicate_intersect(Ray2D(point, Vector2D(1, 0)))
        # check that intersections do not form a polygon segment co-linear with test_ray
        if self.is_convex is False and n_int == 2:
            for _s in self.segments:
                if _s.p1 == inters[0] and _s.p2 == inters[1]:
                    return self.is_point_inside(point, Vector2D(0, 1))
        if n_int % 2 == 0:
            return False

        n_int, inters = non_duplicate_intersect(Ray2D(point, Vector2D(0, 1)))
        # check that intersections do not form a polygon segment co-linear with test_ray
        if self.is_convex is False and n_int == 2:
            for _s in self.segments:
                if _s.p1 == inters[0] and _s.p2 == inters[1]:
                    return self.is_point_inside(point, Vector2D(1, 0))
        if n_int % 2 == 0:
            return False
        return True

    def is_point_inside(self, point, test_vector=Vector2D(1, 0.00001)):
        """Test whether a Point2D lies inside or outside the polygon.

        This method is the fastest way to tell if a point is inside a polygon when
        the given point lies inside the boundary rectangle of the polygon.
        However, while this method gives the correct result in 99.9% of cases,
        there are a few fringe cases where it will not give the correct result.
        Specifically these are:

        .. code-block:: shell

            1 - When the test_ray intersects perfectly with a polygon vertex.
                For example, this case with an X-unit (1, 0) test_vector:
                                    _____________
                                   |      .      |
                                   |            /
                                   |___________/
            2 - When there are two polygon vertices that are colinear with the point
                along the test_ray. For example, this case with an X-unit test_vector:
                                      _____
                                     |  .  |____
                                     |__________|

        Use the `is_point_inside_check` method if a result that covers these fringe
        cases is needed. Oftentimes, it is more practical to use a test_vector
        with a low probability of encountering the fringe cases than to use the
        (much longer) `is_point_inside_check` method.

        Args:
            point: A Point2D for which the inside/outside relationship will be tested.
            test_vector: Optional vector to set the direction in which intersections
                with the polygon edges will be evaluated to determine if the
                point is inside. Default is a slight variation of the X-unit
                vector with a low probability of encountering the unsupported
                fringe cases.

        Returns:
            A boolean denoting whether the point lies inside (True) or outside (False).
        """
        test_ray = Ray2D(point, test_vector)
        n_int = 0
        for _s in self.segments:
            if does_intersection_exist_line2d(_s, test_ray):
                n_int += 1
        if n_int % 2 == 0:
            return False
        return True

    def is_point_inside_bound_rect(self, point, test_vector=Vector2D(1, 0.00001)):
        """Test whether a Point2D lies roughly inside or outside the polygon.

        This function is virtually identical to the `is_point_inside`
        method but will first do a check to see if the point lies inside the
        polygon bounding rectangle. As such, it is a faster approach when one
        expects many of tested points to lie far away from the polygon.

        Args:
            point: A Point2D for which the inside/ outside relationship will be tested.
            test_vector: Optional vector to set the direction in which intersections
                with the polygon edges will be evaluated to determine if the
                point is inside. Default is the X unit vector.

        Returns:
            A boolean denoting whether the point lies inside (True) or outside (False).
        """
        min = self.min
        max = self.max
        if point.x < min.x or point.y < min.y or point.x > max.x or point.y > max.y:
            return False
        return self.is_point_inside(point, test_vector)

    def is_polygon_inside(self, polygon):
        """"Test whether another Polygon2D lies completely inside this polygon.

        Args:
            polygon: A Polygon2D to test whether it is completely inside this one.

        Returns:
            A boolean denoting whether the polygon lies inside (True) or not (False).
        """
        # if the first polygon vertex lies outside, we know it is not inside.
        if not self.is_point_inside_bound_rect(polygon[0]):
            return False
        # if one of the edges intersects, we know it has crossed outside.
        for seg in self.segments:
            for _s in polygon.segments:
                if does_intersection_exist_line2d(seg, _s):
                    return False
        return True

    def is_polygon_outside(self, polygon):
        """"Test whether another Polygon2D lies completely outside this polygon.

        Args:
            polygon: A Polygon2D to test whether it is completely outside this one.

        Returns:
            A boolean denoting whether the polygon lies outside (True) or not (False).
        """
        # if the first polygon vertex lies inside, we know it is not outside.
        if self.is_point_inside_bound_rect(polygon[0]):
            return False
        # if one of the edges intersects, we know it has crossed inside.
        for seg in self.segments:
            for _s in polygon.segments:
                if does_intersection_exist_line2d(seg, _s):
                    return False
        return True

    def polygon_relationship(self, polygon, tolerance):
        """Test whether another Polygon2D lies inside, outside or overlaps this one.

        This method is not usually the fastest for understanding the relationship
        between polygons but it accurately accounts for tolerance such that the
        case of the two polygons sharing edges will not determine the outcome.
        Only when the polygon has vertices that are truly outside this polygon
        within the tolerance will the relationship become outside (or intersecting
        if one of the vertices is already inside within the tolerance).

        In the case of the input polygon being identical to the current polygon,
        the relationship will be Inside.

        Args:
            polygon: A Polygon2D for which the relationship to the current polygon
                will be tested.
            tolerance: The minimum distance from the edge at which a point is
                considered to lie on the edge.

        Returns:
            An integer denoting the relationship of the polygon.

            This will be one of the following:

            * -1 = Outside this polygon
            *  0 = Overlaps (intersects or contains) this polygon
            * +1 = Inside this polygon
        """
        # first evaluate the point relationships to rule out the inside case
        pt_rels1 = [self.point_relationship(pt, tolerance) for pt in polygon]
        pt_rels2 = [polygon.point_relationship(pt, tolerance) for pt in self]
        if all(r1 >= 0 for r1 in pt_rels1) and all(r2 <= 0 for r2 in pt_rels2):
            poi = polygon.pole_of_inaccessibility(tolerance)
            if self.point_relationship(poi, tolerance) == 1:
                return 1  # definitely inside the polygon
        if 1 in pt_rels1 or 1 in pt_rels2:
            return 0  # definitely overlap in the polygons
        if all(r2 == 0 for r2 in pt_rels2):
            poi = self.pole_of_inaccessibility(tolerance)
            if polygon.point_relationship(poi, tolerance) == 1:
                return 0

        # offset one of the polygons inward by the tolerance
        off_poly = polygon.offset(tolerance)
        # if any of the offset segments intersect the other polygon, there is overlap
        for seg in self.segments:
            for _s in off_poly.segments:
                if does_intersection_exist_line2d(seg, _s):
                    return 0

        # we can reliably say that the polygons have nothing to do with one another
        return -1

    def distance_to_point(self, point):
        """Get the minimum distance between this shape and the input point.

        Points that are inside the Polygon2D will return a distance of zero.
        If the distance of an interior point to an edge is needed, the
        distance_from_edge_to_point method should be used.

        Args:
            point: A Point2D object to which the minimum distance will be computed.

        Returns:
            The distance to the input point. Will be zero if the point is
            inside the Polygon2D.
        """
        if self.is_point_inside_bound_rect(point):
            return 0
        return min(seg.distance_to_point(point) for seg in self.segments)

    def distance_from_edge_to_point(self, point):
        """Get the minimum distance between the edge of this shape and the input point.

        Args:
            point: A Point2D object to which the minimum distance will be computed.

        Returns:
            The distance to the input point. Will be zero if the point is
            inside the Polygon2D.
        """
        return min(seg.distance_to_point(point) for seg in self.segments)

    def snap_to_polygon(self, polygon, tolerance):
        """Snap another Polygon2D to this one for differences smaller than the tolerance.

        This is useful to run before performing operations where small tolerance
        differences are likely to cause issues, such as in boolean operations.

        Args:
            polygon: A Polygon2D which will be snapped to the current polygon.
            tolerance: The minimum distance at which points will be snapped.

        Returns:
            A version of the polygon that is snapped to this Polygon2D.
        """
        new_verts = []
        for pt in polygon.vertices:
            # first check if the point can be snapped to a vertex
            for s_pt in self.vertices:
                if pt.is_equivalent(s_pt, tolerance):
                    new_verts.append(s_pt)
                    break
            else:
                # check if the point can be snapped to a segment
                for seg in self.segments:
                    s_pt = seg.closest_point(pt)
                    if s_pt.distance_to_point(pt) <= tolerance:
                        new_verts.append(s_pt)
                        break
                else:  # point could not be snapped
                    new_verts.append(pt)
        return Polygon2D(new_verts)

    def to_dict(self):
        """Get Polygon2D as a dictionary."""
        return {'type': 'Polygon2D',
                'vertices': [pt.to_array() for pt in self.vertices]}

    def to_array(self):
        """Get a list of lists where each sub-list represents a Point2D vertex."""
        return tuple(pt.to_array() for pt in self.vertices)

    def _to_bool_poly(self):
        """A hidden method used to translate the Polygon2D to a BooleanPolygon.

        This is necessary before performing any boolean operations with
        the polygon.
        """
        b_pts = (pb.BooleanPoint(pt.x, pt.y) for pt in self.vertices)
        return pb.BooleanPolygon([b_pts])

    @staticmethod
    def _from_bool_poly(bool_polygon):
        """Get a list of Polygon2D from a BooleanPolygon object."""
        return [Polygon2D(tuple(Point2D(pt.x, pt.y) for pt in new_poly))
                for new_poly in bool_polygon.regions if len(new_poly) > 2]

    def boolean_union(self, polygon, tolerance):
        """Get a list of Polygon2D for the union of this Polygon and another.

        Note that the result will not differentiate hole polygons from boundary
        polygons and so it may be desirable to use the Polygon2D.is_polygon_inside
        method to distinguish whether a given polygon in the result represents
        a hole in another polygon in the result.

        Also note that this method will return the original polygons when there
        is no overlap in the two.

        Args:
            polygon: A Polygon2D for which the union with the current polygon
                will be computed.
            tolerance: The minimum distance between points before they are
                considered distinct from one another.

        Returns:
            A list of Polygon2D representing the union of the two polygons.
        """
        result = pb.union(
            self._to_bool_poly(), polygon._to_bool_poly(), tolerance)
        return Polygon2D._from_bool_poly(result)

    def boolean_intersect(self, polygon, tolerance):
        """Get a list of Polygon2D for the intersection of this Polygon and another.

        Args:
            polygon: A Polygon2D for which the intersection with the current polygon
                will be computed.
            tolerance: The minimum distance between points before they are
                considered distinct from one another.

        Returns:
            A list of Polygon2D representing the intersection of the two polygons.
            Will be an empty list if no overlap exists between the polygons.
        """
        result = pb.intersect(
            self._to_bool_poly(), polygon._to_bool_poly(), tolerance)
        return Polygon2D._from_bool_poly(result)

    def boolean_difference(self, polygon, tolerance):
        """Get a list of Polygon2D for the subtraction of another polygon from this one.

        Args:
            polygon: A Polygon2D for which the difference with the current polygon
                will be computed.
            tolerance: The minimum distance between points before they are
                considered distinct from one another.

        Returns:
            A list of Polygon2D representing the difference of the two polygons.
            Will be an empty list if subtracting the polygons results in the complete
            elimination of this polygon. Will be the original polygon when there
            is no overlap between the polygons.
        """
        result = pb.difference(
            self._to_bool_poly(), polygon._to_bool_poly(), tolerance)
        return Polygon2D._from_bool_poly(result)

    def boolean_xor(self, polygon, tolerance):
        """Get Polygon2D list for the exclusive disjunction of this polygon and another.

        Note that this method is prone to merging holes that may exist in the
        result into the boundary to create a single list of joined vertices,
        which may not always be desirable. In this case, it may be desirable
        to do two separate boolean_difference calculations instead.

        Also note that, when the result includes separate polygons for holes,
        it will not differentiate hole polygons from boundary polygons
        and so it may be desirable to use the Polygon2D.is_polygon_inside
        method to distinguish whether a given polygon in the result represents
        a hole within another polygon in the result.

        Args:
            polygon: A Polygon2D for which the exclusive disjunction with the
                current polygon will be computed.
            tolerance: The minimum distance between points before they are
                considered distinct from one another.

        Returns:
            A list of Polygon2D representing the exclusive disjunction of the
            two polygons. Will be the original polygons when there is no overlap
            in the two.
        """
        result = pb.xor(
            self._to_bool_poly(), polygon._to_bool_poly(), tolerance)
        return Polygon2D._from_bool_poly(result)

    @staticmethod
    def boolean_union_all(polygons, tolerance):
        """Get a list of Polygon2D for the union of several Polygon2D.

        Using this method is more computationally efficient than calling the
        Polygon2D.boolean_union() method multiple times as this method will
        only compute the intersection of the segments once.

        Note that the result will not differentiate hole polygons from boundary
        polygons and so it may be desirable to use the Polygon2D.is_polygon_inside
        method to distinguish whether a given polygon in the result represents
        a hole in another polygon in the result.

        Also note that this method will return the original polygons when there
        is no overlap in the two.

        Args:
            polygons: An array of Polygon2D for which the union will be computed.
            tolerance: The minimum distance between points before they are
                considered distinct from one another.

        Returns:
            A list of Polygon2D representing the union of all the polygons.
        """
        bool_polys = [poly._to_bool_poly() for poly in polygons]
        result = pb.union_all(bool_polys, tolerance)
        return Polygon2D._from_bool_poly(result)

    @staticmethod
    def boolean_intersect_all(polygons, tolerance):
        """Get a list of Polygon2D for the intersection of several Polygon2D.

        Using this method is more computationally efficient than calling the
        Polygon2D.boolean_intersect() method multiple times as this method will
        only compute the intersection of the segments once.

        Note that the result will not differentiate hole polygons from boundary
        polygons and so it may be desirable to use the Polygon2D.is_polygon_inside
        method to distinguish whether a given polygon in the result represents
        a hole in another polygon in the result.

        Also note that this method will return the original polygons when there
        is no overlap in the two.

        Args:
            polygons: An array of Polygon2D for which the intersection will be computed.
            tolerance: The minimum distance between points before they are
                considered distinct from one another.

        Returns:
            A list of Polygon2D representing the intersection of all the polygons.
            Will be an empty list if no overlap exists between the polygons.
        """
        bool_polys = [poly._to_bool_poly() for poly in polygons]
        result = pb.intersect_all(bool_polys, tolerance)
        return Polygon2D._from_bool_poly(result)

    @staticmethod
    def boolean_split(polygon1, polygon2, tolerance):
        """Split two Polygon2D with one another to get the intersection and difference.

        Using this method is more computationally efficient than calling the
        Polygon2D.intersect() and Polygon2D.difference() methods individually as
        this method will only compute the intersection of the segments once.

        Note that the result will not differentiate hole polygons from boundary
        polygons and so it may be desirable to use the Polygon2D.is_polygon_inside
        method to distinguish whether a given polygon in the result represents
        a hole in another polygon in the result.

        Args:
            polygon1: A Polygon2D for the first polygon that will be split with
                the second polygon.
            polygon2: A Polygon2D for the second polygon that will be split with
                the first polygon.
            tolerance: The minimum distance between points before they are
                considered distinct from one another.

        Returns:
            A tuple with three elements

        -   intersection: A list of Polygon2D for the intersection of the two
            input polygons.

        -   poly1_difference: A list of Polygon2D for the portion of polygon1 that does
            not overlap with polygon2. When combined with the intersection, this
            makes a split version of polygon1.

        -   poly2_difference: A list of Polygon2D for the portion of polygon2 that does
            not overlap with polygon1. When combined with the intersection, this
            makes a split version of polygon2.
        """
        int_result, poly1_result, poly2_result = pb.split(
            polygon1._to_bool_poly(), polygon2._to_bool_poly(), tolerance)
        intersection = Polygon2D._from_bool_poly(int_result)
        poly1_difference = Polygon2D._from_bool_poly(poly1_result)
        poly2_difference = Polygon2D._from_bool_poly(poly2_result)
        return intersection, poly1_difference, poly2_difference

    @staticmethod
    def intersect_polygon_segments(polygon_list, tolerance):
        """Intersect the line segments of a Polygon2D array to ensure matching segments.

        Specifically, this method checks a list of polygons in a pairwise manner to
        see if one contains a vertex along an edge segment of the other within the
        given tolerance. If so, the method creates a co-located vertex at that point,
        partitioning the edge segment into two edge segments. Point ordering is
        reserved within each Polygon2D and the order of Polygon2Ds within the input
        polygon_list is also preserved.

        Args:
            polygon_list: List of Polygon2Ds which will have their segments
                intersected with one another.
            tolerance: Distance within which two points are considered to be
                co-located.

        Returns:
            The input list of Polygon2D objects with extra vertices inserted
                where necessary.
        """
        for i in range(len(polygon_list) - 1):
            # No need for j to start at 0 since two polygons are passed
            # and they are compared against one other within intersect_segments.
            for j in range(i + 1, len(polygon_list)):
                polygon_list[i], polygon_list[j] = \
                    Polygon2D.intersect_segments(polygon_list[i], polygon_list[j],
                                                 tolerance)
        return polygon_list

    @staticmethod
    def intersect_segments(polygon1, polygon2, tolerance):
        """Intersect the line segments of two Polygon2Ds to ensure matching segments.

        Specifically, this method checks two adjacent polygons to see if one contains
        a vertex along an edge segment of the other within the given tolerance. If so,
        it creates a co-located vertex at that point, partitioning the edge segment
        into two edge segments. Point ordering is preserved.

        Args:
            polygon1: First polygon to check.
            polygon2: Second polygon to check.
            tolerance: Distance within which two points are considered to be co-located.

        Returns:
            Two polygon objects with extra vertices inserted if necessary.
        """
        polygon1_updates = []
        polygon2_updates = []

        # bounding rectangle check
        if not Polygon2D.overlapping_bounding_rect(polygon1, polygon2, tolerance):
            return polygon1, polygon2  # no overlap

        # test if each point of polygon2 is within the tolerance distance of any segment
        # of polygon1.  If so, add the closest point on the segment to the polygon1
        # update list. And vice versa (testing polygon2 against polygon1).
        for i1, seg1 in enumerate(polygon1.segments):
            for i2, seg2 in enumerate(polygon2.segments):
                # Test polygon1 against polygon2
                x = closest_point2d_on_line2d(seg2.p1, seg1)
                if all(p.distance_to_point(x) > tolerance for p in polygon1.vertices) \
                        and x.distance_to_point(seg2.p1) <= tolerance:
                    polygon1_updates.append((i1, x))
                # Test polygon2 against polygon1
                y = closest_point2d_on_line2d(seg1.p1, seg2)
                if all(p.distance_to_point(y) > tolerance for p in polygon2.vertices) \
                        and y.distance_to_point(seg1.p1) <= tolerance:
                    polygon2_updates.append((i2, y))

        # apply any updates to polygon1
        polygon1 = Polygon2D._insert_updates_in_order(polygon1, polygon1_updates)

        # Apply any updates to polygon2
        polygon2 = Polygon2D._insert_updates_in_order(polygon2, polygon2_updates)

        return polygon1, polygon2

    @staticmethod
    def overlapping_bounding_rect(polygon1, polygon2, tolerance):
        """Check if the bounding rectangles of two polygons overlap within a tolerance.

        This is particularly useful as a check before performing computationally intense
        processes between two polygons like intersection or boolean operations.
        Checking the overlap of the bounding boxes is extremely quick with this
        method's use of the the Separating Axis Theorem.

        Args:
            polygon1: The first polygon to check.
            polygon2: The second polygon to check.
            tolerance: Distance within which two points are considered to be co-located.
        """
        # Bounding rectangle check using the Separating Axis Theorem
        polygon1_width = polygon1.max.x - polygon1.min.x
        polygon2_width = polygon2.max.x - polygon2.min.x
        dist_btwn_x = abs(polygon1.center.x - polygon2.center.x)
        x_gap_btwn_rect = dist_btwn_x - (0.5 * polygon1_width) - (0.5 * polygon2_width)

        polygon1_height = polygon1.max.y - polygon1.min.y
        polygon2_height = polygon2.max.y - polygon2.min.y
        dist_btwn_y = abs(polygon1.center.y - polygon2.center.y)
        y_gap_btwn_rect = dist_btwn_y - (0.5 * polygon1_height) - (0.5 * polygon2_height)

        if x_gap_btwn_rect > tolerance or y_gap_btwn_rect > tolerance:
            return False  # no overlap
        return True  # overlap exists

    def _transfer_properties(self, new_polygon):
        """Transfer properties from this polygon to a new polygon.

        This is used by the transform methods that don't alter the relationship of
        face vertices to one another (move, rotate, reflect).
        """
        new_polygon._perimeter = self._perimeter
        new_polygon._area = self._area
        new_polygon._is_convex = self._is_convex
        new_polygon._is_self_intersecting = self._is_self_intersecting
        new_polygon._is_clockwise = self._is_clockwise

    @staticmethod
    def _get_centroid_cell(polygon):
        """Get a Cell object at the centroid of the Polygon2D."""
        area = 0
        x = 0
        y = 0
        b = polygon[-1]  # prev
        for a in polygon:
            f = a[0] * b[1] - b[0] * a[1]
            x += (a[0] + b[0]) * f
            y += (a[1] + b[1]) * f
            area += f * 3
            b = a
        if area == 0:
            return _Cell(polygon[0][0], polygon[0][1], 0, polygon)
        return _Cell(x / area, y / area, 0, polygon)

    @staticmethod
    def _insert_updates_in_order(polygon, polygon_updates):
        """Insert updates from the intersect_segments method into a polygon.

        This method ensures that multiple updates to a single segment are inserted
        in the correct order over the polygon.

        Args:
            polygon: A Polygon2D to be updated with intersection points
            polygon_updates: A list of tuples where each tuple has two values. The
                first is the index of the segment to be updated and the second is
                the point to insert.
        """
        polygon_updates.sort(key=lambda x: x[0])  # sort updates by order of insertion
        poly_points = list(polygon.vertices)  # convert tuple to mutable list
        last_i = -1
        colinear_count = 0
        for update in polygon_updates[::-1]:  # traverse backwards to preserve order
            new_i = update[0] + 1
            if new_i == last_i:  # check order of new intersections on the same segment
                colinear_count += 1
                p1 = poly_points[update[0]]
                for i, pt in enumerate(poly_points[new_i:new_i + colinear_count]):
                    if p1.distance_to_point(pt) > p1.distance_to_point(update[1]):
                        poly_points.insert(new_i + i, update[1])
                        break
                else:
                    poly_points.insert(new_i + colinear_count, update[1])
            else:
                colinear_count = 0
                poly_points.insert(new_i, update[1])
            last_i = new_i
        return Polygon2D(poly_points)

    @staticmethod
    def _segments_from_vertices(vertices):
        _segs = []
        for i, vert in enumerate(vertices):
            _seg = LineSegment2D.from_end_points(vertices[i - 1], vert)
            _segs.append(_seg)
        _segs.append(_segs.pop(0))  # segments will start from the first point
        return _segs

    @staticmethod
    def _merge_boundary_and_holes(boundary, holes, split=False):
        """Return a list of points for a boundary merged with all holes.

        The holes are merged one-by-one using the shortest distance from all of
        the holes to the boundary to ensure one hole's seam does not cross another.
        This run time of this method scales linearly with the number of hole
        vertices, which makes it significantly better for shapes with many holes
        compared to recursively calling the _merge_boundary_and_closest_hole
        method.

        Args:
            boundary: A list of Point2D objects for the outer boundary inside of
                which the hole is contained.
            hole: A list of lists where each sub-list represents a hole and contains
                several Point2D objects that represent the hole.
            split: A boolean to note whether the last hole should be merged into
                the boundary for a second time, effectively splitting the shape
                into two lists of vertices instead of a single list. It is useful
                to set this tro True when trying to translate a shape with holes
                to a platform that does not support holes and also struggles with
                single lists of vertices that wind inward to cut out the holes
                since this option typically returns two "normal" concave
                polygons. (Default: False).

        Returns:
            A single list of vertices with the holes merged into the boundary. When
            split is True, this will be two lists of vertices for the two split shapes.
        """
        # compute the initial distances between the holes and the boundary
        hole_dicts = []
        min_dists = []
        for hole in holes:
            dist_dict = {}
            for i, b_pt in enumerate(boundary):
                for j, h_pt in enumerate(hole):
                    dist_dict[b_pt.distance_to_point(h_pt)] = [i, j]
            hole_dicts.append(dist_dict)
            min_dists.append(min(dist_dict.keys()))
        # merge each hole into the boundary, moving progressively by minimum distance
        final_hole_count = 0 if not split else 1
        while len(holes) > final_hole_count:
            # merge the hole into the boundary
            hole_index = min_dists.index(min(min_dists))
            boundary, old_hole, b_ind = Polygon2D._merge_boundary_and_hole_detailed(
                boundary, holes[hole_index], hole_dicts[hole_index])
            # remove the hole from the older lists
            hole_dicts.pop(hole_index)
            holes.pop(hole_index)
            # update the hole_dicts based on the new boundary
            add_ind = len(old_hole)
            for hd in hole_dicts:
                for ind_list in hd.values():
                    if ind_list[0] > b_ind:
                        ind_list[0] += add_ind
            # add the distances from the old hole to the remaining holes to hole_dicts
            old_hole_ind = [b_ind + i for i in range(add_ind)]
            for hole, dist_dict in zip(holes, hole_dicts):
                for bi, b_pt in zip(old_hole_ind, old_hole):
                    for j, h_pt in enumerate(hole):
                        dist_dict[b_pt.distance_to_point(h_pt)] = [bi, j]
            # generated an updated min_dists list
            min_dists = [min(dist_dict.keys()) for dist_dict in hole_dicts]
        if not split:
            return boundary

        # sort the distances to find the first and second most distant points
        last_hd = hole_dicts[0]
        sort_dist = sorted(last_hd.keys())
        p1_dist, p2_dist = sort_dist[0], sort_dist[1]
        p1_indices, p2_indices = dist_dict[p1_dist], dist_dict[p2_dist]
        # keep track of the second most distant points for later
        p2_bound_pt = boundary[p2_indices[0]]
        p2_hole_pt = hole[p2_indices[1]]
        # merge the hole into the boundary
        hole_deque = deque(hole)
        hole_deque.rotate(-p1_indices[1])
        hole_insert = [boundary[p1_indices[0]]] + list(hole_deque) + \
            [hole[p1_indices[1]]]
        boundary[p1_indices[0]:p1_indices[0]] = hole_insert
        # use the second most distant points to split the shape\
        p2_bound_i = boundary.index(p2_bound_pt)
        p2_hole_i = boundary.index(p2_hole_pt)
        if p2_hole_i < p2_bound_i:
            boundary_1 = boundary[p2_hole_i:p2_bound_i + 1]
            boundary_2 = boundary[:p2_hole_i + 1] + boundary[p2_bound_i:]
        else:
            boundary_1 = boundary[p2_bound_i:p2_hole_i + 1]
            boundary_2 = boundary[:p2_bound_i + 1] + boundary[p2_hole_i:]
        return boundary_1, boundary_2

    @staticmethod
    def _merge_boundary_and_hole_detailed(boundary, hole, dist_dict):
        """Create a single list of points with a hole and boundary.

        This method will also return the newly-added vertices of the hole as
        well as the index of where the hole was inserted in the larger boundary.

        Args:
            boundary: A list of Point2D objects for the outer boundary inside of
                which the hole is contained.
            hole: A list of Point2D objects for the hole.
            dist_dict: A dictionary with keys of distances between each of the points
                in the boundary and hole lists and values as tuples with two values:
                (the index of the boundary point, the index of the hole point)

        Returns:
            A tuple with three values

            -   boundary: A single list of vertices with the input hole merged
                into the boundary.

            -   hole_insert: A list of vertices representing the hole that has
                been inserted into the boundary.

            -   insert_index: An integer for where the hole was inserted in the
                boundary.
        """
        min_dist = min(dist_dict.keys())
        min_indexes = dist_dict[min_dist]
        hole_deque = deque(hole)
        hole_deque.rotate(-min_indexes[1])
        hole_insert = [boundary[min_indexes[0]]] + list(hole_deque) + \
            [hole[min_indexes[1]]]
        boundary[min_indexes[0]:min_indexes[0]] = hole_insert
        insert_index = min_indexes[0]
        return boundary, hole_insert, insert_index

    @staticmethod
    def _merge_boundary_and_closest_hole(boundary, holes):
        """Return a list of points for a boundary merged with the closest hole."""
        hole_dicts = []
        min_dists = []
        for hole in holes:
            dist_dict = {}
            for i, b_pt in enumerate(boundary):
                for j, h_pt in enumerate(hole):
                    dist_dict[b_pt.distance_to_point(h_pt)] = (i, j)
            hole_dicts.append(dist_dict)
            min_dists.append(min(dist_dict.keys()))
        hole_index = min_dists.index(min(min_dists))
        new_boundary = Polygon2D._merge_boundary_and_hole(
            boundary, holes[hole_index], hole_dicts[hole_index])
        holes.pop(hole_index)
        return new_boundary, holes

    @staticmethod
    def _merge_boundary_and_hole(boundary, hole, dist_dict):
        """Create a single list of points describing a boundary shape with a hole.

        Args:
            boundary: A list of Point2D objects for the outer boundary inside of
                which the hole is contained.
            hole: A list of Point2D objects for the hole.
            dist_dict: A dictionary with keys of distances between each of the points
                in the boundary and hole lists and values as tuples with two values:
                (the index of the boundary point, the index of the hole point)
        """
        min_dist = min(dist_dict.keys())
        min_indexes = dist_dict[min_dist]
        hole_deque = deque(hole)
        hole_deque.rotate(-min_indexes[1])
        hole_insert = [boundary[min_indexes[0]]] + list(hole_deque) + \
            [hole[min_indexes[1]]]
        boundary[min_indexes[0]:min_indexes[0]] = hole_insert
        return boundary

    @staticmethod
    def _are_clockwise(vertices):
        """Check if a list of vertices are clockwise.

        This is a quicker calculation when all you need is the direction and not area.
        """
        _a = 0
        for i, pt in enumerate(vertices):
            _a += vertices[i - 1].x * pt.y - vertices[i - 1].y * pt.x
        return _a < 0

    def __copy__(self):
        _new_poly = Polygon2D(self._vertices)
        _new_poly._segments = self._segments
        _new_poly._perimeter = self._perimeter
        _new_poly._area = self._area
        _new_poly._is_clockwise = self._is_clockwise
        _new_poly._is_convex = self._is_convex
        _new_poly._is_self_intersecting = self._is_self_intersecting
        return _new_poly

    def __key(self):
        """A tuple based on the object properties, useful for hashing."""
        return tuple(hash(pt) for pt in self._vertices)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(other, Polygon2D) and self.__key() == other.__key()

    def __repr__(self):
        return 'Polygon2D ({} vertices)'.format(len(self))


class _Cell(object):
    """2D cell object used in certain Polygon computations (eg. pole_of_inaccessibility).

    Args:
        x: The X coordinate of the cell origin.
        y: The Y coordinate of the cell origin.
        h: The dimension of the cell.
        polygon: An array representation of a Polygon2D.

    Properties:
        * x
        * y
        * h
        * d
        * max
    """
    __slots__ = ('x', 'y', 'h', 'd', 'max')

    def __init__(self, x, y, h, polygon):
        self.h = h
        self.y = y
        self.x = x
        self.d = self._point_to_polygon_distance(x, y, polygon)
        self.max = self.d + self.h * math.sqrt(2)

    def _point_to_polygon_distance(self, x, y, polygon):
        """Get the distance from an X,Y point to the edge of a Polygon."""
        inside = False
        min_dist_sq = inf

        b = polygon[-1]
        for a in polygon:

            if (a[1] > y) != (b[1] > y) and \
                    (x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]):
                inside = not inside

            min_dist_sq = min(min_dist_sq, self._get_seg_dist_sq(x, y, a, b))
            b = a

        result = math.sqrt(min_dist_sq)
        if not inside:
            return -result
        return result

    @staticmethod
    def _get_seg_dist_sq(px, py, a, b):
        """Get the squared distance from a point to a segment."""
        x = a[0]
        y = a[1]
        dx = b[0] - x
        dy = b[1] - y

        if dx != 0 or dy != 0:
            t = ((px - x) * dx + (py - y) * dy) / (dx * dx + dy * dy)

            if t > 1:
                x = b[0]
                y = b[1]

            elif t > 0:
                x += dx * t
                y += dy * t

        dx = px - x
        dy = py - y

        return dx * dx + dy * dy

    def __lt__(self, other):
        return self.max < other.max

    def __lte__(self, other):
        return self.max <= other.max

    def __gt__(self, other):
        return self.max > other.max

    def __gte__(self, other):
        return self.max >= other.max

    def __eq__(self, other):
        return self.max == other.max
