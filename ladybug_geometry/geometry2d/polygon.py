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
from .polyline import Polyline2D
from ..triangulation import _linked_list, _eliminate_holes
from ..intersection2d import intersect_line2d, intersect_line2d_infinite, \
    does_intersection_exist_line2d, closest_point2d_on_line2d, \
    closest_end_point2d_between_line2d, closest_point2d_on_line2d_infinite
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
        * inside_angles
        * outside_angles
        * min
        * max
        * center
        * perimeter
        * area
        * is_clockwise
        * is_convex
        * is_self_intersecting
        * self_intersection_points
        * is_valid
    """
    __slots__ = ('_segments', '_inside_angles', '_outside_angles', '_perimeter', '_area',
                 '_is_clockwise', '_is_convex', '_is_self_intersecting')

    def __init__(self, vertices):
        """Initialize Polygon2D."""
        Base2DIn2D.__init__(self, vertices)
        self._segments = None
        self._perimeter = None
        self._inside_angles = None
        self._outside_angles = None
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
    def inside_angles(self):
        """Tuple of angles in radians for the interior angles of the polygon.

        These are aligned with the vertices such that the first angle corresponds
        to the inside angle at the first vertex, the second at the second vertex,
        and so on.
        """
        if self._inside_angles is None:
            angles, is_clock = [], self.is_clockwise
            for i, pt in enumerate(self.vertices):
                v1 = self.vertices[i - 2] - self.vertices[i - 1]
                v2 = pt - self.vertices[i - 1]
                v_angle = v1.angle_counterclockwise(v2) if is_clock \
                    else v1.angle_clockwise(v2)
                angles.append(v_angle)
            angles.append(angles.pop(0))  # move the start angle to the end
            self._inside_angles = tuple(angles)
        return self._inside_angles

    @property
    def outside_angles(self):
        """Tuple of angles in radians for the exterior angles of the polygon.

        These are aligned with the vertices such that the first angle corresponds
        to the inside angle at the first vertex, the second at the second vertex,
        and so on.
        """
        if self._outside_angles is None:
            pi2 = math.pi * 2
            self._outside_angles = tuple(pi2 - a for a in self.inside_angles)
        return self._outside_angles

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

        Note that this property is relatively computationally intense to obtain compared
        to properties like area and is_convex. Also, most CAD programs forbid geometry
        with self-intersecting edges. So it is recommended that this property only
        be used in quality control scripts where the origin of the geometry is unknown.
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
    def self_intersection_points(self):
        """A tuple of Point2Ds for the locations where the polygon intersects itself.

        This will be an empty tuple if the polygon is not self-intersecting and it
        is generally recommended that the Polygon2D.is_self_intersecting property
        be checked before using this property.
        """
        if self.is_self_intersecting:
            int_pts = []
            _segs = self.segments
            for i, _s in enumerate(_segs[1: len(_segs) - 1]):
                _skip = (i, i + 1, i + 2)
                _other_segs = [x for j, x in enumerate(_segs) if j not in _skip]
                for _oth_s in _other_segs:
                    int_pt = _s.intersect_line_ray(_oth_s)
                    if int_pt is not None:  # intersection!
                        int_pts.append(int_pt)
            return tuple(int_pts)
        return ()

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

    def is_rectangle(self, angle_tolerance):
        """Test whether this Polygon2D is a rectangle given an angle tolerance.

        Note that this method will return False if the Polygon2D does not have
        four vertices, even if they are duplicated or colinear.

        Args:
            angle_tolerance: The max angle in radians that the corners of the
                rectangle can differ from a right angle before it is not
                considered a rectangle.
        """
        if len(self.vertices) != 4:
            return False
        min_ang = (math.pi / 2) - angle_tolerance
        max_ang = (math.pi / 2) + angle_tolerance
        for i, pt in enumerate(self.vertices):
            v1 = self.vertices[i - 2] - self.vertices[i - 1]
            v2 = pt - self.vertices[i - 1]
            v_angle = v1.angle(v2)
            if v_angle < min_ang or v_angle > max_ang:
                return False
        return True

    def rectangular_approximation(self):
        """Get a rectangular Polygon2D with the same area and aspect ratio as this one.

        This is useful when an interface requires a rectangular input but the
        user-defined geometry can be any shape. The resulting rectangle will share
        the same center as this one.
        """
        b_rect_len = self.max.x - self.min.x
        b_rect_hgt = self.max.y - self.min.y
        aspect_ratio = b_rect_len / b_rect_hgt
        final_hgt = math.sqrt(self.area / aspect_ratio)
        final_len = self.area / final_hgt
        b_pt = Point2D(self.center.x - (final_len / 2), self.center.y - (final_hgt / 2))
        return Polygon2D.from_rectangle(b_pt, Vector2D(0, 1), final_len, final_hgt)

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
        max_dim = max(width, height)
        if cell_size == 0 or self.area < max_dim * tolerance:
            # degenerate polygon; just return the center
            return self.center

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

    def remove_duplicate_vertices(self, tolerance):
        """Get a version of this polygon without duplicate vertices.

        Args:
            tolerance: The minimum distance between a two vertices at which
                they are considered co-located or duplicated.
        """
        new_vertices = tuple(
            pt for i, pt in enumerate(self._vertices)
            if not pt.is_equivalent(self._vertices[i - 1], tolerance))
        return Polygon2D(new_vertices)

    def remove_colinear_vertices(self, tolerance):
        """Get a version of this polygon without colinear or duplicate vertices.

        Args:
            tolerance: The minimum distance that a vertex can be from a line
                before it is considered colinear.
        """
        if len(self.vertices) == 3:
            return self  # Polygon2D cannot have fewer than 3 vertices
        new_vertices = []  # list to hold the new vertices
        skip = 0  # track the number of vertices being skipped/removed
        # loop through vertices and remove all cases of colinear verts
        for i, _v in enumerate(self.vertices):
            _a = self[i - 2 - skip].determinant(self[i - 1]) + self[i - 1].determinant(_v) + \
                _v.determinant(self[i - 2 - skip])
            if abs(_a) >= tolerance:
                new_vertices.append(self[i - 1])
                skip = 0
            else:
                skip += 1
        # catch case of last two vertices being equal but distinct from first point
        if skip != 0 and self.vertices[-2].is_equivalent(self.vertices[-1], tolerance):
            pts_2d = self.vertices
            _a = pts_2d[-3].determinant(pts_2d[-1]) + \
                pts_2d[-1].determinant(pts_2d[0]) + pts_2d[0].determinant(pts_2d[-3])
            if abs(_a) >= tolerance:
                new_vertices.append(pts_2d[-1])
        return Polygon2D(new_vertices)

    def split_through_self_intersection(self, tolerance):
        """Get a list of non-intersecting Polygon2D if this polygon intersects itself.

        If the Polygon2D does not intersect itself, then a list with the current
        Polygon2D will be returned.

        Args:
            tolerance: The minimum difference between vertices before they are
                considered co-located.
        """
        # loop over the segments and group the vertices by intersection points
        intersect_groups = [[]]
        _segs = self.segments
        seg_count = len(_segs)
        for i, _s in enumerate(_segs):
            # loop over the other segments and find any intersection points
            if i == 0:
                _skip = (len(_segs) - 1, i, i + 1) 
            elif i == seg_count - 1:
                _skip = (i - 1, i, 0)
            else:
                _skip = (i - 1, i, i + 1)
            _other_segs = [x for j, x in enumerate(_segs) if j not in _skip]
            int_pts = []
            for _oth_s in _other_segs:
                int_pt = _s.intersect_line_ray(_oth_s)
                if int_pt is not None:  # intersection!
                    int_pts.append(int_pt)
            # if intersection points were found, adjust the groups accordingly
            if len(int_pts) == 0:  # no self intersection on this segment
                intersect_groups[-1].append(_s.p2)
            elif len(int_pts) == 1:  # typical self-intersection case we should split
                intersect_groups[-1].append(int_pts[0])
                intersect_groups.append([_s.p2])
            else:  # rare case of multiple intersections on the same segment
                # sort the intersection points along the segment
                dists = [_s.p1.distance_to_point(ipt) for ipt in int_pts]
                sort_pts = [pt for _, pt in sorted(zip(dists, int_pts),
                                                   key=lambda pair: pair[0])]
                intersect_groups[-1].append(sort_pts[0])
                for s_pt in sort_pts[1:]:
                    intersect_groups.append([s_pt])
                intersect_groups.append([_s.p2])

        # process the intersect groups into polygon objects
        if len(intersect_groups) == 1:
            return [self]  # not a self-intersecting shape
        split_polygons = []
        poly_count = int(len(intersect_groups) / 2)
        if len(intersect_groups[poly_count]) == 1:  # rare case of start at intersect
            for i in range(poly_count):
                vert_group = [intersect_groups[i], intersect_groups[-i - 1]]
                for verts_list in vert_group:
                    if len(verts_list) > 2:
                        try:
                            clean_poly = Polygon2D(verts_list)
                            clean_poly = clean_poly.remove_duplicate_vertices(tolerance)
                            split_polygons.append(clean_poly)
                        except AssertionError:  # degenerate polygon that should not be added
                            pass
        else:  # typical case of intersection in the middle
            for i in range(poly_count):
                verts_list = intersect_groups[i] + intersect_groups[-i - 1]
                if len(verts_list) > 2:
                    try:
                        clean_poly = Polygon2D(verts_list)
                        clean_poly = clean_poly.remove_duplicate_vertices(tolerance)
                        split_polygons.append(clean_poly)
                    except AssertionError:  # degenerate polygon that should not be added
                        pass
            final_verts = intersect_groups[i + 1]
            try:
                clean_poly = Polygon2D(final_verts)
                clean_poly = clean_poly.remove_duplicate_vertices(tolerance)
                split_polygons.append(clean_poly)
            except AssertionError:  # degenerate polygon that should not be added
                pass
        return split_polygons

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
                returned instead of the self-intersecting polygon.
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
        """Test whether another Polygon2D lies completely inside this polygon.

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
        """Test whether another Polygon2D lies completely outside this polygon.

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

    def does_polygon_touch(self, polygon, tolerance):
        """Test whether another Polygon2D touches, overlaps or is inside this polygon.

        Args:
            polygon: A Polygon2D to test whether it touches this polygon.
            tolerance: The minimum distance from an edge at which a point is
                considered to touch the edge.

        Returns:
            A boolean denoting whether the polygon touches (True) or not (False).
        """
        # perform a bounding rectangle check to see if the polygons cannot overlap
        if not Polygon2D.overlapping_bounding_rect(self, polygon, tolerance):
            return False

        # first evaluate the point relationships
        pt_rels1 = [self.point_relationship(pt, tolerance) for pt in polygon]
        if 0 in pt_rels1 or 1 in pt_rels1:
            return True  # definitely touching polygons
        pt_rels2 = [polygon.point_relationship(pt, tolerance) for pt in self]
        if 0 in pt_rels2 or 1 in pt_rels2:
            return True  # definitely touching polygons

        # if any of the segments intersect the other polygon, there is overlap
        for seg in self.segments:
            for _s in polygon.segments:
                if does_intersection_exist_line2d(seg, _s):
                    return True

        # we can reliably say that the polygons do not touch
        return False

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
            *  0 = Overlaps (intersects) this polygon
            * +1 = Inside this polygon
        """
        # perform a bounding rectangle check to see if the polygons cannot overlap
        if not Polygon2D.overlapping_bounding_rect(self, polygon, tolerance):
            return -1

        # first evaluate the point relationships to rule out the inside case
        pt_rels1 = [self.point_relationship(pt, tolerance) for pt in polygon]
        pt_rels2 = [polygon.point_relationship(pt, tolerance) for pt in self]
        if all(r1 >= 0 for r1 in pt_rels1) and all(r2 <= 0 for r2 in pt_rels2):
            poi = polygon._point_in_polygon(tolerance)
            if self.is_point_inside(poi) == 1:
                return 1  # definitely inside the polygon
        if 1 in pt_rels1 or 1 in pt_rels2:
            return 0  # definitely overlap in the polygons
        if all(r2 == 0 for r2 in pt_rels2):
            poi = self._point_in_polygon(tolerance)
            if polygon.is_point_inside(poi) == 1:
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
        """Get the minimum distance between this shape and a point.

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

    def snap_to_grid(self, grid_increment):
        """Snap this polygon's vertices to the nearest grid node defined by an increment.

        Args:
            grid_increment: A positive number for dimension of each grid cell. This
                typically should be equal to the tolerance or larger but should
                not be larger than the smallest detail of the polygon that you
                wish to resolve.

        Returns:
            A version of this polygon that is snapped to the grid.
        """
        new_verts = []
        for pt in self.vertices:
            new_x = grid_increment * round(pt.x / grid_increment)
            new_y = grid_increment * round(pt.y / grid_increment)
            new_verts.append(Point2D(new_x, new_y))
        return Polygon2D(new_verts)

    def to_dict(self):
        """Get Polygon2D as a dictionary."""
        return {'type': 'Polygon2D',
                'vertices': [pt.to_array() for pt in self.vertices]}

    def to_array(self):
        """Get a tuple of tuples where each sub-tuple represents a Point2D vertex."""
        return tuple(pt.to_array() for pt in self.vertices)

    def _to_bool_poly(self):
        """Translate the Polygon2D to a BooleanPolygon."""
        b_pts = (pb.BooleanPoint(pt.x, pt.y) for pt in self.vertices)
        return pb.BooleanPolygon([b_pts])

    def _to_snapped_bool_poly(self, snap_ref_polygon, tolerance):
        """Snap a Polygon2D to this one and translate it to a BooleanPolygon.

        This is necessary to ensure that boolean operations will succeed between
        two polygons.
        """
        new_poly = snap_ref_polygon.snap_to_polygon(self, tolerance)
        return new_poly._to_bool_poly()

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
            self._to_bool_poly(),
            polygon._to_snapped_bool_poly(self, tolerance),
            tolerance / 100)
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
            self._to_bool_poly(),
            polygon._to_snapped_bool_poly(self, tolerance),
            tolerance / 100)
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
            self._to_bool_poly(),
            polygon._to_snapped_bool_poly(self, tolerance),
            tolerance / 100)
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
            self._to_bool_poly(),
            polygon._to_snapped_bool_poly(self, tolerance),
            tolerance / 100)
        return Polygon2D._from_bool_poly(result)

    @staticmethod
    def snap_polygons(polygons, tolerance):
        """Snap several Polygon2D to each other if they differ less than the tolerance.

        This is useful to run before performing operations where small tolerance
        differences are likely to cause issues, such as in boolean operations.

        Args:
            polygons: A list of Polygon2D, which will be snapped to each other.
            tolerance: The minimum distance at which points will be snapped.

        Returns:
            A list of the input polygon2D that have been snapped to one another.
        """
        new_polygons = list(polygons)
        for i, poly_1 in enumerate(new_polygons):
            try:
                for j, poly_2 in enumerate(new_polygons[i + 1:]):
                    new_polygons[i + j + 1] = poly_1.snap_to_polygon(poly_2, tolerance)
            except IndexError:
                pass  # we have reached the end of the list of polygons
        return new_polygons

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
        polygons = Polygon2D.snap_polygons(polygons, tolerance)
        bool_polys = [poly._to_bool_poly() for poly in polygons]
        result = pb.union_all(bool_polys, tolerance / 100)
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
        polygons = Polygon2D.snap_polygons(polygons, tolerance)
        bool_polys = [poly._to_bool_poly() for poly in polygons]
        result = pb.intersect_all(bool_polys, tolerance / 100)
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
            polygon1._to_bool_poly(),
            polygon2._to_snapped_bool_poly(polygon1, tolerance),
            tolerance / 100)
        intersection = Polygon2D._from_bool_poly(int_result)
        poly1_difference = Polygon2D._from_bool_poly(poly1_result)
        poly2_difference = Polygon2D._from_bool_poly(poly2_result)
        return intersection, poly1_difference, poly2_difference

    @staticmethod
    def perimeter_core_by_offset(polygon, distance, holes=None):
        """Compute perimeter and core sub-polygons using a simple offset method.

        This method will only return polygons when the distance is shallow enough
        that the perimeter offset does not intersect itself or turn inward on itself.
        Otherwise, the method will simply return None.

        Args:
            polygon: A Polygon2D to split into perimeter and core sub-polygons.
            distance: Distance in model units to offset perimeter sub-polygon.
            holes: A list of Polygon2D objects representing holes in the
                polygon. (Default: None).

        Returns:
            A tuple with two items.

            * perimeter_sub_polys -- A list of perimeter sub-polygons as Polygon2D
                objects. Will be None if the offset distance is too deep.

            * core_sub_polys -- A list of core sub-polygons as Polygon2D objects. In the
                event of a core sub-polygon with a hole, a list with be returned with
                the first item being a boundary and successive items as hole polygons.
                Will be None if the offset distance is too deep.
        """
        # extract the core polygon and make sure it doesn't intersect itself
        core_sub_poly = polygon.offset(distance, check_intersection=True)
        if core_sub_poly is None:
            return None, None
        # generate the perimeter polygons
        if holes is None:
            perimeter_sub_polys = []
            for out_seg, in_seg in zip(polygon.segments, core_sub_poly.segments):
                pts = (out_seg.p1, out_seg.p2, in_seg.p2, in_seg.p1)
                perimeter_sub_polys.append(Polygon2D(pts))
            return perimeter_sub_polys, [core_sub_poly]
        else:
            # offset all of the holes into the shape
            core_sub_polys = [core_sub_poly]
            for hole in holes:
                hole_sub_poly = hole.offset(-distance, check_intersection=True)
                if hole_sub_poly is None:
                    return None, None
                core_sub_polys.append(hole_sub_poly)
            # check that None of the holes intersect one another
            for i, c_pgon in enumerate(core_sub_polys):
                for other_pgon in core_sub_polys[i + 1:]:
                    if Polygon2D._do_polygons_intersect(c_pgon, other_pgon):
                        return None, None
            # if nothing intersects, we can build the perimeter polygons
            out_polys = [polygon] + list(holes)
            perimeter_sub_polys = []
            for p_count, (out_poly, in_poly) in enumerate(zip(out_polys, core_sub_polys)):
                for out_seg, in_seg in zip(out_poly.segments, in_poly.segments):
                    if p_count == 0:
                        pts = (out_seg.p1, out_seg.p2, in_seg.p2, in_seg.p1)
                    else:
                        if not out_poly.is_clockwise:
                            pts = (out_seg.p1, in_seg.p1, in_seg.p2, out_seg.p2)
                        else:
                            (out_seg.p1, out_seg.p2, in_seg.p2, in_seg.p1)
                    perimeter_sub_polys.append(Polygon2D(pts))
            return perimeter_sub_polys, core_sub_polys

    @staticmethod
    def _do_polygons_intersect(polygon_1, polygon_2):
        """Test to see if two polygons intersect one another."""
        for seg in polygon_1.segments:
            for _s in polygon_2.segments:
                if does_intersection_exist_line2d(seg, _s):
                    return True
        return False

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

    @staticmethod
    def group_by_overlap(polygons, tolerance):
        """Group Polygon2Ds that overlap one another within the tolerance.

        This is useful as a pre-step before running Polygon2D.boolean_union_all()
        in order to assess whether unionizing is necessary and to ensure that
        it is only performed among the necessary groups of polygons.

        This method will return the minimal number of overlapping polygon groups
        thanks to a recursive check of whether groups can be merged.

        Args:
            polygons: A list of Polygon2D to be grouped by their overlapping.
            tolerance: The minimum distance from the edge of a neighboring polygon
                at which a point is considered to overlap with that polygon.

        Returns:
            A list of lists where each sub-list represents a group of polygons
            that all overlap with one another.
        """
        # sort the polygons by area to help larger ones grab smaller ones
        polygons = list(sorted(polygons, key=lambda x: x.area, reverse=True))

        # loop through the polygons and check to see if it overlaps with the others
        grouped_polys = [[polygons[0]]]
        for poly in polygons[1:]:
            group_found = False
            for poly_group in grouped_polys:
                for oth_poly in poly_group:
                    if poly.polygon_relationship(oth_poly, tolerance) >= 0:
                        poly_group.append(poly)
                        group_found = True
                        break
                if group_found:
                    break
            if not group_found:  # the polygon does not overlap with any of the others
                grouped_polys.append([poly])  # make a new group for the polygon

        # if some groups were found, recursively merge groups together
        old_group_len = len(polygons)
        while len(grouped_polys) != old_group_len:
            new_groups, g_to_remove = grouped_polys[:], []
            for i, group_1 in enumerate(grouped_polys):
                try:
                    for j, group_2 in enumerate(grouped_polys[i + 1:]):
                        if Polygon2D._groups_overlap(group_1, group_2, tolerance):
                            new_groups[i] = new_groups[i] + group_2
                            g_to_remove.append(i + j + 1)
                except IndexError:
                    pass  # we have reached the end of the list of polygons
            if len(g_to_remove) != 0:
                g_to_remove = list(set(g_to_remove))
                g_to_remove.sort()
                for ri in reversed(g_to_remove):
                    new_groups.pop(ri)
            old_group_len = len(grouped_polys)
            grouped_polys = new_groups
        return grouped_polys

    @staticmethod
    def group_by_touching(polygons, tolerance):
        """Group Polygon2Ds that touch or overlap one another within the tolerance.

        This is useful to group geometries together before extracting a bounding
        rectangle or convex hull around multiple polygons.

        This method will return the minimal number of polygon groups
        thanks to a recursive check of whether groups can be merged.

        Args:
            polygons: A list of Polygon2D to be grouped by their touching.
            tolerance: The minimum distance from the edge of a neighboring polygon
                at which a point is considered to touch that polygon.

        Returns:
            A list of lists where each sub-list represents a group of polygons
            that all touch or overlap with one another.
        """
        # sort the polygons by area to help larger ones grab smaller ones
        polygons = list(sorted(polygons, key=lambda x: x.area, reverse=True))

        # loop through the polygons and check to see if it overlaps with the others
        grouped_polys = [[polygons[0]]]
        for poly in polygons[1:]:
            group_found = False
            for poly_group in grouped_polys:
                for oth_poly in poly_group:
                    if poly.does_polygon_touch(oth_poly, tolerance):
                        poly_group.append(poly)
                        group_found = True
                        break
                if group_found:
                    break
            if not group_found:  # the polygon does not touch any of the others
                grouped_polys.append([poly])  # make a new group for the polygon

        # if some groups were found, recursively merge groups together
        old_group_len = len(polygons)
        while len(grouped_polys) != old_group_len:
            new_groups, g_to_remove = grouped_polys[:], []
            for i, group_1 in enumerate(grouped_polys):
                try:
                    for j, group_2 in enumerate(grouped_polys[i + 1:]):
                        if Polygon2D._groups_touch(group_1, group_2, tolerance):
                            new_groups[i] = new_groups[i] + group_2
                            g_to_remove.append(i + j + 1)
                except IndexError:
                    pass  # we have reached the end of the list of polygons
            if len(g_to_remove) != 0:
                g_to_remove = list(set(g_to_remove))
                g_to_remove.sort()
                for ri in reversed(g_to_remove):
                    new_groups.pop(ri)
            old_group_len = len(grouped_polys)
            grouped_polys = new_groups
        return grouped_polys

    @staticmethod
    def _groups_overlap(group_1, group_2, tolerance):
        """Evaluate whether two groups of Polygons overlap with one another."""
        for poly_1 in group_1:
            for poly_2 in group_2:
                if poly_1.polygon_relationship(poly_2, tolerance) >= 0:
                    return True
        return False

    @staticmethod
    def _groups_touch(group_1, group_2, tolerance):
        """Evaluate whether two groups of Polygons touch with one another."""
        for poly_1 in group_1:
            for poly_2 in group_2:
                if poly_1.does_polygon_touch(poly_2, tolerance):
                    return True
        return False

    @staticmethod
    def joined_intersected_boundary(polygons, tolerance):
        """Get the boundary around several Polygon2D that are touching one another.

        This method is faster and more reliable than the gap_crossing_boundary
        but requires that the Polygon2D be touching one another within the tolerance.

        Args:
            polygons: The polygons to be joined into a boundary. These polygons
                should have colinear vertices removed and they should not contain
                degenerate polygons at the tolerance. The remove_colinear_vertices
                method can be used to pre-process the input polygons to ensure they
                meet these criteria.
            tolerance: The tolerance at which the polygons are to be intersected
                and then joined to give a resulting boundary.

        Returns:
            A list of Polygon2D that represent the boundary around the input polygons.
            Note that some of these Polygon2D may represent 'holes' within others
            and it may be necessary to assess this when interpreting the result.
        """
        # intersect the polygons with one another
        int_poly = Polygon2D.intersect_polygon_segments(polygons, tolerance)

        # get indices of all unique vertices across the polygons
        vertices = []  # collection of vertices as point objects
        poly_indices = []  # collection of polygon indices
        for loop in int_poly:
            ind = []
            for v in loop:
                found = False
                for i, vert in enumerate(vertices):
                    if v.is_equivalent(vert, tolerance):
                        found = True
                        ind.append(i)
                        break
                if not found:  # add new point
                    vertices.append(v)
                    ind.append(len(vertices) - 1)
            poly_indices.append(ind)

        # use the unique vertices to extract naked edges
        edge_i = []
        edge_t = []
        for poly_i in poly_indices:
            for i, vi in enumerate(poly_i):
                try:  # this can get slow for large number of vertices
                    ind = edge_i.index((vi, poly_i[i - 1]))
                    edge_t[ind] += 1
                except ValueError:  # make sure reversed edge isn't there
                    try:
                        ind = edge_i.index((poly_i[i - 1], vi))
                        edge_t[ind] += 1
                    except ValueError:  # add a new edge
                        if poly_i[i - 1] != vi:  # avoid cases of same start and end
                            edge_i.append((poly_i[i - 1], vi))
                            edge_t.append(0)
        ext_edges = []
        for i, et in enumerate(edge_t):
            if et == 0:
                edg_ind = edge_i[i]
                pts_2d = (vertices[edg_ind[0]], vertices[edg_ind[1]])
                ext_edges.append(LineSegment2D.from_end_points(*pts_2d))

        # join the naked edges into closed polygons
        outlines = Polyline2D.join_segments(ext_edges, tolerance)
        closed_polys = []
        for bnd in outlines:
            if isinstance(bnd, Polyline2D) and bnd.is_closed(tolerance):
                closed_polys.append(bnd.to_polygon(tolerance))
        return closed_polys

    @staticmethod
    def gap_crossing_boundary(polygons, min_separation, tolerance):
        """Get the boundary around several Polygon2D, crossing gaps of min_separation.

        This method is less reliable than the joined_intersected_boundary because
        higher values of min_separation that are greater than the lengths of polygon
        segments can cause important details of the polygons to disappear. However,
        when used appropriately, it can provide a boundary that jumps across gaps
        to give resulting polygons that effectively bound all of the input polygons.

        Args:
            polygons: The polygons to be joined into a boundary. These polygons
                should have colinear vertices removed and they should not contain
                degenerate polygons at the tolerance. The remove_colinear_vertices
                method can be used to pre-process the input polygons to ensure they
                meet these criteria.
            min_separation: A number for the minimum distance between Polygon2D that
                is considered a meaningful separation. In other words, this is
                the maximum distance of the gap across
            tolerance: The maximum difference between coordinate values of two
                vertices at which they can be considered equivalent.

        Returns:
            A list of Polygon2D that represent the boundary around the input polygons.
            Note that some of these Polygon2D may represent 'holes' within others
            and it may be necessary to assess this when interpreting the result.
        """
        # ensure that all of the input polygons are counterclockwise
        cclock_poly = []
        for poly in polygons:
            if poly.is_clockwise:
                cclock_poly.append(poly.reverse())
            else:
                cclock_poly.append(poly)

        # determine which Polygon2D segments are 'exterior' using the min_separation
        right_ang = -math.pi / 2
        ext_segs = []
        for i, poly in enumerate(cclock_poly):
            # remove any short segments
            rel_segs = [s for s in poly.segments if s.length > min_separation]

            # create min_separation line segments to be used to test intersection
            test_segs = []
            for _s in rel_segs:
                d_vec = _s.v.rotate(right_ang).normalize()
                seg_pts = _s.subdivide(min_separation)
                if len(seg_pts) <= 3:
                    seg_pts = [_s.midpoint]
                else:
                    seg_pts = seg_pts[1:-1]
                spec_test_segs = []
                for spt in seg_pts:
                    m_pt = spt.move(d_vec * -tolerance)
                    spec_test_segs.append(LineSegment2D(m_pt, d_vec * min_separation))
                test_segs.append(spec_test_segs)

            # intersect the test line segments to asses which parts are exterior
            non_int_segs = []
            other_poly = [p for j, p in enumerate(cclock_poly) if j != i]
            for j, (_s, int_lins) in enumerate(zip(rel_segs, test_segs)):
                int_vals = [0] * len(int_lins)
                for m, int_lin in enumerate(int_lins):
                    for _oth_p in other_poly:
                        if _oth_p.intersect_line_ray(int_lin):  # intersection!
                            int_vals[m] = 1
                            break
                if sum(int_vals) == len(int_lins):  # fully internal line
                    continue
                else:  # if the polygon is concave, also check for self intersection
                    if not poly.is_convex:
                        _other_segs = [x for k, x in enumerate(rel_segs) if k != j]
                        for m, int_lin in enumerate(int_lins):
                            for _oth_s in _other_segs:
                                if int_lin.intersect_line_ray(_oth_s) is not None:
                                    int_vals[m] = 1
                                    break

                # determine the exterior segments using the intersections
                check_sum = sum(int_vals)
                if check_sum == 0:  # fully external line
                    non_int_segs.append(_s)
                elif len(int_vals) == 1 or check_sum == len(int_vals):
                    continue  # fully internal line
                else:  # line that extends from inside to outside
                    # first see if the exterior part is meaningful
                    count_in_a_rows, repeat_count = [], 0
                    for v in int_vals:
                        if v == 0:
                            repeat_count += 1
                            count_in_a_rows.append(repeat_count)
                        else:
                            repeat_count = 0
                    max_repeat = max(count_in_a_rows)
                    # if the exterior part is meaningful, split it
                    if max_repeat != 1:
                        last_pt = _s.p1 if int_vals[0] == 0 else None
                        for v, ts in zip(int_vals, int_lins):
                            if v == 0 and last_pt is None:
                                last_pt = ts.p1
                            elif v == 1 and last_pt is not None:
                                lin_seg = LineSegment2D.from_end_points(last_pt, ts.p1)
                                last_pt = None
                                non_int_segs.append(lin_seg)
                        if last_pt is not None:
                            lin_seg = LineSegment2D.from_end_points(last_pt, _s.p2)
                            non_int_segs.append(lin_seg)

            ext_segs.extend(non_int_segs)

        # loop through exterior segments and add segments across the min_separation
        joining_segs = []
        for i, e_seg in enumerate(ext_segs):
            try:
                for o_seg in ext_segs[i + 1:]:
                    dist, pts = closest_end_point2d_between_line2d(e_seg, o_seg)
                    if tolerance < dist <= min_separation:
                        joining_segs.append(LineSegment2D.from_end_points(*pts))
            except IndexError:
                pass  # we have reached the end of the list

        # join all of the segments together into polylines
        all_segs = ext_segs + joining_segs
        ext_bounds = Polyline2D.join_segments(all_segs, tolerance)

        # separate valid closed boundaries from open ones
        closed_polys, open_bounds = [], []
        for bnd in ext_bounds:
            if isinstance(bnd, Polyline2D) and bnd.is_closed(tolerance):
                try:
                    closed_polys.append(bnd.to_polygon(tolerance))
                except AssertionError:  # not a valid polygon
                    pass
            else:
                open_bounds.append(bnd)

        # if the resulting polylines are not closed, join the nearest end points
        if len(closed_polys) != len(ext_bounds):
            extra_segs = []
            for i, s_bnd in enumerate(open_bounds):
                self_seg = LineSegment2D.from_end_points(s_bnd.p1, s_bnd.p2)
                poss_segs = [self_seg]
                try:
                    for o_bnd in open_bounds[i + 1:]:
                        pts = [
                            (s_bnd.p1, o_bnd.p1), (s_bnd.p1, o_bnd.p2),
                            (s_bnd.p2, o_bnd.p1), (s_bnd.p2, o_bnd.p2)]
                        for comb in pts:
                            poss_segs.append(LineSegment2D.from_end_points(*comb))
                except IndexError:
                    continue  # we have reached the end of the list
                # sort the possible segments by their length
                poss_segs.sort(key=lambda x: x.length, reverse=False)
                if poss_segs[0] is self_seg:
                    extra_segs.append(poss_segs[0])
                else:  # two possible connecting segments
                    extra_segs.append(poss_segs[0])
                    extra_segs.append(poss_segs[1])
            # remove any duplicates from the extra segment list
            non_dup_segs = []
            for e_seg in extra_segs:
                for f_seg in non_dup_segs:
                    if e_seg.is_equivalent(f_seg, tolerance):
                        break
                else:
                    non_dup_segs.append(e_seg)
            extra_segs = non_dup_segs
            # take the best available segments that fit the criteria
            extra_segs.sort(key=lambda x: x.length, reverse=False)
            extra_segs = extra_segs[:len(open_bounds)]

            # join all segments, hopefully into a final closed polyline
            all_segs = ext_segs + joining_segs + extra_segs
            ext_bounds = Polyline2D.join_segments(all_segs, tolerance)
            closed_polys = []
            for bnd in ext_bounds:
                if isinstance(bnd, Polyline2D) and bnd.is_closed(tolerance):
                    try:
                        closed_polys.append(bnd.to_polygon(tolerance))
                    except AssertionError:  # not a valid polygon
                        pass

        return closed_polys

    @staticmethod
    def common_axes(polygons, direction, min_distance, merge_distance, angle_tolerance):
        """Get LineSegment2Ds for the most common axes across a set of Polygon2Ds.

        This is often useful as a step before aligning a set of polygons to these
        common axes.

        Args:
            polygons: A list or tuple of Polygon2D objects for which common axes
                will be evaluated.
            direction: A Vector2D object to represent the direction in which the
                common axes will be evaluated and generated
            min_distance: The minimum distance at which common axes will be evaluated.
                This value should typically be a little larger than the model
                tolerance (eg. 5 to 20 times the tolerance) in order to ensure that
                possible common axes across the input polygons are not missed.
            merge_distance: The distance at which common axes next to one another
                will be merged into a single axis. This should typically be 2-3
                times the min_distance in order to avoid generating several axes
                that are immediately adjacent to one another. When using this
                method to generate axes for alignment, this merge_distance should
                be in the range of the alignment distance.
            angle_tolerance: The max angle difference in radians that the polygon
                segments direction can differ from the input direction before the
                segments are not factored into this calculation of common axes.

            Returns:
                A tuple with two elements.

            -   common_axes: A list of LineSegment2D objects for the common
                axes across the input polygons.

            -   axis_values: A list of integers that aligns with the common_axes
                and denotes how many segments of the input polygons each axis
                relates to. Higher numbers indicate that that the axis is more
                common among all of the possible axes.
        """
        # gather the relevant segments of the input polygons
        min_ang, max_ang = angle_tolerance, math.pi - angle_tolerance
        rel_segs = []
        for p_gon in polygons:
            for seg in p_gon.segments:
                try:
                    s_ang = direction.angle(seg.v)
                    if s_ang < min_ang or s_ang > max_ang:
                        rel_segs.append(seg)
                except ZeroDivisionError:  # zero length segment to ignore
                    continue
        if len(rel_segs) == 0:
            return [], []  # none of the polygon segments are relevant in the direction

        # determine the extents around the polygons and the input direction
        gen_vec = direction.rotate(math.pi / 2)
        axis_angle = Vector2D(0, 1).angle_counterclockwise(gen_vec)
        orient_poly = polygons
        if axis_angle != 0:  # rotate geometry to the bounding box
            cpt = polygons[0].vertices[0]
            orient_poly = [pl.rotate(-axis_angle, cpt) for pl in polygons]
        xx = Polygon2D._bounding_domain_x(orient_poly)
        yy = Polygon2D._bounding_domain_y(orient_poly)
        min_pt = Point2D(xx[0], yy[0])
        max_pt = Point2D(xx[1], yy[1])
        if axis_angle != 0:  # rotate the points back
            min_pt = min_pt.rotate(axis_angle, cpt)
            max_pt = max_pt.rotate(axis_angle, cpt)

        # generate all possible axes from the extents and min_distance
        axis_vec = direction.normalize() * (xx[1] - xx[0])
        incr_vec = gen_vec.normalize() * (min_distance)
        current_pt = min_pt
        current_dist, max_dist = 0, yy[1] - yy[0]
        all_axes = []
        while current_dist < max_dist:
            axis = LineSegment2D(current_pt, axis_vec)
            all_axes.append(axis)
            current_pt = current_pt.move(incr_vec)
            current_dist += min_distance

        # evaluate the axes based on how many relevant segments they are next to
        mid_pts = [seg.midpoint for seg in rel_segs]
        rel_axes, axes_value = [], []
        for axis in all_axes:
            axis_val = 0
            for pt in mid_pts:
                close_pt = closest_point2d_on_line2d_infinite(pt, axis)
                if close_pt.distance_to_point(pt) <= min_distance:
                    axis_val += 1
            if axis_val != 0:
                rel_axes.append(axis)
                axes_value.append(axis_val)
        if len(rel_axes) == 0:
            return [], []  # none of the generated axes are relevant

        # group the axes by proximity
        last_ax = rel_axes[0]
        axes_groups = [[last_ax]]
        group_values = [[axes_value[0]]]
        for axis, val in zip(rel_axes[1:], axes_value[1:]):
            if axis.p.distance_to_point(last_ax.p) <= merge_distance:
                axes_groups[-1].append(axis)
                group_values[-1].append(val)
            else:  # start a new group
                axes_groups.append([axis])
                group_values.append([val])
            last_ax = axis

        # average the line segments that are within the merge_distance of one another
        axis_values = [max(val) for val in group_values]
        common_axes = []
        for ax_group, grp_vals in zip(axes_groups, group_values):
            if len(ax_group) == 1:
                common_axes.append(ax_group[0])
            else:
                index_max = max(range(len(grp_vals)), key=grp_vals.__getitem__)
                common_axes.append(ax_group[index_max])
        return common_axes, axis_values

    @staticmethod
    def _bounding_domain_x(geometries):
        """Get minimum and maximum X coordinates of multiple polygons."""
        min_x, max_x = geometries[0].min.x, geometries[0].max.x
        for geom in geometries[1:]:
            if geom.min.x < min_x:
                min_x = geom.min.x
            if geom.max.x > max_x:
                max_x = geom.max.x
        return min_x, max_x

    @staticmethod
    def _bounding_domain_y(geometries):
        """Get minimum and maximum Y coordinates of multiple polygons."""
        min_y, max_y = geometries[0].min.y, geometries[0].max.y
        for geom in geometries[1:]:
            if geom.min.y < min_y:
                min_y = geom.min.y
            if geom.max.y > max_y:
                max_y = geom.max.y
        return min_y, max_y

    def _point_in_polygon(self, tolerance):
        """Get a Point2D that is always reliably inside this Polygon2D.

        The point will be close to the edge of the Polygon but it will always
        be inside it for all concave geometries. Furthermore, it is relatively
        fast compared with computing the pole_of_inaccessibility.
        """
        try:
            poly = self.remove_colinear_vertices(tolerance)
            move_vec, v_angle = self._inward_pointing_vec(poly)
        except (AssertionError, ZeroDivisionError):  # zero area Polygon2D; use center
            return self.center

        move_vec = move_vec * ((tolerance / math.sin(v_angle / 2)) + 0.00001)
        point_in_poly = poly.vertices[0] + move_vec
        if not self.is_point_inside(point_in_poly):
            point_in_poly = poly.vertices[0] - move_vec
        return point_in_poly

    @staticmethod
    def _inward_pointing_vec(polygon):
        """Get a unit vector pointing inward/outward from the first vertex of the Polygon
        """
        v1 = polygon.vertices[-1] - polygon.vertices[0]
        v2 = polygon.vertices[1] - polygon.vertices[0]
        v_angle = v1.angle(v2)
        if v_angle == math.pi:  # colinear vertices; prevent averaging to zero
            rgt_ang = math.pi / 2
            return v1.rotate(rgt_ang).normalize(), rgt_ang
        else:  # average the two edge vectors together
            avg_coords = ((v1.x + v2.x) / 2), ((v1.y + v2.y) / 2)
            return Vector2D(*avg_coords).normalize(), v_angle

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
        The run time of this method scales linearly with the total number of
        vertices, which makes it significantly better for shapes with many holes
        compared to recursively calling the Polygon2D._merge_boundary_and_closest_hole
        method. However, it is not nearly as efficient as the method used in
        Polygon2D.from_shape_with_holes_fast, which is typically not as beautiful
        of a result as this method.

        Args:
            boundary: A list of Point2D objects for the outer boundary inside of
                which the hole is contained.
            hole: A list of lists where each sub-list represents a hole and contains
                several Point2D objects that represent the hole.
            split: A boolean to note whether the last hole should be merged into
                the boundary, effectively splitting the shape into two lists of
                vertices instead of a single list. This is useful when trying to
                translate a shape with holes to a platform that does not support
                holes or struggles with single lists of vertices that wind inward
                to cut out the holes since this option returns two "normal" concave
                polygons. However, it is possible that the shape cannot be
                reliably split this way and, in this case, this method will
                return None. (Default: False).

        Returns:
            A single list of vertices with the holes merged into the boundary. When
            split is True and the splitting is successful, this will be two lists
            of vertices for the split shapes. If splitting was not successful,
            this method will return None and, if a hole-less split shape is
            still required, it is recommended that triangulation be used to get
            the hole-less shapes.
        """
        # compute the initial distances between the holes and the boundary
        original_boundary = boundary[:]
        hole_dicts, min_dists = [], []
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

        # sort the distances to find the closest point
        last_hd = hole_dicts[0]
        sort_dist = sorted(last_hd.keys())
        # find the closest connection between the hole and the original boundary polygon
        p1_indices = dist_dict[sort_dist[0]]
        p2_index = 1
        p2_indices = dist_dict[sort_dist[p2_index]]
        p2_bound_pt = boundary[p2_indices[0]]
        while p2_bound_pt not in original_boundary:
            p2_index += 1
            p2_indices = dist_dict[sort_dist[p2_index]]
            p2_bound_pt = boundary[p2_indices[0]]
        p2_hole_pt = hole[p2_indices[1]]
        # merge the hole into the boundary
        hole_deque = deque(hole)
        hole_deque.rotate(-p1_indices[1])
        hole_insert = [boundary[p1_indices[0]]] + list(hole_deque) + \
            [hole[p1_indices[1]]]
        boundary[p1_indices[0]:p1_indices[0]] = hole_insert
        # use the second most distant points to split the shape
        p2_bound_i = boundary.index(p2_bound_pt)
        p2_hole_i = boundary.index(p2_hole_pt)
        if p2_hole_i < p2_bound_i:
            boundary_1 = boundary[p2_hole_i:p2_bound_i + 1]
            boundary_2 = boundary[:p2_hole_i + 1] + boundary[p2_bound_i:]
        else:
            boundary_1 = boundary[p2_bound_i:p2_hole_i + 1]
            boundary_2 = boundary[:p2_bound_i + 1] + boundary[p2_hole_i:]
        poly_1, poly_2 = Polygon2D(boundary_1), Polygon2D(boundary_2)

        # if the split polygons are self-intersecting, try to find a solution
        p2_index = 0
        while poly_1.is_self_intersecting or poly_2.is_self_intersecting:
            p2_index += 1
            try:
                p2_indices = dist_dict[sort_dist[p2_index]]
            except IndexError:  # no solution was found; just return None
                return None
            p2_bound_pt = boundary[p2_indices[0]]
            p2_hole_pt = hole[p2_indices[1]]
            p2_bound_i = boundary.index(p2_bound_pt)
            p2_hole_i = boundary.index(p2_hole_pt)
            if p2_hole_i < p2_bound_i:
                boundary_1 = boundary[p2_hole_i:p2_bound_i + 1]
                boundary_2 = boundary[:p2_hole_i + 1] + boundary[p2_bound_i:]
            else:
                boundary_1 = boundary[p2_bound_i:p2_hole_i + 1]
                boundary_2 = boundary[:p2_bound_i + 1] + boundary[p2_hole_i:]
            try:
                poly_1, poly_2 = Polygon2D(boundary_1), Polygon2D(boundary_2)
            except AssertionError:
                pass  # the polygons are not valid; keep searching
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
