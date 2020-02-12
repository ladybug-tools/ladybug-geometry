# coding=utf-8
"""2D Polygon"""
from __future__ import division

from .pointvector import Point2D, Vector2D
from .line import LineSegment2D
from .ray import Ray2D
from ..intersection2d import intersect_line2d, intersect_line2d_infinite, \
    does_intersection_exist_line2d, closest_point2d_on_line2d
from ._2d import Base2DIn2D

from collections import deque
import math


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
    __slots__ = ('_segments', '_triangulated_mesh', '_perimeter', '_area',
                 '_is_clockwise', '_is_convex', '_is_self_intersecting')

    def __init__(self, vertices):

        self._vertices = self._check_vertices_input(vertices)
        self._segments = None
        self._triangulated_mesh = None
        self._min = None
        self._max = None
        self._center = None
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
                polgygon. This number must be greater than 2.
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
        bound_direction = Polygon2D._are_clockwise(boundary)
        for hole in holes:
            if Polygon2D._are_clockwise(hole) is bound_direction:
                hole.reverse()

        # recursively add the nearest hole to the boundary until there are none left.
        while len(holes) > 0:
            boundary, holes = Polygon2D._merge_boundary_and_closest_hole(boundary, holes)

        # return the polygon with some properties set based on what we know
        _new_poly = cls(boundary)
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
            _segs = Polygon2D._segments_from_vertices(self.vertices)
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

    def to_array(self):
        """ Nested list of nested list of points. """
        return tuple(pt.to_array() for pt in self.vertices)

    def remove_colinear_vertices(self, tolerance):
        """Get a version of this polygon without colinear or duplicate vertices.

        Args:
            tolerance: The minimum distance that a vertex can be from a line
                before it is considered colinear.
        """
        if len(self.vertices) == 3:
            return self
        new_vertices = []
        for i, _v in enumerate(self.vertices):
            _a = self[i - 2].determinant(self[i - 1]) + self[i - 1].determinant(_v) + \
                _v.determinant(self[i - 2])
            if abs(_a) >= tolerance:
                new_vertices.append(self[i - 1])
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
        """Get a polygon reflected across a plane with the input normal vector and origin.

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
        """Get the intersections between this polygon and a Ray2D extended infintiely.

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

    def point_relationship(self, point, tolerance):
        """Test whether a Point2D lies inside, outside or on the boundary of the polygon.

        This is the slowest of the methods for understanding the relationship
        of a given point to a polygon. However, it covers all edge cases, including
        the literal edge of the polygon.

        Args:
            point: A Point2D for which the relationship to the polygon will be tested.
            tolerance: The minimum distance from the edge at wich a point is
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
        if self.is_point_inside_check(point):
            return 1
        return -1

    def is_point_on_edge(self, point, tolerance):
        """Test whether a Point2D lies on the boundary edges of the polygon.

        Args:
            point: A Point2D for which the edge relationship will be tested.
            tolerance: The minimum distance from the edge at wich a point is
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
        yeild the right result for all convex polygons and concave polygons with
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
        (1e-16). If distinguishing edge conditions from inside/ outside is
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

    def is_point_inside(self, point, test_vector=Vector2D(1, 0)):
        """Test whether a Point2D lies inside or outside the polygon.

        This method is the fastest way to tell if a point is inside a polygon when
        the given point lies inside the boundary rectangle of the polygon.
        However, while this method gives the correct result in 99.9% of cases,
        there are a few fringe cases where it will not give the correct result.
        Specifically these are:

        .. code-block:: shell

            1 - When the test_ray intersects perfectly with a polygon vertex.
                For example, this case with an X-unit test_vector:
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
        cases is needed.

        Args:
            point: A Point2D for which the inside/outside relationship will be tested.
            test_vector: Optional vector to set the direction in which intersections
                with the polygon edges will be evaluated to determine if the
                point is inside. Default is the X-unit vector.

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

    def is_point_inside_bound_rect(self, point, test_vector=Vector2D(1, 0)):
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

    def to_dict(self):
        """Get Polygon2D as a dictionary."""
        return {'type': 'Polygon2D',
                'vertices': [pt.to_array() for pt in self.vertices]}

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

        # Bounding rectangle check
        if not Polygon2D.overlapping_bounding_rect(polygon1, polygon2, tolerance):
            return polygon1, polygon2  # no overlap

        # Test if each point of polygon2 is within the tolerance distance of any segment
        # of polygon1.  If so, add the closest point on the segment to the polygon1
        # update list. And vice versa (testing polygon2 against polygon1).
        for i1, seg1 in enumerate(polygon1.segments):
            for i2, seg2 in enumerate(polygon2.segments):
                # Test polygon1 against polygon2
                x = closest_point2d_on_line2d(seg2.p1, seg1)
                if all(p.distance_to_point(x) > tolerance for p in polygon1.vertices) \
                        and x.distance_to_point(seg2.p1) <= tolerance:
                    polygon1_updates.append([i1, x])
                # Test polygon2 against polygon1
                y = closest_point2d_on_line2d(seg1.p1, seg2)
                if all(p.distance_to_point(y) > tolerance for p in polygon2.vertices) \
                        and y.distance_to_point(seg1.p1) <= tolerance:
                    polygon2_updates.append([i2, y])

        # Apply any updates to polygon1
        poly_points = list(polygon1.vertices)
        for update in polygon1_updates[::-1]:  # Traverse backwards to preserve order
            poly_points.insert(update[0] + 1, update[1])
        polygon1 = Polygon2D(poly_points)

        # Apply any updates to polygon2
        poly_points = list(polygon2.vertices)
        for update in polygon2_updates[::-1]:  # Traverse backwards to preserve order
            poly_points.insert(update[0] + 1, update[1])
        polygon2 = Polygon2D(poly_points)

        return polygon1, polygon2

    @staticmethod
    def overlapping_bounding_rect(polygon1, polygon2, tolerance):
        """Check if the bounding rectangles of two polygons overlap within a tolerance.

        This is particularly useful as a check before performing computationally intense
        processes between two polygons like intersection or checking for adjacency.
        Checking the overlap of the bounding boxes is extremely quick given this
        method's use of the Separating Axis Theorem.

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
    def _segments_from_vertices(vertices):
        _segs = []
        for i, vert in enumerate(vertices):
            _seg = LineSegment2D.from_end_points(vertices[i - 1], vert)
            _segs.append(_seg)
        _segs.append(_segs.pop(0))  # segments will start from the first point
        return _segs

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
        return Polygon2D(self._vertices)

    def __key(self):
        """A tuple based on the object properties, useful for hashing."""
        return tuple(hash(pt) for pt in self._vertices)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(other, Polygon2D) and self.__key() == other.__key()

    def __repr__(self):
        return 'Polygon2D ({} vertices)'.format(len(self))
