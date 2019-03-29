# coding=utf-8
"""2D Polygon"""
from __future__ import division

from .pointvector import Point2D, Point2DImmutable, Vector2D, Vector2DImmutable
from .line import LineSegment2DImmutable
from .ray import Ray2D
from ..intersection2d import intersect_line2d, does_intersection_exist_line2d, \
    closest_point2d_on_line2d
from ._2d import Base2DIn2D


class Polygon2D(Base2DIn2D):
    """2D polygon object.

    Properties:
        vertices
        segments
        perimeter
        area
        is_clockwise
        is_convex
        is_self_intersecting
        min
        max
        center
    """
    _check_required = True
    _segments = None
    _triangulated_mesh = None
    _perimeter = None
    _area = None
    _centroid = None
    _is_clockwise = None
    _is_convex = None
    _is_complex = None

    def __init__(self, vertices):
        """Initilize Polygon2D.

        Args:
            vertices: A list of Point2D objects representing the vertices of the polygon.
        """
        if self._check_required:
            self._check_vertices_input(vertices)
        else:
            self._vertices = vertices

    @classmethod
    def from_rectangle(cls, base_point, height_vector, base, height):
        """Initialize Polygon from rectangle parameters.

        Initializing a polygon this way has the added benefit of having its properties
        quickly computed.

        Args:
            base_point: A Point2D for the lower left vertex of the polygon.
            height_vector: A vector denoting the direction of the rectangle height.
            base: A number indicating the length of the base of the rectangle.
            height: A number indicating the length of the height of the rectangle.
        """
        assert isinstance(base_point, (Point2D, Point2DImmutable)), \
            'base_point must be Point2D. Got {}.'.format(type(base_point))
        assert isinstance(height_vector, (Vector2D, Vector2DImmutable)), \
            'height_vector must be Vector2D. Got {}.'.format(type(height_vector))
        assert isinstance(base, (float, int)), 'base must be a number.'
        assert isinstance(height, (float, int)), 'height must be a number.'
        _hv_norm = height_vector.normalized()
        _bv = Vector2D(_hv_norm.y, -_hv_norm.x) * base
        _hv = _hv_norm * height
        _verts = (base_point, base_point + _hv, base_point + _hv + _bv, base_point + _bv)
        polygon = cls(_verts)
        polygon._perimeter = base * 2 + height * 2
        polygon._area = base * height
        polygon._is_clockwise = True
        polygon._is_convex = False
        polygon._is_complex = False
        return polygon

    @property
    def vertices(self):
        """Tuple of all vertices in this geometry."""
        return self._vertices

    @property
    def segments(self):
        """Tuple of all line segments in the polygon."""
        if self._segments is None:
            _segs = []
            for i, vert in enumerate(self.vertices):
                _seg = LineSegment2DImmutable.from_end_points(self.vertices[i - 1], vert)
                _segs.append(_seg)
            _segs.append(_segs.pop(0))  # segments will start from the base point
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

        Note that this property is computationally intense and most CAD programs forbid
        surfaces with self-intersecting edges.
        So this property should only be used in quality control scripts where the
        origin of the geometry is unknown.
        """
        if self._is_complex is None:
            self._is_complex = False
            if self.is_convex is False:
                _segs = self.segments
                for i, _s in enumerate(_segs[1: len(_segs) - 1]):
                    _skip = (i, i + 1, i + 2)
                    _other_segs = [x for j, x in enumerate(_segs) if j not in _skip]
                    for _oth_s in _other_segs:
                        if _s.intersect_line2(_oth_s) is not None:  # Found intersection
                            self._is_complex = True
                            break
                    if self._is_complex is True:
                        break
        return self._is_complex

    def reversed(self):
        """Get a copy of this polygon where the vertices are reversed."""
        return Polygon2D([pt for pt in reversed(self.vertices)])

    def move(self, moving_vec):
        """Get a polygon that has been moved along a vector.

        Args:
            moving_vec: A Vector2D with the direction and distance to move the polygon.
        """
        return Polygon2D([pt.move(moving_vec) for pt in self.vertices])

    def rotate(self, angle, origin):
        """Get a polygon that is rotated counterclockwise by a certain angle.

        Args:
            angle: An angle for rotation in radians.
            origin: A Point2D for the origin around which the point will be rotated.
        """
        return Polygon2D([pt.rotate(angle, origin) for pt in self.vertices])

    def reflect(self, normal, origin):
        """Get a polygon reflected across a plane with the input normal vector and origin.

        Args:
            normal: A Vector2D representing the normal vector for the plane across
                which the polygon will be reflected. THIS VECTOR MUST BE NORMALIZED.
            origin: A Point2D representing the origin from which to reflect.
        """
        return Polygon2D([pt.reflect(normal, origin) for pt in self.vertices])

    def scale(self, factor, origin):
        """Scale a polygon by a factor from an origin point.

        Args:
            factor: A number representing how much the polygon should be scaled.
            origin: A Point2D representing the origin from which to scale.
        """
        return Polygon2D([pt.scale(factor, origin) for pt in self.vertices])

    def scale_world_origin(self, factor, origin):
        """Scale a polygon by a factor from the world origin. Faster than Polygon2D.scale.

        Args:
            factor: A number representing how much the polygon should be scaled.
        """
        return Polygon2D([pt.scale_world_origin(factor) for pt in self.vertices])

    def intersect_line2(self, line_ray):
        """Get the intersections between this polygon and a Ray2 or LineSegment2D.

        Args:
            line_ray: A LineSegment2D or Ray2 or to intersect.

        Returns:
            A list with Point2D objects for the intersections. List will be empty if no
                intersection exists.
        """
        intersections = []
        for _s in self.segments:
            inters = intersect_line2d(_s, line_ray)
            if inters is not None:
                intersections.append(inters)
        return intersections

    def point_relationship(self, point, tolerance):
        """Test whether a Point2D lies inside, outside or on the boundary of the polygon.

        Args:
            point: A Point2D for which the relationship to the polygon will be tested.
            tolerance: The minimum distance from the edge at wich a point is
                considered to lie on the edge.

        Returns:
            An integer denoting the relationship of the point. This will be one
                of the following:
                    -1 = Outside polygon
                     0 = On the edge of the polygon
                    +1 = Inside polygon
        """
        if self.is_point_on_edge(point, tolerance):
            return 0
        if self.is_point_inside(point):
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
            if point.distance_to_point(close_pt) < tolerance:
                return True
        return False

    def is_point_inside(self, point):
        """Test whether a Point2D lies inside or outside the polygon.

        This method will not test for whether a point lies on the edge of the
        polygon so it can only assess whether a point lies inside up to Python
        floating point tolerance (1e-16).
        If distinguishing edge conditions from inside/ outside is
        important, the point_relationship method should be used.

        Args:
            point: A Point2D for which the inside/ outside relationship will be tested.

        Returns:
            A boolean denoting whether the point lies inside (True) or outside (False).
        """
        ray = Ray2D(point, Vector2D(1, 0))
        n_int = 0
        for _s in self.segments:
            if does_intersection_exist_line2d(_s, ray):
                n_int += 1
        if n_int % 2 == 0:
            return False
        return True

    def is_point_inside_bound_rect(self, point):
        """Test whether a Point2D lies roughly inside or outside the polygon.

        This function is virtually identical to the Polygon2D.is_point_inside
        method but will first do a check to see if the point lies inside the
        polygon bounding rectangle. As such, it is a faster approach when one
        expects many of the tested points to lie far away from the polygon.

        Args:
            point: A Point2D for which the inside/ outside relationship will be tested.

        Returns:
            A boolean denoting whether the point lies inside (True) or outside (False).
        """
        min = self.min
        max = self.max
        if point.x < min.x or point.y < min.y or point.x > max.x or point.y > max.y:
            return False
        return self.is_point_inside(point)

    def _check_vertices_input(self, vertices):
        assert isinstance(vertices, (list, tuple)), \
            'vertices should be a list or tuple. Got {}'.format(type(vertices))
        assert len(vertices) > 2, 'There must be at least 3 vertices for a Polygon.' \
            ' Got {}'.format(len(vertices))
        _verts_immutable = []
        for p in vertices:
            assert isinstance(p, (Point2D, Point2DImmutable)), \
                'Expected Point2D. Got {}.'.format(type(p))
            _verts_immutable.append(p.to_immutable())
        self._vertices = tuple(_verts_immutable)

    def __copy__(self):
        Polygon2D._check_required = False  # Turn off check since we know input is valid
        _new_poly = Polygon2D(self.vertices)
        Polygon2D._check_required = True  # Turn the checks back on
        return _new_poly

    def __repr__(self):
        return 'Ladybug Polygon2D ({} vertices)'.format(len(self))
