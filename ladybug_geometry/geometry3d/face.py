# coding=utf-8
"""Planar Face in 3D Space"""
from __future__ import division
import math
import sys
if (sys.version_info > (3, 0)):  # python 3
    xrange = range

from .pointvector import Point3D, Vector3D
from .ray import Ray3D
from .line import LineSegment3D
from .polyline import Polyline3D
from .plane import Plane
from .mesh import Mesh3D
from ._2d import Base2DIn3D

from ..intersection3d import closest_point3d_on_line3d

from ..geometry2d.pointvector import Point2D, Vector2D
from ..geometry2d.ray import Ray2D
from ..geometry2d.polygon import Polygon2D
from ..geometry2d.mesh import Mesh2D

import ladybug_geometry.boolean as pb


class Face3D(Base2DIn3D):
    """Planar Face in 3D space.

    Args:
        boundary: A list or tuple of Point3D objects representing the outer
            boundary vertices of the face.
        plane: A Plane object indicating the plane in which the face exists.
            If None, the Plane normal will automatically be calculated by
            analyzing the input vertices and the origin of the plane will be
            the first vertex of the input vertices. Default: None.
        holes: Optional list of lists with one list for each hole in the face.
            Each hole should be a list of at least 3 Point3D objects.
            If None, it will be assumed that there are no holes in the face.
            The boundary and holes are stored as separate lists of Point3Ds on the
            `boundary` and `holes` properties of this object. However, the
            `vertices` property will always contain all vertices across the shape.
            For a Face3D that has holes, it will trace out a single shape that
            turns inwards from the boundary to cut out the holes.
        enforce_right_hand: Boolean to note whether a check should be run to
            ensure that input vertices are counterclockwise within the input plane,
            thereby enforcing the right-hand rule. By default, this is True
            and ensures that all Face3D objects adhere to the right-hand rule.
            It is recommended that this only be set to False in cases where you
            are certain that the input vertices are counter-clockwise
            within the input plane and you would like to avoid the extra
            unnecessary check.

    Properties:
        * vertices
        * plane
        * boundary
        * holes
        * polygon2d
        * boundary_polygon2d
        * hole_polygon2d
        * triangulated_mesh2d
        * triangulated_mesh3d
        * boundary_segments
        * hole_segments
        * normal
        * min
        * max
        * center
        * perimeter
        * area
        * centroid
        * altitude
        * azimuth
        * is_clockwise
        * is_convex
        * is_self_intersecting
        * is_valid
        * has_holes
        * upper_left_corner
        * lower_left_corner
        * upper_right_corner
        * lower_right_corner
        * upper_left_counter_clockwise_vertices
        * lower_left_counter_clockwise_vertices
        * lower_right_counter_clockwise_vertices
        * upper_right_counter_clockwise_vertices
    """
    __slots__ = ('_plane', '_polygon2d', '_mesh2d', '_mesh3d',
                 '_boundary', '_holes', '_boundary_segments', '_hole_segments',
                 '_boundary_polygon2d', '_hole_polygon2d',
                 '_perimeter', '_area', '_centroid',
                 '_is_convex', '_is_self_intersecting')
    HOLE_VERTEX_THRESHOLD = 400  # threshold at which faster hole merging method is used

    def __init__(self, boundary, plane=None, holes=None, enforce_right_hand=True):
        """Initialize Face3D."""
        # process the boundary and plane inputs
        self._boundary = self._check_vertices_input(boundary)
        if plane is not None:
            assert isinstance(plane, Plane), 'Expected Plane for Face3D.' \
                ' Got {}.'.format(type(plane))
        else:
            plane = self._plane_from_vertices(boundary)
        self._plane = plane

        # process boundary and holes input
        if holes:
            assert isinstance(holes, (tuple, list)), \
                'holes should be a tuple or list. Got {}'.format(type(holes))
            self._holes = tuple(
                self._check_vertices_input(hole, 'hole') for hole in holes)
            # create a Polygon2D from the vertices
            _boundary2d = [self._plane.xyz_to_xy(_v) for _v in boundary]
            _holes2d = [[self._plane.xyz_to_xy(_v) for _v in hole] for hole in holes]
            v_count = len(_boundary2d)  # count the vertices for hole merging method
            for h in _holes2d:
                v_count += len(h)
            _polygon2d = Polygon2D.from_shape_with_holes_fast(_boundary2d, _holes2d) \
                if v_count > self.HOLE_VERTEX_THRESHOLD else \
                Polygon2D.from_shape_with_holes(_boundary2d, _holes2d)
            # convert Polygon2D vertices to 3D to become the vertices of the face.
            self._vertices = tuple(self._plane.xy_to_xyz(_v)
                                   for _v in _polygon2d.vertices)
            self._polygon2d = _polygon2d
        else:
            self._holes = None
            self._vertices = self._boundary
            self._polygon2d = None

        # perform a check of vertex orientation and enforce counter clockwise vertices
        if enforce_right_hand is True:
            if self.is_clockwise is True:
                self._boundary = tuple(reversed(self._boundary))
                self._vertices = tuple(reversed(self._vertices))
                if self._polygon2d is not None:
                    self._polygon2d = self._polygon2d.reverse()

        # set other properties to None for now
        self._mesh2d = None
        self._mesh3d = None
        self._boundary_polygon2d = None
        self._hole_polygon2d = None
        self._boundary_segments = None
        self._hole_segments = None
        self._min = None
        self._max = None
        self._center = None
        self._perimeter = None
        self._area = None
        self._centroid = None
        self._is_convex = None
        self._is_self_intersecting = None

    @classmethod
    def from_dict(cls, data):
        """Create a Face3D from a dictionary.

        Args:
            data: A python dictionary in the following format

        .. code-block:: python

            {
                "type": "Face3D",
                "boundary": [(0, 0, 0), (10, 0, 0), (0, 10, 0)],
                "plane": {"n": (0, 0, 1), "o": (0, 0, 0), "x": (1, 0, 0)},
                "holes": [[(2, 2, 0), (5, 2, 0), (2, 5, 0)]]
            }
        """
        holes = None
        if 'holes' in data and data['holes'] is not None:
            holes = tuple(tuple(
                Point3D.from_array(pt) for pt in hole) for hole in data['holes'])
        plane = None
        if 'plane' in data and data['plane'] is not None:
            plane = Plane.from_dict(data['plane'])
        return cls(tuple(Point3D.from_array(pt) for pt in data['boundary']),
                   plane, holes)

    @classmethod
    def from_extrusion(cls, line_segment, extrusion_vector):
        """Initialize Face3D by extruding a line segment.

        Initializing a face this way has the added benefit of having its
        properties quickly computed.

        Args:
            line_segment: A LineSegment3D to be extruded.
            extrusion_vector: A vector denoting the direction and distance to
                extrude the line segment.
        """
        assert isinstance(line_segment, LineSegment3D), \
            'line_segment must be LineSegment3D. Got {}.'.format(type(line_segment))
        assert isinstance(extrusion_vector, Vector3D), \
            'extrusion_vector must be Vector3D. Got {}.'.format(type(extrusion_vector))
        _p1 = line_segment.p1
        _p2 = line_segment.p2
        _verts = (_p1, _p2, _p2 + extrusion_vector, _p1 + extrusion_vector)
        _plane = Plane(line_segment.v.cross(extrusion_vector), _p1)
        face = cls(_verts, _plane, enforce_right_hand=False)
        _base = line_segment.length
        _dist = extrusion_vector.magnitude
        _height = _dist * math.sin(extrusion_vector.angle(line_segment.v))
        face._perimeter = _base * 2 + _dist * 2
        face._area = _base * _height
        face._centroid = _p1 + (line_segment.v * 0.5) + (extrusion_vector * 0.5)
        face._is_convex = True
        face._is_self_intersecting = False
        return face

    @classmethod
    def from_rectangle(cls, base, height, base_plane=None):
        """Initialize Face3D from rectangle parameters (base + height) and a base plane.

        Initializing a face this way has the added benefit of having its
        properties quickly computed.

        Args:
            base: A number indicating the length of the base of the rectangle.
            height: A number indicating the length of the height of the rectangle.
            base_plane: A Plane object in which the rectangle will be created.
                The origin of this plane will be the lower left corner of the
                rectangle and the X and Y axes will form the sides.
                Default is the world XY plane.
        """
        assert isinstance(base, (float, int)), 'Rectangle base must be a number.'
        assert isinstance(height, (float, int)), 'Rectangle height must be a number.'
        if base_plane is not None:
            assert isinstance(base_plane, Plane), \
                'base_plane must be Plane. Got {}.'.format(type(base_plane))
        else:
            base_plane = Plane(Vector3D(0, 0, 1), Point3D(0, 0, 0))
        _o = base_plane.o
        _b_vec = base_plane.x * base
        _h_vec = base_plane.y * height
        _verts = (_o, _o + _b_vec, _o + _h_vec + _b_vec, _o + _h_vec)
        face = cls(_verts, base_plane, enforce_right_hand=False)
        face._perimeter = base * 2 + height * 2
        face._area = base * height
        face._centroid = _o + (_b_vec * 0.5) + (_h_vec * 0.5)
        face._is_convex = True
        face._is_self_intersecting = False
        return face

    @classmethod
    def from_regular_polygon(cls, side_count, radius=1, base_plane=None):
        """Initialize Face3D from regular polygon parameters and a base_plane.

        Args:
            side_count: An integer for the number of sides on the regular
                polygon. This number must be greater than 2.
            radius: A number indicating the distance from the polygon's center
                where the vertices of the polygon will lie.
                The default is set to 1.
            base_plane: A Plane object for the plane in which the face exists.
                The origin of this plane will be used as the center of the polygon.
                If None, the default will be the WorldXY plane.
        """
        # set the default base_plane
        if base_plane is not None:
            assert isinstance(base_plane, Plane), 'Expected Plane. Got {}'.format(
                type(base_plane))
        else:
            base_plane = Plane(Vector3D(0, 0, 1), Point3D(0, 0, 0))

        # create the regular polygon face
        _polygon2d = Polygon2D.from_regular_polygon(side_count, radius)
        _vert3d = tuple(base_plane.xy_to_xyz(_v) for _v in _polygon2d.vertices)
        _face = cls(_vert3d, base_plane, enforce_right_hand=False)

        # assign extra properties that we know to the face
        _face._polygon2d = _polygon2d
        _face._center = base_plane.o
        _face._centroid = base_plane.o
        _face._is_convex = True
        _face._is_self_intersecting = False
        return _face

    @classmethod
    def from_punched_geometry(cls, base_face, sub_faces):
        """Create a face with holes punched in it from sub-faces.

        Args:
            base_face: A Face3D that acts as a parent to the sub_faces, completely
                encircling them.
            sub_faces: A list of Face3D objects that will be punched into the
                base_face. These faces must lie completely within the base_face
                for the result to be valid. The is_sub_face() method can be
                used to check sub_faces before they are input here.
        """
        assert isinstance(base_face, Face3D), \
            'base_face should be a Face3D. Got {}'.format(type(base_face))
        for hole in sub_faces:
            assert isinstance(hole, Face3D), \
                'sub_face should be a list. Got {}'.format(type(hole))
        hole_verts = [list(sf.boundary) for sf in sub_faces]
        if base_face.has_holes:
            hole_verts.extend([list(h) for h in base_face.holes])
        return cls(base_face.boundary, base_face.plane, hole_verts,
                   enforce_right_hand=False)

    @property
    def vertices(self):
        """Tuple of all vertices in this face.

        Note that, in the case of a face with holes, some vertices will be repeated
        since this property effectively traces out a single boundary around the
        whole shape, winding inward to cut out the holes.
        """
        return self._vertices

    @property
    def plane(self):
        """Tuple of all vertices in this face."""
        return self._plane

    @property
    def polygon2d(self):
        """A Polygon2D of this face in the 2D space of the face's plane.

        Note that this is a single polygon object even when there are holes in the
        face since such a polygon can be made by drawing a line from the holes to
        the outer boundary.
        """
        if self._polygon2d is None:
            _vert2d = tuple(self._plane.xyz_to_xy(_v) for _v in self.vertices)
            self._polygon2d = Polygon2D(_vert2d)
        return self._polygon2d

    @property
    def triangulated_mesh2d(self):
        """A triangulated Mesh2D in the 2D space of the face's plane."""
        if self._mesh2d is None:
            self._mesh2d = Mesh2D.from_polygon_triangulated(
                self.boundary_polygon2d, self.hole_polygon2d)
        return self._mesh2d

    @property
    def triangulated_mesh3d(self):
        """A triangulated Mesh3D of this face."""
        if self._mesh3d is None:
            _vert3d = tuple(self._plane.xy_to_xyz(_v) for _v in
                            self.triangulated_mesh2d.vertices)
            self._mesh3d = Mesh3D(_vert3d, self.triangulated_mesh2d.faces)
        return self._mesh3d

    @property
    def boundary(self):
        """Tuple of vertices on the boundary of this face.

        For most Face3D objects, this will be identical to the vertices property.
        However, when the Face3D has holes within it, this property stores
        the outer boundary of the shape.
        """
        return self._boundary

    @property
    def holes(self):
        """Tuple with one tuple of vertices for each hole within this face.

        This property will be None when the face has no holes in it.
        """
        return self._holes

    @property
    def boundary_segments(self):
        """Tuple of all line segments bordering the face.

        Note that this does not include segments for any holes in the face.
        Just the outer boundary.
        """
        if self._boundary_segments is None:
            _segs = []
            for i, vert in enumerate(self.boundary):
                _seg = LineSegment3D.from_end_points(self.boundary[i - 1], vert)
                _segs.append(_seg)
            _segs.append(_segs.pop(0))  # segments will start from the first vertex
            self._boundary_segments = tuple(_segs)
        return self._boundary_segments

    @property
    def hole_segments(self):
        """Tuple with a tuple of line segments for each hole in the face.

        This will be None if there are no holes in the face.
        """
        if self._holes is not None and self._hole_segments is None:
            _all_segs = []
            for hole in self.holes:
                _segs = []
                for i, vert in enumerate(hole):
                    _seg = LineSegment3D.from_end_points(hole[i - 1], vert)
                    _segs.append(_seg)
                _segs.append(_segs.pop(0))  # segments will start from the first vertex
                _all_segs.append(_segs)
            self._hole_segments = tuple(tuple(_s) for _s in _all_segs)
        return self._hole_segments

    @property
    def boundary_polygon2d(self):
        """A Polygon2D of the face boundary in the 2D space of the face's plane.

        Note that this does not include any holes in the face. Just the outer boundary.
        """
        if self._boundary_polygon2d is None:
            _vert2d = tuple(self._plane.xyz_to_xy(_v) for _v in self.boundary)
            self._boundary_polygon2d = Polygon2D(_vert2d)
        return self._boundary_polygon2d

    @property
    def hole_polygon2d(self):
        """A list of Polygon2D for the face holes in the 2D space of the face's plane.
        """
        if self._holes is not None and self._hole_polygon2d is None:
            self._hole_polygon2d = []
            for hole in self.holes:
                _vert2d = tuple(self._plane.xyz_to_xy(_v) for _v in hole)
                self._hole_polygon2d.append(Polygon2D(_vert2d))
        return self._hole_polygon2d

    @property
    def normal(self):
        """Normal vector for the plane in which the face exists."""
        return self._plane.n

    @property
    def perimeter(self):
        """The perimeter of the face. This includes the length of holes in the face."""
        if self._perimeter is None:
            self._perimeter = sum([seg.length for seg in self.boundary_segments])
            if self._holes is not None:
                for hole in self.hole_segments:
                    self._perimeter += sum([seg.length for seg in hole])
        return self._perimeter

    @property
    def area(self):
        """The area of the face."""
        if self._area is None:
            self._area = self.polygon2d.area
        return self._area

    @property
    def centroid(self):
        """The centroid of the face as a Point3D (aka. center of mass).

        Note that the centroid is more time consuming to compute than the center
        (or the middle point of the face bounding box). So the center might be
        preferred over the centroid if you just need a rough point for the middle
        of the face.
        """
        if self._centroid is None:
            _cent2d = self.triangulated_mesh2d.centroid
            self._centroid = self._plane.xy_to_xyz(_cent2d)
        return self._centroid

    @property
    def azimuth(self):
        """Get the azimuth of the Face3D (between 0 and 2 * Pi).

        This will be zero if the Face3D is perfectly horizontal.
        """
        return self.plane.azimuth

    @property
    def altitude(self):
        """Get the altitude of the Face3D (between Pi/2 and -Pi/2)."""
        return self.plane.altitude

    @property
    def is_clockwise(self):
        """Boolean for whether the face vertices and boundary are in clockwise order.

        Note that all Face3D objects should have counterclockwise vertices (meaning
        that this property should always be False). This property exists largely
        for testing / debugging purposes.
        """
        return self.polygon2d.is_clockwise

    @property
    def is_convex(self):
        """Boolean noting whether the face is convex (True) or non-convex (False).

        Note that any face with holes will be automatically considered non-convex
        since the underlying polygon_2d is always non-convex in this case.
        """
        if self._is_convex is None:
            self._is_convex = self.polygon2d.is_convex
        return self._is_convex

    @property
    def is_self_intersecting(self):
        """Boolean noting whether the face has self-intersecting edges.

        Note that this property is relatively computationally intense to obtain and
        most CAD programs forbid all surfaces with self-intersecting edges.
        So this property should only be used in quality control scripts where the
        origin of the geometry is unknown.
        """
        if self._is_self_intersecting is None:
            self._is_self_intersecting = False
            if self.boundary_polygon2d.is_self_intersecting:
                self._is_self_intersecting = True
            if self.has_holes:
                for hp in self.hole_polygon2d:
                    if hp.is_self_intersecting:
                        self._is_self_intersecting = True
                        break
        return self._is_self_intersecting

    @property
    def is_valid(self):
        """Boolean noting whether the face is valid (having a non-zero area).

        Note that faces are still considered valid if they have out-of-plane vertices,
        self-intersecting edges, or duplicate/colinear vertices. The check_planar
        method can be used to detect if there are out-of-plane vertices. The
        is_self_intersecting property identifies self-intersecting edges, and the
        remove_colinear_vertices method will remove duplicate/colinear vertices."""
        return not self.area == 0

    @property
    def has_holes(self):
        """Boolean noting whether the face has holes within it."""
        return self._holes is not None

    @property
    def upper_left_corner(self):
        """Get the vertex in the upper-left corner of the face's bounding box."""
        return self._corner_point('min', 'max')

    @property
    def lower_left_corner(self):
        """Get the vertex in the lower-left corner of the face's bounding box."""
        return self._corner_point('min', 'min')

    @property
    def upper_right_corner(self):
        """Get the vertex in the upper-right corner of the face's bounding box."""
        return self._corner_point('max', 'max')

    @property
    def lower_right_corner(self):
        """Get the vertex in the lower-right corner of the face's bounding box."""
        return self._corner_point('max', 'min')

    @property
    def upper_left_counter_clockwise_vertices(self):
        """Get face vertices starting from the upper left and moving counterclockwise.

        Horizontal faces will treat the positive Y axis as up. All other faces
        treat the positive Z axis as up.
        """
        corner_pt, polygon = self._corner_point_and_polygon(self._vertices, 'min', 'max')
        verts3d, verts2d = self._counter_clockwise_verts(polygon)
        return self._corner_pt_verts(corner_pt, verts3d, verts2d)

    @property
    def lower_left_counter_clockwise_vertices(self):
        """Get face vertices starting from the lower left and moving counterclockwise.

        Horizontal faces will treat the positive Y axis as up. All other faces
        treat the positive Z axis as up.
        """
        corner_pt, polygon = self._corner_point_and_polygon(self._vertices, 'min', 'min')
        verts3d, verts2d = self._counter_clockwise_verts(polygon)
        return self._corner_pt_verts(corner_pt, verts3d, verts2d)

    @property
    def lower_right_counter_clockwise_vertices(self):
        """Get face vertices starting from the lower left and moving counterclockwise.

        Horizontal faces will treat the positive Y axis as up. All other faces
        treat the positive Z axis as up.
        """
        corner_pt, polygon = self._corner_point_and_polygon(self._vertices, 'max', 'min')
        verts3d, verts2d = self._counter_clockwise_verts(polygon)
        return self._corner_pt_verts(corner_pt, verts3d, verts2d)

    @property
    def upper_right_counter_clockwise_vertices(self):
        """Get face vertices starting from the lower left and moving counterclockwise.

        Horizontal faces will treat the positive Y axis as up. All other faces
        treat the positive Z axis as up.
        """
        corner_pt, polygon = self._corner_point_and_polygon(self._vertices, 'max', 'max')
        verts3d, verts2d = self._counter_clockwise_verts(polygon)
        return self._corner_pt_verts(corner_pt, verts3d, verts2d)

    def pole_of_inaccessibility(self, tolerance):
        """Get the pole of inaccessibility for the Face3D.

        The pole of inaccessibility is the most distant internal point from the
        Face3D outline. It is not to be confused with the centroid, which
        represents the "center of mass" of the shape and may be outside of
        the Face3D if the shape is concave. The poly of inaccessibility is
        useful for optimal placement of a text label on the Face3D.

        Args:
            tolerance: The precision to which the pole of inaccessibility
                will be computed.
        """
        return self.plane.xy_to_xyz(self.polygon2d.pole_of_inaccessibility(tolerance))

    def is_horizontal(self, tolerance):
        """Check whether a this face is horizontal within a given tolerance.

        Args:
            tolerance: The minimum difference between the coordinate values of two
                vertices at which they can be considered equivalent.

        Returns:
            True if the face is horizontal. False if it is not.
        """
        return self.max.z - self.min.z <= tolerance

    def is_geometrically_equivalent(self, face, tolerance):
        """Check whether a given face is geometrically equivalent to this Face.

        Geometrical equivalence is defined as being coplanar with this face,
        having the same number of vertices, and having each vertex map-able between
        the faces. Clockwise relationships do not have to match nor does the normal
        direction of the face. However, all other properties must be matching to
        within the input tolerance.

        This is useful for identifying matching surfaces when solving for adjacency
        and you need to ensure that two faces match perfectly in their area and vertices.
        Note that you may also want to use the remove_colinear_vertices() method
        on input faces before using this method in order to count faces with the
        same non-colinear vertices as geometrically equivalent.

        Args:
            face: Another face for which geometric equivalency will be tested.
            tolerance: The minimum difference between the coordinate values of two
                vertices at which they can be considered geometrically equivalent.

        Returns:
            True if geometrically equivalent. False if not geometrically equivalent.
        """
        # rule out surfaces if they don't fit key criteria
        if not self.center.is_equivalent(face.center, tolerance):
            return False
        if len(self.vertices) != len(face.vertices):
            return False

        # see if we can find a matching vertex
        match_i = None
        for i, pt in enumerate(self.vertices):
            if pt.is_equivalent(face[0], tolerance):
                match_i = i
                break

        # check equivalency of each vertex
        if match_i is None:
            return False
        elif self[match_i - 1].is_equivalent(face[1], tolerance):
            for i in xrange(len(self.vertices)):
                if self[match_i - i].is_equivalent(face[i], tolerance) is False:
                    return False
        elif self[match_i + 1].is_equivalent(face[1], tolerance):
            for i in xrange(0, -len(self.vertices), -1):
                if self[match_i + i].is_equivalent(face[i], tolerance) is False:
                    return False
        else:
            return False
        return True

    def is_centered_adjacent(self, face, tolerance):
        """Check whether a given face is centered adjacent with this Face.

        Centered adjacency is defined as sharing the same center point as this face
        and being next to one another to within the tolerance.

        This is useful for identifying matching faces when you want to quickly
        solve for adjacency and you are not concerned about false positives in cases
        where one face does not perfectly match the other in terms of vertex ordering.

        Args:
            face: Another face for which centered adjacency will be tested.
            tolerance: The minimum difference between the coordinate values of two
                centers at which they can be considered centered adjacent.
        Returns:
            True if centered adjacent. False if not centered adjacent.
        """
        if not self.center.is_equivalent(face.center, tolerance):  # center check
            return False
        # construct a ray using this face's normal and a point just behind this face
        point_on_face = self._point_on_face(tolerance)
        point_on_face = point_on_face - (self.normal * tolerance)  # move below
        test_ray = Ray3D(point_on_face, self.normal)
        # shoot ray from this face to the other to verify adjacency
        if face.intersect_line_ray(test_ray):
            return True
        return False

    def is_sub_face(self, face, tolerance, angle_tolerance):
        """Check whether a given face is a sub-face of this face.

        Sub-faces will lie in the same plane as this one and have all of their
        vertices completely within the boundary of this face.

        This is useful for identifying whether a given sub-face (ie. a window or door)
        can be assigned as a child to this face.

        Args:
            face: Another face for which sub-face equivalency will be tested.
            tolerance: The minimum difference between the coordinate values of two
                vertices at which they can be considered equivalent.
            angle_tolerance: The max angle in radians that the plane normals can
                differ from one another in order for them to be considered coplanar.

        Returns:
            True if it can be a valid sub-face. False if it is not a valid sub-face.
        """
        # test whether the surface is coplanar
        if not self.plane.is_coplanar_tolerance(face.plane, tolerance, angle_tolerance):
            return False

        # if it is, convert sub-face to a polygon in this face's plane
        return self._is_sub_face(face)

    def polygon_in_face(self, sub_face, origin=None, flip=False):
        """Get a Polygon2D for a sub_face within the plane of this Face3D.

        Args:
            sub_face: A Face3D for which a Polygon2D in the plane of this
                Face3D will be returned.
            origin: An optional Point3D to set the origin of the plane in which
                the sub_face will be evaluated. Plugging in values like the
                Face's lower_left_corner can standardize the geometry rules
                for the resulting polygon. If None, this face's own
                plane will be used. (Default: None).
            flip: Boolean to note whether the x-axis of the plane should be flipped
                when translating this the sub_face vertices.
        """
        # set the process the origin into a plane
        if origin is None:
            plane = self.plane if not flip else self.plane.flip()
        else:
            if self._plane.n.z in (1, -1):
                plane = Plane(self._plane.n, origin, Vector3D(1, 0, 0)) if not flip \
                    else Plane(self._plane.n, origin, Vector3D(-1, 0, 0))
            else:
                proj_y = Vector3D(0, 0, 1).project(self._plane.n)
                proj_x = proj_y.rotate(self._plane.n, math.pi / -2)
                plane = Plane(self._plane.n, origin, proj_x)
        pts_2d = tuple(plane.xyz_to_xy(pt) for pt in sub_face.boundary)
        return Polygon2D(pts_2d)

    def is_point_on_face(self, point, tolerance):
        """Check whether a given point is on this face.

        This includes both a check to be sure that the point is in the plane of this
        face and a check to ensure that point lies in the boundary of the face.

        Args:
            point: A Point3D to evaluate whether it lies on the face.
            tolerance: The minimum difference between the coordinate values of two
                vertices at which they can be considered equivalent.
        Returns:
            True if the point is on the face. False if it is not.
        """
        # test whether the point is in the plane of the face
        if self.plane.distance_to_point(point) > tolerance:
            return False
        # if it is, convert the point into this face's plane
        vert2d = self.plane.xyz_to_xy(point)
        return self.polygon2d.is_point_inside(vert2d)

    def check_planar(self, tolerance, raise_exception=True):
        """Check that all of the face's vertices lie within the face's plane.

        This check is not done by default when creating the face since
        it is assumed that there is likely a check for planarity before the face
        is created (ie. in CAD software where the face likely originates from).
        This method is intended for quality control checking when the origin of
        face geometry is unknown or is known to come from a place where no
        planarity check was performed.

        Args:
            tolerance: The minimum distance between a given vertex and a the
                face's plane at which the vertex is said to lie in the plane.
            raise_exception: Boolean to note whether an exception should be raised
                if a vertex does not lie within the face's plane. If True, an
                exception message will be given in such cases, which notes the non-planar
                vertex and its distance from the plane. If False, this method will
                simply return a False boolean if a vertex is found that is out
                of plane. Default is True to raise an exception.

        Return:
            True if planar within the tolerance. False if not planar.
        """
        for _v in self.vertices:
            if self._plane.distance_to_point(_v) >= tolerance:
                if raise_exception is True:
                    raise ValueError(
                        'Vertex {} is out of plane with its parent face.\nDistance '
                        'to plane is {}'.format(_v, self._plane.distance_to_point(_v)))
                else:
                    return False
        return True

    def remove_duplicate_vertices(self, tolerance):
        """Get a version of this face without duplicate vertices.

        Args:
            tolerance: The minimum distance between a two vertices at which
                they are considered co-located or duplicated.
        """
        if not self.has_holes:  # we only need to evaluate one list of vertices
            new_vertices = tuple(
                pt for i, pt in enumerate(self._vertices)
                if not pt.is_equivalent(self._vertices[i - 1], tolerance))
            _new_face = Face3D(new_vertices, self.plane, enforce_right_hand=False)
            return _new_face
        # the face has holes
        _boundary = tuple(
            pt for i, pt in enumerate(self._boundary)
            if not pt.is_equivalent(self._boundary[i - 1], tolerance))
        _holes = tuple(
            tuple(p for i, p in enumerate(h) if not p.is_equivalent(h[i - 1], tolerance))
            for j, h in enumerate(self._holes))
        _new_face = Face3D(_boundary, self.plane, _holes, enforce_right_hand=False)
        return _new_face

    def remove_colinear_vertices(self, tolerance):
        """Get a version of this face without colinear or duplicate vertices.

        Args:
            tolerance: The minimum distance between a vertex and the boundary segments
                at which point the vertex is considered colinear.
        """
        if not self.has_holes:  # we only need to evaluate one list of vertices
            new_vertices = self._remove_colinear(
                self._vertices, self.polygon2d, tolerance)
            _new_face = Face3D(new_vertices, self.plane, enforce_right_hand=False)
            return _new_face
        # the face has holes
        _boundary = self._remove_colinear(
            self._boundary, self.boundary_polygon2d, tolerance)
        _holes = tuple(self._remove_colinear(hole, self.hole_polygon2d[i], tolerance)
                       for i, hole in enumerate(self._holes))
        _new_face = Face3D(_boundary, self.plane, _holes, enforce_right_hand=False)
        return _new_face

    def flip(self):
        """Get a face with a flipped direction from this one."""
        _new_face = Face3D(reversed(self.vertices), self.plane.flip(),
                           enforce_right_hand=False)
        self._transfer_properties(_new_face)
        if self._holes is not None:
            _new_face._boundary = tuple(reversed(self._boundary))
            _new_face._holes = self._holes
        return _new_face

    def move(self, moving_vec):
        """Get a face that has been moved along a vector.

        Args:
            moving_vec: A Vector3D with the direction and distance to move the face.
        """
        _verts = self._move(self.vertices, moving_vec)
        _new_face = self._face_transform(_verts, self.plane.move(moving_vec))
        if self._holes is not None:
            _new_face._boundary = self._move(self._boundary, moving_vec)
            _new_face._holes = tuple(self._move(hole, moving_vec)
                                     for hole in self._holes)
        return _new_face

    def rotate(self, axis, angle, origin):
        """Rotate a face by a certain angle around an axis and origin.

        Right hand rule applies:
        If axis has a positive orientation, rotation will be clockwise.
        If axis has a negative orientation, rotation will be counterclockwise.

        Args:
            axis: A Vector3D axis representing the axis of rotation.
            angle: An angle for rotation in radians.
            origin: A Point3D for the origin around which the object will be rotated.
        """
        _verts = self._rotate(self.vertices, axis, angle, origin)
        _new_face = self._face_transform(_verts, self.plane.rotate(axis, angle, origin))
        if self._holes is not None:
            _new_face._boundary = self._rotate(self._boundary, axis, angle, origin)
            _new_face._holes = tuple(self._rotate(hole, axis, angle, origin)
                                     for hole in self._holes)
        return _new_face

    def rotate_xy(self, angle, origin):
        """Get a face rotated counterclockwise in the world XY plane by a certain angle.

        Args:
            angle: An angle in radians.
            origin: A Point3D for the origin around which the object will be rotated.
        """
        _verts = self._rotate_xy(self.vertices, angle, origin)
        _new_face = self._face_transform(_verts, self.plane.rotate_xy(angle, origin))
        if self._holes is not None:
            _new_face._boundary = self._rotate_xy(self._boundary, angle, origin)
            _new_face._holes = tuple(self._rotate_xy(hole, angle, origin)
                                     for hole in self._holes)
        return _new_face

    def reflect(self, normal, origin):
        """Get a face reflected across a plane with the input normal vector and origin.

        Args:
            normal: A Vector3D representing the normal vector for the plane across
                which the face will be reflected. THIS VECTOR MUST BE NORMALIZED.
            origin: A Point3D representing the origin from which to reflect.
        """
        _verts = self._reflect(self.vertices, normal, origin)
        _new_face = self._face_transform_reflect(
            _verts, self.plane.reflect(normal, origin))
        if self._holes is not None:
            _new_face._boundary = self._reflect(self._boundary, normal, origin)
            _new_face._holes = tuple(self._reflect(hole, normal, origin)
                                     for hole in self._holes)
        return _new_face

    def scale(self, factor, origin=None):
        """Scale a face by a factor from an origin point.

        Args:
            factor: A number representing how much the face should be scaled.
            origin: A Point3D representing the origin from which to scale.
                If None, it will be scaled from the World origin (0, 0, 0).
        """
        _verts = self._scale(self.vertices, factor, origin)
        _new_face = self._face_transform_scale(_verts, None, factor)
        if self._holes is not None:
            _new_face._boundary = self._scale(self._boundary, factor, origin)
            _new_face._holes = tuple(self._scale(hole, factor, origin)
                                     for hole in self._holes)
        return _new_face

    def split_through_holes(self):
        """Get this Face3D split through its holes to get Face3D without holes.

        This method attempts to return the minimum number of non-holed shapes that
        are needed to represent the original Face3D. If this fails, the result
        will be derived from a triangulated shape. If getting a minimum number
        of constituent Face3D is not important, it is more efficient to just
        use all of the triangles in Face3D.triangulated_mesh3d instead of the
        result of this method.

        Returns:
            A list of Face3D without holes that together form a geometric
            representation of this Face3D. If this Face3D has no holes a list
            with a single Face3D is returned.
        """
        def _shared_vertex_count(vert_set, verts):
            """Get the number of shared vertices."""
            in_set = tuple(v for v in verts if v in vert_set)
            return len(in_set)

        def _shared_edge_count(edge_set, verts):
            """Get the number of shared edges."""
            edges = tuple((verts[i], verts[i - 1]) for i in range(3))
            in_set = tuple(e for e in edges if e in edge_set)
            return len(in_set)

        if not self.has_holes:
            return (self,)
        # check that the direction of vertices for the hole is opposite the boundary
        boundary = list(self.boundary_polygon2d.vertices)
        holes = [list(hole.vertices) for hole in self.hole_polygon2d]
        bound_direction = Polygon2D._are_clockwise(boundary)
        for hole in holes:
            if Polygon2D._are_clockwise(hole) is bound_direction:
                hole.reverse()
        # try to split the polygon neatly in two
        s_result = Polygon2D._merge_boundary_and_holes(boundary, holes, split=True)
        if s_result is not None:
            poly_1, poly_2 = s_result
            vert_1 = tuple(self.plane.xy_to_xyz(pt) for pt in poly_1)
            vert_2 = tuple(self.plane.xy_to_xyz(pt) for pt in poly_2)
            face_1 = Face3D(vert_1, plane=self.plane)
            face_2 = Face3D(vert_2, plane=self.plane)
            return face_1, face_2
        # if splitting in two did not work, then triangulate it and merge them together
        tri_mesh = self.triangulated_mesh3d
        tri_verts = tri_mesh.vertices
        rel_f = tri_mesh.faces[0]
        tri_faces = [[tuple(tri_verts[pt] for pt in rel_f)]]
        tri_face_sets = [set(rel_f)]
        tri_edge_sets = [set((rel_f[i - 1], rel_f[i]) for i in range(3))]
        faces_to_test = list(tri_mesh.faces[1:])
        # group the faces along matched edges
        for f in faces_to_test:
            connected = False
            for tfs, fs, es in zip(tri_faces, tri_face_sets, tri_edge_sets):
                svc = _shared_vertex_count(fs, f)
                sec = _shared_edge_count(es, f)
                if svc == 2 and sec == 1:  # matched edge
                    tfs.append(tuple(tri_verts[pt] for pt in f))
                    for i, v in enumerate(f):
                        fs.add(v)
                        es.add((f[i - 1], f[i]))
                    break
                elif svc == 3:  # definitely a new shape
                    connected = True
            else:  # not ready to be merged; put it to the back
                if connected:
                    tri_faces.append([tuple(tri_verts[pt] for pt in f)])
                    tri_face_sets.append(set(f))
                    tri_edge_sets.append(set((f[i - 1], f[i]) for i in range(3)))
                else:
                    faces_to_test.append(f)
        # create Face3Ds from the triangle groups
        final_faces = []
        for tf in tri_faces:
            t_mesh = Mesh3D.from_face_vertices(tf)
            ed_len = (seg.length for seg in t_mesh.naked_edges)
            tol = min(ed_len) / 10
            f_bound = Polyline3D.join_segments(t_mesh.naked_edges, tol)
            final_faces.append(Face3D(f_bound[0].vertices, plane=self.plane))
        return final_faces

    def intersect_line_ray(self, line_ray):
        """Get the intersection between this face and the input Line3D or Ray3D.

        Args:
            line_ray: A Line3D or Ray3D object for which intersection will be computed.

        Returns:
            Point3D for the intersection. Will be None if no intersection exists.
        """
        _plane_int = self._plane.intersect_line_ray(line_ray)
        if _plane_int is not None:
            _int2d = self._plane.xyz_to_xy(_plane_int)
            if self.polygon2d.is_point_inside_bound_rect(_int2d):
                return _plane_int
        return None

    def intersect_plane(self, plane):
        """Get the intersection between this face and the input plane.

        Args:
            plane: A Plane object for which intersection will be computed.

        Returns:
            List of LineSegment3D objects for the intersection.
            Will be None if no intersection exists.
        """
        _plane_int_ray = self._plane.intersect_plane(plane)
        if _plane_int_ray is not None:
            _p12d = self._plane.xyz_to_xy(_plane_int_ray.p)
            _p22d = self._plane.xyz_to_xy(_plane_int_ray.p + _plane_int_ray.v)
            _v2d = _p22d - _p12d
            _int_ray2d = Ray2D(_p12d, _v2d)
            _int_pt2d = self.polygon2d.intersect_line_infinite(_int_ray2d)
            if len(_int_pt2d) != 0:
                if len(_int_pt2d) > 2:  # sort the points along the intersection line
                    _int_pt2d.sort(key=lambda pt: pt.x)
                _int_pt3d = [self._plane.xy_to_xyz(pt) for pt in _int_pt2d]
                _int_seg3d = []
                for i in xrange(0, len(_int_pt3d) - 1, 2):
                    _int_seg3d.append(LineSegment3D.from_end_points(
                        _int_pt3d[i], _int_pt3d[i + 1]))
                return _int_seg3d
        return None

    def project_point(self, point):
        """Project a Point3D onto this face.

        Note that this method does a check to see if the point can be projected to
        within this face's boundary. If all that is needed is a point projected
        into the plane of this face, the Plane.project_point() method should be
        used with this face's plane property.

        Args:
            point: A Point3D object to project.

        Returns:
            Point3D for the point projected onto this face. Will be None if the
            point cannot be projected to within the boundary of the face.
        """
        _plane_int = point.project(self._plane.n, self._plane.o)
        _plane_int2d = self._plane.xyz_to_xy(_plane_int)
        if self.polygon2d.is_point_inside_bound_rect(_plane_int2d):
            return _plane_int
        return None

    def mesh_grid(self, x_dim, y_dim=None, offset=None, flip=False,
                  generate_centroids=True):
        """Get a gridded Mesh3D over this face.

        This method generates a mesh grid over the domain of the face
        and then removes any vertices that do not lie within it.

        Note that the x_dim and y_dim refer to dimensions within the X and Y
        coordinate system of this faces's plane. So rotating this plane will
        result in rotated grid cells.

        Args:
            x_dim: The x dimension of the grid cells as a number.
            y_dim: The y dimension of the grid cells as a number. Default is None,
                which will assume the same cell dimension for y as is set for x.
            offset: A number for how far to offset the grid from the base face.
                Default is None, which will not offset the grid at all.
            flip: Set to True to have the mesh normals reversed from the direction
                of this face and to have the offset input move the mesh in the
                opposite direction from this face's normal.
            generate_centroids: Set to True to have the face centroids generated
                alongside the grid of vertices, which is much faster than having
                them generated upon request as they typically are. However, if you
                have no need for the face centroids, you would save time and memory
                by setting this to False. Default is True.
        """
        # check the inputs and set defaults
        self._check_number_mesh_grid(x_dim, 'x_dim')
        if y_dim is not None:
            self._check_number_mesh_grid(y_dim, 'y_dim')
        else:
            y_dim = x_dim
        if offset is not None:
            self._check_number_mesh_grid(offset, 'offset')

        # generate the mesh grid and convert it to a 3D mesh
        grid_mesh2d = Mesh2D.from_polygon_grid(
            self.polygon2d, x_dim, y_dim, generate_centroids)
        if offset is None or offset == 0:
            vert_3d = tuple(self._plane.xy_to_xyz(pt)
                            for pt in grid_mesh2d.vertices)
        else:
            _off_num = -1 * offset if flip is True else offset
            _off_plane = self.plane.move(self.plane.n * _off_num)
            vert_3d = tuple(_off_plane.xy_to_xyz(pt)
                            for pt in grid_mesh2d.vertices)
        grid_mesh3d = Mesh3D(vert_3d, grid_mesh2d.faces)
        grid_mesh3d._face_areas = grid_mesh2d._face_areas

        # assign the face plane normal to the mesh normals
        if flip is True:
            grid_mesh3d._face_normals = self._plane.n.reverse()
            grid_mesh3d._vertex_normals = self._plane.n.reverse()
            grid_mesh3d._faces = tuple(
                tuple(reversed(face)) for face in grid_mesh3d._faces)  # right-hand rule
        else:
            grid_mesh3d._face_normals = self._plane.n
            grid_mesh3d._vertex_normals = self._plane.n

        # transform the centroids to 3D space if they were generated
        if generate_centroids is True:
            _conv_plane = self._plane if offset is None or offset == 0 else _off_plane
            grid_mesh3d._face_centroids = tuple(_conv_plane.xy_to_xyz(pt)
                                                for pt in grid_mesh2d.face_centroids)

        return grid_mesh3d

    def contour_by_number(self, contour_count, direction_vector, flip_side, tolerance):
        """Generate a list of LineSegment3D objects contouring the face.

        Args:
            contour_count: A positive integer for the number of contours
                to generate over the face.
            direction_vector: A Vector2D for the direction along which contours
                are generated. This 2D vector will be interpreted into a 3D vector
                within the plane of this Face. (0, 1) will usually generate
                horizontal contours in 3D space, (1, 0) will generate vertical
                contours, and (1, 1) will generate diagonal contours. Recommended
                value is Vector2D(0, 1).
            flip_side: Boolean to note whether the side the contours start from
                should be flipped. Recommended value is False to have contours
                on top or right.
                Setting to True will start contours on the bottom or left.
            tolerance: An optional value to remove any contours with a length less
                than the tolerance.
        """
        # interpret the 2D direction_vector into one that exists in 3D space
        ref_plane = Plane(self._plane.n, Point3D(0, 0, 0), self._plane.x)
        if ref_plane.y.z < 0:
            ref_plane = ref_plane.rotate(ref_plane.n, math.pi, ref_plane.o)
        plane_normal = ref_plane.xy_to_xyz(direction_vector).normalize()

        # get a diagonal going across the face
        diagonal = self._diagonal_along_self(direction_vector, tolerance)
        if not flip_side:
            diagonal = diagonal.flip()  # flip diagonal if user has requested it

        # generate the contours
        contours = []
        for pt in diagonal.subdivide_evenly(contour_count)[:-1]:
            result = self.intersect_plane(Plane(plane_normal, pt))
            if result is not None:
                contours.extend(result)

        # remove any contours that are smaller than the tolerance.
        if tolerance != 0:
            contours = [l_seg for l_seg in contours if l_seg.length >= tolerance]
        return contours

    def contour_by_distance_between(self, distance, direction_vector, flip_side,
                                    tolerance):
        """Generate a list of LineSegment3D objects contouring the face.

        Args:
            distance: A number for the distance between each contour.
            direction_vector: A Vector2D for the direction along which contours
                are generated. This 2D vector will be interpreted into a 3D vector
                within the plane of this Face. (0, 1) will usually generate
                horizontal contours in 3D space, (1, 0) will generate vertical
                contours, and (1, 1) will generate diagonal contours. Recommended
                value is Vector2D(0, 1).
            flip_side: Boolean to note whether the side the contours start from
                should be flipped. Recommended value is is False to have contours
                start on top or right. Setting to True will start contours on
                the bottom or left.
            tolerance: An optional value to remove any contours with a length less
                than the tolerance.
        """
        # interpret the 2D direction_vector into one that exists in 3D space
        ref_plane = Plane(self._plane.n, Point3D(0, 0, 0), self._plane.x)
        if ref_plane.y.z < 0:
            ref_plane = ref_plane.rotate(ref_plane.n, math.pi, ref_plane.o)
        plane_normal = ref_plane.xy_to_xyz(direction_vector).normalize()

        # get a diagonal going across the face
        diagonal = self._diagonal_along_self(direction_vector, tolerance)
        if not flip_side:
            diagonal = diagonal.flip()  # flip diagonal if user has requested it

        # compute the diagonal subdivision distance using the plane_normal
        angle = plane_normal.angle(diagonal.v)
        angle = abs(angle - math.pi) if angle > math.pi / 2 else angle
        proj_dist = distance / math.cos(angle)

        # generate the contours
        contours = []
        for pt in diagonal.subdivide(proj_dist)[:-1]:
            pass
            result = self.intersect_plane(Plane(plane_normal, pt))
            if result is not None:
                contours.extend(result)

        # remove any contours that are smaller than the tolerance.
        if tolerance != 0:
            contours = [l_seg for l_seg in contours if l_seg.length >= tolerance]
        return contours

    def contour_fins_by_number(self, fin_count, depth, offset, angle,
                               contour_vector, flip_side, tolerance):
        """Generate a list of Fac3D objects over this face (like louvers or fins).

        Args:
            fin_count: A positive integer for the number of fins to generate.
            depth: A number for the depth to extrude the fins.
            offset: A number for the distance to offset fins from this face.
                Recommended value is 0 for no offset.
            angle: A number for the for an angle to rotate the fins in radians.
                Recommended value is 0 for no rotation.
            contour_vector: A Vector2D for the direction along which contours
                are generated. This 2D vector will be interpreted into a 3D vector
                within the plane of this Face. (0, 1) will usually generate
                horizontal contours in 3D space, (1, 0) will generate vertical
                contours, and (1, 1) will generate diagonal contours. Recommended
                value is Vector2D(0, 1).
            flip_side: Boolean to note whether the side the fins start from
                should be flipped. Recommended value is False to have contours
                start on top or right. Setting to True will start contours on
                the bottom or left.
            tolerance: An optional value to remove any contours with a length less
                than the tolerance.
        """
        extru_vec = self._get_fin_extrusion_vector(depth, angle, contour_vector)
        contours = self.contour_by_number(
            fin_count, contour_vector, flip_side, tolerance)
        return self._get_extrusion_fins(contours, extru_vec, offset)

    def contour_fins_by_distance_between(self, distance, depth, offset, angle,
                                         contour_vector, flip_side, tolerance):
        """Generate a list of Fac3D objects over this face (like louvers or fins).

        Args:
            distance: A number for the approximate distance between each contour.
            depth: A number for the depth to extrude the fins.
            offset: A number for the distance to offset fins from this face.
                Recommended value is 0 for no offset.
            angle: A number for the for an angle to rotate the fins in radians.
                Recommended value is 0 for no rotation.
            contour_vector: A Vector2D for the direction along which contours
                are generated. This 2D vector will be interpreted into a 3D vector
                within the plane of this Face. (0, 1) will usually generate
                horizontal contours in 3D space, (1, 0) will generate vertical
                contours, and (1, 1) will generate diagonal contours. Recommended
                value is Vector2D(0, 1).
            flip_side: Boolean to note whether the side the fins start from
                should be flipped. Recommended value is False to have contours
                start on top or right. Setting to True will start contours on
                the bottom or left.
            tolerance: An optional value to remove any contours with a length less
                than the tolerance.
        """
        extru_vec = self._get_fin_extrusion_vector(depth, angle, contour_vector)
        contours = self.contour_by_distance_between(
            distance, contour_vector, flip_side, tolerance)
        return self._get_extrusion_fins(contours, extru_vec, offset)

    def sub_faces_by_ratio(self, ratio):
        """Get a list of faces with a combined area equal to ratio times this face area.

        All sub faces will lie inside the boundaries of this face and will have
        the same normal as this face.

        Args:
            ratio: A number between 0 and 1 for the ratio between the area of
                the sub faces and the area of this face.

        Returns:
            A list of Face3D objects for sub faces.
        """
        scale_factor = ratio ** .5
        if self.is_convex:
            return [self.scale(scale_factor, self.centroid)]
        else:
            _tri_mesh = self.triangulated_mesh3d
            _tri_faces = [[_tri_mesh[i] for i in face] for face in _tri_mesh.faces]
            _scaled_verts = []
            for i, _tri in enumerate(_tri_faces):
                _scaled_verts.append(
                    [pt.scale(scale_factor, _tri_mesh.face_centroids[i]) for pt in _tri])
            return [Face3D(_t, self.plane) for _t in _scaled_verts]

    def sub_faces_by_ratio_gridded(self, ratio, x_dim, y_dim=None):
        """Get a list of faces with a combined area equal to ratio times this face area.

        All sub faces will lie inside the boundaries of this face and have the same
        normal as this face.

        Sub faces will be arranged in a grid derived from this face's plane property.
        Because the x_dim and y_dim refer to dimensions within the X and Y
        coordinate system of this faces's plane, rotating this plane will
        result in rotated grid cells.

        If the x_dim and/or y_dim are too large for this face, this method will
        return essentially the same result as the sub_faces_by_ratio method.

        Args:
            ratio: A number between 0 and 1 for the ratio between the area of
                the sub faces and the area of this face.
            x_dim: The x dimension of the grid cells as a number.
            y_dim: The y dimension of the grid cells as a number. Default is None,
                which will assume the same cell dimension for y as is set for x.

        Returns:
            A list of Face3D objects for sub faces.
        """
        try:  # get the gridded mesh derived from this face
            grid_mesh = self.mesh_grid(x_dim, y_dim)
        except AssertionError:  # there are no faces; just return sub_faces_by_ratio
            return self.sub_faces_by_ratio(ratio)

        # compute the area that each of the mesh faces need to be scaled to
        _verts, _faces = grid_mesh.vertices, grid_mesh.faces
        _x_dim = _verts[_faces[0][0]].distance_to_point(_verts[_faces[0][1]])
        _y_dim = _verts[_faces[0][1]].distance_to_point(_verts[_faces[0][2]])
        fac = (self.area * ratio) / (_x_dim * _y_dim * len(_faces))

        # if the factor is greater than 1, sub-faces will be overlapping
        if fac >= 1:
            return self.sub_faces_by_ratio(ratio)
        s_fac = fac ** 0.5

        # generate the Face3D objects while scaling them to the correct size
        sub_faces = []
        for face, centr in zip(_faces, grid_mesh.face_centroids):
            _f = Face3D(tuple(_verts[i].scale(s_fac, centr) for i in face), self.plane)
            if self._is_sub_face(_f):  # catch edge cases
                sub_faces.append(_f)
        return sub_faces

    def sub_faces_by_ratio_rectangle(self, ratio, tolerance):
        """Get a list of faces with a combined area equal to ratio times this face area.

        This function is virtually equivalent to the sub_faces_by_ratio method
        but a check will be performed to see if any rectangles can be pulled out
        of this face's geometry. This tends to make the result a bit cleaner,
        especially for concave faces that have rectangles (like L-shaped faces).

        Args:
            ratio: A number between 0 and 1 for the ratio between the area of
                the sub faces and the area of this face.
            tolerance: The maximum difference between point values for them to be
                considered a part of a rectangle.

        Returns:
            A list of Face3D objects for sub faces. If there is a rectangle in this
            shape, the scaled rectangle will be the first item in this list.
        """
        rect_res = self.extract_rectangle(tolerance)
        if rect_res is None:
            return self.sub_faces_by_ratio(ratio)
        bottom_seg, top_seg, other_faces = rect_res
        rect_face = Face3D((bottom_seg.p1, bottom_seg.p2, top_seg.p2, top_seg.p1),
                           self.plane)
        scale_factor = ratio ** .5
        sub_faces = [rect_face.scale(scale_factor, rect_face.center)]
        for face in other_faces:
            sfs = face.sub_faces_by_ratio(ratio)
            for sf in sfs:
                if sf.area > tolerance:
                    sub_faces.append(sf)
        return sub_faces

    def sub_faces_by_ratio_sub_rectangle(self, ratio, sub_rect_height, sill_height,
                                         horizontal_separation, vertical_separation,
                                         tolerance):
        """Get a list of faces with a combined area equal to ratio times this face area.

        This function is virtually equivalent to the sub_faces_by_ratio_rectangle
        method but any rectangles that are found will be broken down into sub-rectangles
        using the other inputs (sub_rect_height, sill_height, horizontal_separation,
        vertical_separation). This allows for the creation of a wide array of
        rectangular sub-face geometries.

        Args:
            ratio: A number between 0 and 1 for the ratio between the area of
                the sub faces and the area of this face.
            sub_rect_height: A number for the target height of the output sub-
                rectangles. Note that, if the ratio is too large for the height,
                the ratio will take precedence and the sub-rectangle height will
                be larger than this value.
            sill_height: A number for the target height above the bottom edge of
                the rectangle to start the sub-rectangles. Note that, if the
                ratio is too large for the height, the ratio will take precedence
                and the sub-rectangle height will be smaller than this value.
            horizontal_separation: A number for the target separation between
                individual sub-rectangle centerlines.  If this number is larger than
                the parent rectangle base, only one sub-rectangle will be produced.
            vertical_separation: An optional number to create a single vertical
                separation between top and bottom sub-rectangles. The default is
                0 for no separation.
            tolerance: The maximum difference between point values for them to be
                considered a part of a rectangle.

        Returns:
            A list of Face3D objects for sub faces. If there is a rectangle in this
            shape, the scaled rectangle will be the first item in this list.
        """
        rect_res = self.extract_rectangle(tolerance)
        if rect_res is None:
            return self.sub_faces_by_ratio(ratio)
        bottom_seg, top_seg, other_faces = rect_res
        height_seg = LineSegment3D.from_end_points(bottom_seg.p, top_seg.p)
        norm_tup = self._normal_from_3pts(bottom_seg.p, bottom_seg.p2, top_seg.p)
        norm = Vector3D(*norm_tup).normalize()
        base_plane = Plane(norm, bottom_seg.p, bottom_seg.v)
        sub_faces = Face3D.sub_rects_from_rect_ratio(
            base_plane, bottom_seg.length, height_seg.length, ratio,
            sub_rect_height, sill_height, horizontal_separation, vertical_separation)
        for face in other_faces:
            sfs = face.sub_faces_by_ratio(ratio)
            for sf in sfs:
                if sf.area > tolerance:
                    sub_faces.append(sf)
        return sub_faces

    def sub_faces_by_dimension_rectangle(self, sub_rect_height, sub_rect_width,
                                         sill_height, horizontal_separation, tolerance):
        """Get a list of rectangular faces within this Face3D.

        Note that this method will only yield results if there is a rectangle to
        be extracted from this Face3D's geometry.

        Args:
            sub_rect_height: A number for the target height of the output rectangles.
            sub_rect_width: A number for the target width of the output rectangles.
            sill_height: A number for the target height above the bottom edge of
                the rectangle to start the sub-rectangles. If the sub_rect_height
                is too large for the sill_height to fit within the rectangle,
                the sub_rect_height will take precedence.
            horizontal_separation: A number for the target separation between
                individual sub-rectangle centerlines.  If this number is larger than
                the parent rectangle base, only one sub-rectangle will be produced.
            tolerance: The maximum difference between point values for them to be
                considered a part of a rectangle.

        Returns:
            A list of Face3D objects for sub faces.
        """
        rect_res = self.extract_rectangle(tolerance)
        if rect_res is None:
            return []
        bottom_seg, top_seg, other_faces = rect_res
        height_seg = LineSegment3D.from_end_points(bottom_seg.p, top_seg.p)
        base_plane = Plane(self.normal, bottom_seg.p, bottom_seg.v)
        sub_faces = Face3D.sub_rects_from_rect_dimensions(
            base_plane, bottom_seg.length, height_seg.length, sub_rect_height,
            sub_rect_width, sill_height, horizontal_separation)
        return sub_faces

    def get_top_bottom_horizontal_edges(self, tolerance):
        """Get top and bottom horizontal edges of this Face if they exist.

        Args:
            tolerance: The maximum difference between the z values of the start and
                end coordinates at which an edge is considered horizontal.

        Returns:
            (bottom_edge, top_edge) with each as LineSegment3D if they exist.
            None if they do not exist.
        """
        # test if each of the edges are vertical.
        horizontal_edges = []
        for edge in self.boundary_segments:
            if edge.is_horizontal(tolerance):
                horizontal_edges.append(edge)

        if len(horizontal_edges) < 2:
            return None
        else:
            sorted_edges = sorted(horizontal_edges, key=lambda edge: edge.p.z)
            return sorted_edges[0], sorted_edges[1]

    def get_left_right_vertical_edges(self, tolerance):
        """Get left and right vertical edges of this Face if they exist.

        Args:
            tolerance: The maximum difference between the x any y values of the start
                and end coordinates at which an edge is considered vertical.

        Returns:
            (left_edge, right_edge) with each as LineSegment3D if they exist. Left in
            this case is defined as the edge with the lower X coordinates.
            Result will be None if vertical edges do not exist.
        """
        # test if each of the edges are vertical.
        vertical_edges = []
        for edge in self.boundary_segments:
            if edge.is_vertical(tolerance):
                vertical_edges.append(edge)

        if len(vertical_edges) < 2:
            return None
        else:
            if abs(self.normal.x) != 1:
                sorted_edges = sorted(vertical_edges, key=lambda edge: edge.p.x)
            else:
                sorted_edges = sorted(vertical_edges, key=lambda edge: edge.p.y)
            return sorted_edges[0], sorted_edges[-1]

    def extract_rectangle(self, tolerance):
        """Extract top and bottom line segments of a rectangle within this Face.

        This method will only return geometry if:

        1)  There are no holes in the face.

        2)  The face is not parallel to the World XY plane.

        3)  There are two parallel edges to this face, which are either
            oriented horizontally or vertically.

        4)  There must be enough overlap between these edges for a rectangle
            to be drawn between them.

        If this Face does not satisfy this criteria, None will be returned.

        Args:
            tolerance: The maximum difference between point values for them to be
                considered a part of a rectangle.

        Returns:
            A tuple with three elements

            -   bottom_edge: A LineSegment3D representing the bottom of the rectangle.

            -   top_edge: A LineSegment3D representing the top of the rectangle.

            -   other_faces:
                A list of Face3D objects for the parts of this face not
                included in the rectangle. The length of this list will be between
                0 (if this face is already rectangular) and 2 (if there are non-
                rectangular geometries on either side of the rectangle.)
        """
        # perform checks on the face to see if a rectangle is extractable
        if self.has_holes:
            return None
        if abs(self.normal.x) <= tolerance and abs(self.normal.y) <= tolerance:
            # face lies within a horizontal plane; we cannot distinguish top and bottom
            return None
        clean_face = self.remove_colinear_vertices(tolerance)

        # try to extract a rectangle from horizontal curves
        horiz_result = clean_face.get_top_bottom_horizontal_edges(tolerance)
        if horiz_result is not None:
            bottom_seg, top_seg = horiz_result
            split_res = clean_face._split_with_rectangle(bottom_seg, top_seg, tolerance)
            if split_res is not None:
                return LineSegment3D.from_end_points(split_res[0][1], split_res[0][3]), \
                    LineSegment3D.from_end_points(split_res[0][0], split_res[0][2]), \
                    split_res[1]

        # try to extract a rectangle from vertical curves
        vert_result = clean_face.get_left_right_vertical_edges(tolerance)
        if vert_result is not None:
            left_seg, right_seg = vert_result
            split_res = clean_face._split_with_rectangle(left_seg, right_seg, tolerance)
            if split_res is not None:
                seg_1 = LineSegment3D.from_end_points(split_res[0][0], split_res[0][1])
                seg_2 = LineSegment3D.from_end_points(split_res[0][2], split_res[0][3])
                sorted_edges = sorted([seg_1, seg_2], key=lambda edge: edge.p.z)
                return sorted_edges[0], sorted_edges[1], split_res[1]
        return None

    @staticmethod
    def sub_rects_from_rect_ratio(
            base_plane, parent_base, parent_height, ratio, sub_rect_height, sill_height,
            horizontal_separation, vertical_separation=0):
        """Get a list of rectangular Face3D objects using an area ratio and parameters.

        All of the resulting Face3D objects lie within a parent rectangle defined
        by the parent_base, parent_height, and base_plane. The combined area of the
        resulting rectangles is equal to the area of the larger rectangle multiplied
        by the input ratio. This method is particularly useful for generating
        rectangular window surfaces.

        Args:
            base_plane: A Plane object in which the rectangle exists.
                The origin of this plane will be the lower left corner of the
                rectangle and the X and Y axes will form the sides.
            parent_base: A number indicating the length of the base of the
                parent rectangle.
            parent_height: A number indicating the length of the height of the
                parent rectangle.
            ratio: A number between 0 and 1 for the ratio between the area of
                the sub rectangle faces and the area of this face.
            sub_rect_height: A number for the target height of the output sub-
                rectangles. Note that, if the ratio is too large for the height,
                the ratio will take precedence and the sub-rectangle height will
                be larger than this value.
            sill_height: A number for the target height above the bottom edge of
                the rectangle to start the sub-rectangles. Note that, if the
                ratio is too large for the height, the ratio will take precedence
                and the sub-rectangle height will be smaller than this value.
            horizontal_separation: A number for the target separation between
                individual sub-rectangle center lines.  If this number is larger than
                the parent rectangle base, only one sub-rectangle will be produced.
            vertical_separation: An optional number to create a single vertical
                separation between top and bottom sub-rectangles. The default is
                0 for no separation.

        Returns:
            A list of Face3D objects for sub faces.
        """
        # calculate the target area to make the combined sub-rectangles
        target_area = parent_base * parent_height * ratio
        # find the maximum area for subdivision into smaller, taller sub-rectangles
        max_area_subdiv = parent_base * 0.98 * sub_rect_height
        # if sub_rect_height > parent_height, set it to just under parent_height
        max_subh = 0.98 * parent_height
        sub_rect_height = max_subh if sub_rect_height > max_subh else sub_rect_height
        # if sill_height is close to 0, set it to just above 0.
        min_sill = 0.01 * parent_height
        sill_height = min_sill if sill_height < min_sill else sill_height
        # properties used throughout the computation of sub-rectangles
        bottom_seg = LineSegment3D.from_sdl(base_plane.o, base_plane.x, parent_base)

        if target_area < max_area_subdiv:
            # divide up the rectangle into points on the bottom.
            if parent_base > (horizontal_separation / 2):
                num_div = round(parent_base / horizontal_separation, 0)
            else:
                num_div = 1
            btm_div_pts = bottom_seg.subdivide_evenly(num_div)
            btm_div_segs = tuple(LineSegment3D.from_end_points(pt, btm_div_pts[i + 1])
                                 for i, pt in enumerate(btm_div_pts[:-1]))
            # move the segments to the sill height
            max_sill_h = parent_height * 0.99 - sub_rect_height
            sill_vec = base_plane.y * sill_height if sill_height < max_sill_h \
                else base_plane.y * max_sill_h
            div_segs = tuple(seg.move(sill_vec) for seg in btm_div_segs)
            # scale the segments along their center points
            seg_width = div_segs[0].length
            subrect_width = (target_area / sub_rect_height) / num_div
            scale_fac = subrect_width / seg_width
            scaled_segs = [seg.scale(scale_fac, seg.midpoint) for seg in div_segs]
            # find the maximum acceptable area for splitting the glazing vertically.
            if vertical_separation != 0:
                max_split_vert = parent_height - sill_height - sub_rect_height \
                    - (0.02 * parent_height)
                if vertical_separation < 0 or max_split_vert < 0:
                    vertical_separation = 0
                elif vertical_separation > max_split_vert:
                    vertical_separation = max_split_vert
            # generate the vertices by 'extruding' along a window height vector.
            final_faces = []
            if vertical_separation != 0:
                sub_rect_height = sub_rect_height / 2
                h_vec = base_plane.y * sub_rect_height
                vert_move_vec = base_plane.y * (sub_rect_height + vertical_separation)
                vert_segs = [seg.move(vert_move_vec) for seg in scaled_segs]
                for seg in scaled_segs + vert_segs:
                    final_faces.append(Face3D(
                        (seg.p1, seg.p2, seg.p2 + h_vec, seg.p1 + h_vec), base_plane))
            else:
                h_vec = base_plane.y * sub_rect_height
                for seg in scaled_segs:
                    final_faces.append(Face3D(
                        (seg.p1, seg.p2, seg.p2 + h_vec, seg.p1 + h_vec), base_plane))
        else:
            # make a single sub-rectangle at an appropriate sill height
            max_sill_h = parent_height * 0.99 - (target_area / (parent_base * 0.98))
            sill_vec = base_plane.y * sill_height if sill_height < max_sill_h \
                else base_plane.y * max_sill_h
            seg_init = bottom_seg.move(sill_vec)
            seg = seg_init.scale(0.98, seg_init.midpoint)
            # find the maximum acceptable area for splitting the glazing vertically.
            if vertical_separation != 0:
                max_split_vert = parent_height - sill_height - \
                    (target_area / (parent_base * 0.98)) - (0.02 * parent_height)
                if vertical_separation < 0 or max_split_vert < 0:
                    vertical_separation = 0
                elif vertical_separation > max_split_vert:
                    vertical_separation = max_split_vert
            # generate the vertices by 'extruding' along a window height vector.
            if vertical_separation != 0:
                sub_rect_height = (target_area / (parent_base * 0.98)) / 2
                h_vec = base_plane.y * sub_rect_height
                vert_move_vec = base_plane.y * (sub_rect_height + vertical_separation)
                vert_seg = seg.move(vert_move_vec)
                final_faces = []
                for seg in [seg, vert_seg]:
                    final_faces.append(Face3D(
                        (seg.p1, seg.p2, seg.p2 + h_vec, seg.p1 + h_vec), base_plane))
            else:
                h_vec = base_plane.y * (target_area / (parent_base * 0.98))
                final_faces = [Face3D((seg.p1, seg.p2, seg.p2 + h_vec, seg.p1 + h_vec),
                                      base_plane)]
        return final_faces

    @staticmethod
    def sub_rects_from_rect_dimensions(
            base_plane, parent_base, parent_height, sub_rect_height, sub_rect_width,
            sill_height, horizontal_separation):
        """Get a list of rectangular Face3D objects from dimensions and parameters.

        All of the resulting Face3D objects lie within a parent rectangle defined
        by the parent_base, parent_height, and base_plane.

        Args:
            base_plane: A Plane object in which the rectangle exists.
                The origin of this plane will be the lower left corner of the
                rectangle and the X and Y axes will form the sides.
            parent_base: A number indicating the length of the base of the
                parent rectangle.
            parent_height: A number indicating the length of the height of the
                parent rectangle.
            sub_rect_height: A number for the target height of the output rectangles.
            sub_rect_width: A number for the target width of the output rectangles.
            sill_height: A number for the target height above the bottom edge of
                the rectangle to start the sub-rectangles. If the sub_rect_height
                is too large for the sill_height to fit within the rectangle,
                the sub_rect_height will take precedence.
            horizontal_separation: A number for the target separation between
                individual sub-rectangle center lines.  If this number is larger than
                the parent rectangle base, only one sub-rectangle will be produced.

        Returns:
            A list of Face3D objects for sub faces.
        """
        # if sub_rect_height > parent_height, set it to just under parent_height
        sub_rect_height = parent_height - 0.02 * parent_height if \
            sub_rect_height >= parent_height else sub_rect_height
        # if sill_height is close to 0, set it to just above 0
        sill_hgt = 0.01 * parent_height if sill_height < 0.01 * parent_height \
            else sill_height
        # adjust sill_hgt if sum of it and sub_rect_height > parent_height
        if sub_rect_height + sill_hgt >= parent_height:
            sill_hgt = parent_height - sub_rect_height - (parent_height * 0.01)

        # ensure that the horizontal_separation is always greater than sub_rect_width
        if sub_rect_width >= horizontal_separation:
            horizontal_separation = sub_rect_width * 1.02

        # determine if the parameters should yield multiple sub-windows or just one
        max_width_break_up = parent_base / 2
        num_div = round(parent_base / horizontal_separation) if \
            parent_base > horizontal_separation / 2 else 1
        # properties used throughout the computation of sub-rectangles
        sill_vec = base_plane.y * sill_hgt
        bottom_seg = LineSegment3D.from_sdl(base_plane.o, base_plane.x, parent_base)

        if sub_rect_width < max_width_break_up:
            # determine the number of times that the rectangle should be subdivided
            div_dist = parent_base / 2 if num_div == 1 else horizontal_separation
            if num_div * sub_rect_width + (num_div - 1) * \
                    (horizontal_separation - sub_rect_width) > parent_base:
                num_div = math.floor(parent_base / horizontal_separation)

            # Get a segment in the center of the bottom
            scale_fac = (div_dist * num_div) / parent_base
            rect_seg = bottom_seg.scale(scale_fac, bottom_seg.point_at(0.5))
            rect_seg = rect_seg.move(sill_vec)
            btm_div_pts = rect_seg.subdivide_evenly(num_div)
            if len(btm_div_pts) == num_div:
                btm_div_pts.append(rect_seg.p2)

            # divide up the rectangle into points on the bottom
            btm_div_segs = tuple(LineSegment3D.from_end_points(pt, btm_div_pts[i + 1])
                                 for i, pt in enumerate(btm_div_pts[:-1]))
            # scale the line segments along their center points
            line_cent_pt = tuple(line.point_at(0.5) for line in btm_div_segs)
            scale_factor = sub_rect_width / div_dist
            btm_div_segs = tuple(line.scale(scale_factor, mid_pt)
                                 for line, mid_pt in zip(btm_div_segs, line_cent_pt))
            # generate the vertices by 'extruding' along window height vector
            h_vec = base_plane.y * sub_rect_height
            final_faces = [Face3D((line.p2, line.p1, line.p1 + h_vec, line.p2 + h_vec),
                                  base_plane) for line in btm_div_segs]
        else:  # make a single sub-rectangle at an appropriate sill height
            if sub_rect_width >= parent_base:
                sub_rect_width = parent_base * 0.98
            scale_fac = sub_rect_width / parent_base
            rect_seg = bottom_seg.scale(scale_fac, bottom_seg.point_at(0.5))
            seg = rect_seg.move(sill_vec)
            # generate the vertices by 'extruding' along window height vector
            h_vec = base_plane.y * sub_rect_height
            final_faces = [Face3D((seg.p2, seg.p1, seg.p1 + h_vec, seg.p2 + h_vec),
                                  base_plane)]
        return final_faces

    @staticmethod
    def coplanar_union(face1, face2, tolerance, angle_tolerance):
        """Boolean Union two coplanar Face3D with one another.

        Args:
            face1: A Face3D for the first face that will be unioned with the second face.
            face2: A Face3D for the second face that will be unioned with the first face.
            tolerance: The minimum distance between points before they are considered
                distinct from one another.
            angle_tolerance: The max angle in radians that the plane normals can
                differ from one another in order for them to be considered coplanar.

        Returns:
            A single Face3D for the Union of the two input Face3D. When the faces
            are not coplanar or they do not overlap, None will be returned.
        """
        # test whether the faces are coplanar
        prim_pl = face1.plane
        if not prim_pl.is_coplanar_tolerance(face2.plane, tolerance, angle_tolerance):
            return None
        # test whether the two polygons have any overlap in 2D space
        f1_poly = face1.boundary_polygon2d
        f2_poly = Polygon2D(tuple(prim_pl.xyz_to_xy(pt) for pt in face2.boundary))
        if not Polygon2D.overlapping_bounding_rect(f1_poly, f2_poly, tolerance):
            return None
        if f1_poly.polygon_relationship(f2_poly, tolerance) == -1:
            return None
        # snap the polygons to one another to avoid tolerance issues
        try:
            f1_poly = f1_poly.remove_colinear_vertices(tolerance)
            f2_poly = f2_poly.remove_colinear_vertices(tolerance)
        except AssertionError:  # degenerate faces input
            return None
        s2_poly = f1_poly.snap_to_polygon(f2_poly, tolerance)
        # get BooleanPolygons of the two faces
        f1_polys = [(pb.BooleanPoint(pt.x, pt.y) for pt in f1_poly.vertices)]
        f2_polys = [(pb.BooleanPoint(pt.x, pt.y) for pt in s2_poly.vertices)]
        if face1.has_holes:
            for hole in face1.hole_polygon2d:
                f1_polys.append((pb.BooleanPoint(pt.x, pt.y) for pt in hole.vertices))
        if face2.has_holes:
            for hole in face2.holes:
                h_pt2d = (prim_pl.xyz_to_xy(pt) for pt in hole)
                f2_polys.append((pb.BooleanPoint(pt.x, pt.y) for pt in h_pt2d))
        b_poly1 = pb.BooleanPolygon(f1_polys)
        b_poly2 = pb.BooleanPolygon(f2_polys)
        # split the two boolean polygons with one another
        int_tol = tolerance / 100
        try:
            poly_result = pb.union(b_poly1, b_poly2, int_tol)
        except Exception:
            return [face1], [face2]  # typically a tolerance issue causing failure
        # rebuild the Face3D from the results and return them
        union_faces = Face3D._from_bool_poly(poly_result, prim_pl)
        return union_faces[0]

    @staticmethod
    def coplanar_split(face1, face2, tolerance, angle_tolerance):
        """Split two coplanar Face3D with one another (ensuring matching overlapped area)

        When the faces are not coplanar or they do not overlap, the original
        faces will be returned.

        Args:
            face1: A Face3D for the first face that will be split with the second face.
            face2: A Face3D for the second face that will be split with the first face.
            tolerance: The minimum distance between points before they are considered
                distinct from one another.
            angle_tolerance: The max angle in radians that the plane normals can
                differ from one another in order for them to be considered coplanar.

        Returns:
            A tuple with two elements

        -   face1_split: A list of Face3D for the split version of the input face1.

        -   face2_split: A list of Face3D for the split version of the input face2.
        """
        # test whether the faces are coplanar
        prim_pl = face1.plane
        if not prim_pl.is_coplanar_tolerance(face2.plane, tolerance, angle_tolerance):
            return [face1], [face2]
        # test whether the two polygons have any overlap in 2D space
        f1_poly = face1.boundary_polygon2d
        f2_poly = Polygon2D(tuple(prim_pl.xyz_to_xy(pt) for pt in face2.boundary))
        if not Polygon2D.overlapping_bounding_rect(f1_poly, f2_poly, tolerance):
            return [face1], [face2]
        if f1_poly.polygon_relationship(f2_poly, tolerance) == -1:
            return [face1], [face2]
        # snap the polygons to one another to avoid tolerance issues
        try:
            f1_poly = f1_poly.remove_colinear_vertices(tolerance)
            f2_poly = f2_poly.remove_colinear_vertices(tolerance)
        except AssertionError:  # degenerate faces input
            return [face1], [face2]
        s2_poly = f1_poly.snap_to_polygon(f2_poly, tolerance)
        # get BooleanPolygons of the two faces
        f1_polys = [(pb.BooleanPoint(pt.x, pt.y) for pt in f1_poly.vertices)]
        f2_polys = [(pb.BooleanPoint(pt.x, pt.y) for pt in s2_poly.vertices)]
        if face1.has_holes:
            for hole in face1.hole_polygon2d:
                f1_polys.append((pb.BooleanPoint(pt.x, pt.y) for pt in hole.vertices))
        if face2.has_holes:
            for hole in face2.holes:
                h_pt2d = (prim_pl.xyz_to_xy(pt) for pt in hole)
                f2_polys.append((pb.BooleanPoint(pt.x, pt.y) for pt in h_pt2d))
        b_poly1 = pb.BooleanPolygon(f1_polys)
        b_poly2 = pb.BooleanPolygon(f2_polys)
        # split the two boolean polygons with one another
        int_tol = tolerance / 100
        try:
            int_result, poly1_result, poly2_result = pb.split(b_poly1, b_poly2, int_tol)
        except Exception:
            return [face1], [face2]  # typically a tolerance issue causing failure
        # rebuild the Face3D from the results and return them
        int_faces = Face3D._from_bool_poly(int_result, prim_pl)
        poly1_faces = Face3D._from_bool_poly(poly1_result, prim_pl)
        poly2_faces = Face3D._from_bool_poly(poly2_result, prim_pl)
        face1_split = poly1_faces + int_faces
        face2_split = poly2_faces + int_faces
        return face1_split, face2_split

    @staticmethod
    def _from_bool_poly(bool_polygon, plane):
        """Get a list of Face3D from a BooleanPolygon.

        This method will automatically check whether any of the regions is meant
        to be a hole within the others when it creates the Face3D.

        Args:
            bool_polygon: A BooleanPolygon to be interpreted to Face3D.
            plane: The Plane in which the resulting Face3Ds exist.
        """
        # serialize the BooleanPolygon into Polygon2D
        polys = [Polygon2D(tuple(Point2D(pt.x, pt.y) for pt in new_poly))
                 for new_poly in bool_polygon.regions if len(new_poly) > 2]
        if len(polys) == 0:
            return []
        if len(polys) == 1:
            verts_3d = tuple(plane.xy_to_xyz(pt) for pt in polys[0].vertices)
            return [Face3D(verts_3d, plane)]
        # sort the polygons by area and check if any are inside the others
        polys.sort(key=lambda x: x.area, reverse=True)
        poly_groups = [[polys[0]]]
        for sub_poly in polys[1:]:
            for i, pg in enumerate(poly_groups):
                if pg[0].is_polygon_inside(sub_poly):  # it's a hole
                    poly_groups[i].append(sub_poly)
                    break
            else:  # it's a separate Face3D
                poly_groups.append([sub_poly])
        # convert all vertices to 3D and return the Face3D
        face_3d = []
        for pg in poly_groups:
            pg_3d = []
            for shp in pg:
                pg_3d.append(tuple(plane.xy_to_xyz(pt) for pt in shp.vertices))
            face_3d.append(Face3D(pg_3d[0], plane, holes=pg_3d[1:]))
        return face_3d

    @staticmethod
    def group_by_coplanar_overlap(faces, tolerance):
        """Group coplanar Face3Ds depending on whether they overlap one another.

        This is useful as a pre-step before running Face3D.coplanar_union()
        in order to assess whether unionizing is necessary and to ensure that
        it is only performed among the necessary groups of faces.

        Args:
            faces: A list of Face3D to be grouped by their overlapping.
            tolerance: The minimum distance from the edge of a neighboring Face3D
                at which a point is considered to overlap with that Face3D.

        Returns:
            A list of lists where each sub-list represents a group of Face3Ds
            that all overlap with one another.
        """
        # create polygons for all of the faces
        r_plane = faces[0].plane
        polygons = [Polygon2D([r_plane.xyz_to_xy(pt) for pt in face.vertices])
                    for face in faces]
        # loop through the polygons and check to see if it overlaps with the others
        grouped_polys, grouped_faces = [[polygons[0]]], [[faces[0]]]
        for poly, face in zip(polygons[1:], faces[1:]):
            group_found = False
            for poly_group, face_group in zip(grouped_polys, grouped_faces):
                for oth_poly in poly_group:
                    if poly.polygon_relationship(oth_poly, tolerance) >= 0:
                        poly_group.append(poly)
                        face_group.append(face)
                        group_found = True
                        break
                if group_found:
                    break
            if not group_found:  # the polygon does not overlap with any of the others
                grouped_polys.append([poly])  # make a new group for the polygon
                grouped_faces.append([face])  # make a new group for the face
        return grouped_faces

    @staticmethod
    def join_coplanar_faces(faces, tolerance):
        """Join a list of coplanar Face3Ds together to get as few as possible.

        Args:
            faces: A list of Face3D objects to be joined together. These should
                all be coplanar but they do not need to have their colinear
                vertices removed or be intersected for matching segments along
                which they are joined.
            tolerance: The maximum difference between values at which point vertices
                are considered to be the same.

        Returns:
            A list of Face3Ds for the minimum number joined together.
        """
        # get polygons for the faces that all lie within the same plane
        face_polys, base_plane = [], faces[0].plane
        for fg in faces:
            verts2d = tuple(base_plane.xyz_to_xy(_v) for _v in fg.boundary)
            face_polys.append(Polygon2D(verts2d))
            if fg.has_holes:
                for hole in fg.holes:
                    verts2d = tuple(base_plane.xyz_to_xy(_v) for _v in hole)
                    face_polys.append(Polygon2D(verts2d))

        # remove colinear vertices
        clean_face_polys = []
        for geo in face_polys:
            try:
                clean_face_polys.append(geo.remove_colinear_vertices(tolerance))
            except AssertionError:  # degenerate geometry to ignore
                pass

        # get the joined boundaries around the Polygon2D
        joined_bounds = Polygon2D.joined_intersected_boundary(
            clean_face_polys, tolerance)

        # convert the boundary polygons back to Face3D
        if len(joined_bounds) == 1:  # can be represented with a single Face3D
            verts3d = tuple(base_plane.xy_to_xyz(_v) for _v in joined_bounds[0])
            return [Face3D(verts3d, plane=base_plane)]
        else:  # need to separate holes from distinct Face3Ds
            bound_faces = []
            for poly in joined_bounds:
                verts3d = tuple(base_plane.xy_to_xyz(_v) for _v in poly)
                bound_faces.append(Face3D(verts3d, plane=base_plane))
            return Face3D.merge_faces_to_holes(bound_faces, tolerance)

    @staticmethod
    def merge_faces_to_holes(faces, tolerance):
        """Take of list of Face3Ds and merge any sub-faces into the others as holes.

        This is particularly useful when translating 2D Polygons back into a 3D
        space and it is unknown whether certain polygons represent holes in the
        others.

        Args:
            faces: A list of Face3D which will be merged into fewer faces with
                any sub-faces represented as holes.
            tolerance: The tolerance to be used for evaluating sub-faces.
        """
        # sort the faces by area and separate base face from the remaining
        faces = sorted(faces, key=lambda x: x.area, reverse=True)
        base_face = faces[0]
        remain_faces = list(faces[1:])

        # merge the smaller faces into the larger faces
        merged_face3ds = []
        while len(remain_faces) > 0:
            merged_face3ds.append(
                Face3D._match_holes_to_face(base_face, remain_faces, tolerance))
            if len(remain_faces) > 1:
                base_face = remain_faces[0]
                del remain_faces[0]
            elif len(remain_faces) == 1:  # lone last Face3D
                merged_face3ds.append(remain_faces[0])
                del remain_faces[0]
        return merged_face3ds

    @staticmethod
    def _match_holes_to_face(base_face, other_faces, tol):
        """Attempt to merge other faces into a base face as holes.

        Args:
            base_face: A Face3D to serve as the base.
            other_faces: A list of other Face3D objects to attempt to merge into
                the base_face as a hole. This method will delete any faces
                that are successfully merged into the output from this list.
            tol: The tolerance to be used for evaluating sub-faces.

        Returns:
            A Face3D which has holes in it if any of the other_faces is a valid
            sub face.
        """
        holes = []
        more_to_check = True
        while more_to_check:
            for i, r_face in enumerate(other_faces):
                if base_face.is_sub_face(r_face, tol, 1):
                    holes.append(r_face)
                    del other_faces[i]
                    break
            else:
                more_to_check = False
        if len(holes) == 0:
            return base_face
        else:
            hole_verts = [hole.vertices for hole in holes]
            return Face3D(base_face.vertices, Plane(n=Vector3D(0, 0, 1)), hole_verts)

    def to_dict(self, include_plane=True, enforce_upper_left=False):
        """Get Face3D as a dictionary.

        Args:
            include_plane: Set to True to include the Face3D plane in the
                dictionary, which will preserve the underlying orientation
                of the face plane. Default True.
            enforce_upper_left: Set to True to ensure that the boundary vertices all
                start from the upper-left corner. This takes extra time to compute but
                ensures that the vertices in the dictionary are directly usable in an
                EnergyPlus simulations. Default: False.
        """
        base = {'type': 'Face3D'}
        if not enforce_upper_left:
            base['boundary'] = [pt.to_array() for pt in self.boundary]
        else:
            base['boundary'] = [pt.to_array() for pt in
                                self._upper_left_counter_clockwise_boundary()]
        if include_plane:
            base['plane'] = self.plane.to_dict()
        if self.has_holes:
            base['holes'] = [[pt.to_array() for pt in hole]
                             for hole in self.holes]
        return base

    @staticmethod
    def extract_all_from_stl(file_path):
        """Get a list of Face3Ds imported from all of the triangles in an STL file.

        Args:
            file_path: Path to an STL file as a text string. The STL file can be
                in either ASCII or binary format.
        """
        from ladybug_geometry.interop.stl import STL  # avoid circular import
        stl_obj = STL.from_file(file_path)
        all_faces = []
        for verts, normal in zip(stl_obj.face_vertices, stl_obj.face_normals):
            all_faces.append(Face3D(verts, plane=Plane(normal, verts[0])))
        return all_faces

    def _check_vertices_input(self, vertices, loop_name='boundary'):
        if not isinstance(vertices, tuple):
            vertices = tuple(vertices)
        assert len(vertices) >= 3, 'There must be at least 3 vertices for a Face3D {}.' \
            ' Got {}'.format(loop_name, len(vertices))
        for vert in vertices:
            assert isinstance(vert, Point3D), \
                'Expected Point3D for Face3D {} vertex. Got {}.'.format(
                    loop_name, type(vert))
        return vertices

    def _check_number_mesh_grid(self, input, name):
        assert isinstance(input, (float, int)), '{} for Face3D.get_mesh_grid' \
            ' must be a number. Got {}.'.format(name, type(input))

    def _move(self, vertices, mov_vec):
        return tuple(pt.move(mov_vec) for pt in vertices)

    def _rotate(self, vertices, axis, angle, origin):
        return tuple(pt.rotate(axis, angle, origin) for pt in vertices)

    def _rotate_xy(self, vertices, angle, origin):
        return tuple(pt.rotate_xy(angle, origin) for pt in vertices)

    def _reflect(self, vertices, normal, origin):
        return tuple(pt.reflect(normal, origin) for pt in reversed(vertices))

    def _scale(self, vertices, factor, origin):
        if origin is None:
            return tuple(
                Point3D(pt.x * factor, pt.y * factor, pt.z * factor)
                for pt in vertices)
        else:
            return tuple(pt.scale(factor, origin) for pt in vertices)

    def _face_transform(self, verts, plane):
        """Transform face in a way that transfers properties and avoids checks."""
        _new_face = Face3D(verts, plane, enforce_right_hand=False)
        self._transfer_properties(_new_face)
        _new_face._polygon2d = self._polygon2d
        _new_face._mesh2d = self._mesh2d
        return _new_face

    def _face_transform_reflect(self, verts, plane):
        """Reflect face in a way that transfers properties and avoids checks."""
        _new_face = Face3D(verts, plane, enforce_right_hand=False)
        self._transfer_properties(_new_face)
        return _new_face

    def _face_transform_scale(self, verts, plane, factor):
        """Scale face in a way that transfers properties and avoids checks."""
        _new_face = Face3D(verts, plane, enforce_right_hand=False)
        self._transfer_properties_scale(_new_face, factor)
        return _new_face

    def _transfer_properties(self, new_face):
        """Transfer properties from this face to a new face.

        This is used by the transform methods that don't alter the relationship of
        face vertices to one another (move, rotate, reflect).
        """
        new_face._perimeter = self._perimeter
        new_face._area = self._area
        new_face._is_convex = self._is_convex
        new_face._is_self_intersecting = self._is_self_intersecting

    def _transfer_properties_scale(self, new_face, factor):
        """Transfer properties from this face to a new face.

        This is used by the methods that scale the face.
        """
        new_face._is_convex = self._is_convex
        new_face._is_self_intersecting = self._is_self_intersecting
        if self._perimeter is not None:
            new_face._perimeter = self._perimeter * factor
        if self._area is not None:
            new_face._area = self._area * factor ** 2

    def _calculate_min_max(self):
        """Calculate maximum and minimum Point3D for this object."""
        min_pt = [self.boundary[0].x, self.boundary[0].y, self.boundary[0].z]
        max_pt = [self.boundary[0].x, self.boundary[0].y, self.boundary[0].z]

        for v in self.boundary[1:]:
            if v.x < min_pt[0]:
                min_pt[0] = v.x
            elif v.x > max_pt[0]:
                max_pt[0] = v.x
            if v.y < min_pt[1]:
                min_pt[1] = v.y
            elif v.y > max_pt[1]:
                max_pt[1] = v.y
            if v.z < min_pt[2]:
                min_pt[2] = v.z
            elif v.z > max_pt[2]:
                max_pt[2] = v.z

        self._min = Point3D(min_pt[0], min_pt[1], min_pt[2])
        self._max = Point3D(max_pt[0], max_pt[1], max_pt[2])

    def _remove_colinear(self, pts_3d, pts_2d, tolerance):
        """Remove colinear vertices from a list of Point2D.

        This method determines co-linearity by checking whether the area of the
        triangle formed by 3 vertices is less than the tolerance.
        """
        new_vertices = []  # list to hold the new vertices
        skip = 0  # track the number of vertices being skipped/removed
        # loop through vertices and remove all cases of colinear verts
        for i, _v in enumerate(pts_2d):
            _a = pts_2d[i - 2 - skip].determinant(pts_2d[i - 1]) + \
                pts_2d[i - 1].determinant(_v) + _v.determinant(pts_2d[i - 2 - skip])
            if abs(_a) >= tolerance:  # vertex is not colinear
                new_vertices.append(pts_3d[i - 1])
                skip = 0
            else:  # vertex is colinear
                skip += 1
        # catch case of last two vertices being equal but distinct from first point
        if skip != 0 and pts_3d[-2].is_equivalent(pts_3d[-1], tolerance):
            _a = pts_2d[-3].determinant(pts_2d[-1]) + \
                pts_2d[-1].determinant(pts_2d[0]) + pts_2d[0].determinant(pts_2d[-3])
            if abs(_a) >= tolerance:
                new_vertices.append(pts_3d[-1])
        return new_vertices

    def _is_sub_face(self, face):
        """Check if a face is a sub-face of this face, bypassing coplanar check.

        Args:
            face: Another face for which sub-face equivalency will be tested.
        """
        verts2d = tuple(self.plane.xyz_to_xy(_v) for _v in face.vertices)
        sub_poly = Polygon2D(verts2d)

        if not self.has_holes:
            return self.polygon2d.is_polygon_inside(sub_poly)
        else:
            if not self.boundary_polygon2d.is_polygon_inside(sub_poly):
                return False
            for hole_poly in self.hole_polygon2d:
                if not hole_poly.is_polygon_outside(sub_poly):
                    return False
            return True

    def _vertices_between_points(self, start_pt, end_pt, tolerance):
        """Get the vertices between a start and end point.

        This method is used by the extract_rectangle method.
        """
        new_verts = [start_pt]
        vert_ind = self.vertices.index(start_pt)
        found_other = False
        while found_other is False:
            vert_ind -= 1
            new_verts.append(self[vert_ind])
            if self[vert_ind].is_equivalent(end_pt, tolerance):
                found_other = True
        return new_verts

    def _diagonal_along_self(self, direction_vector, tolerance):
        """Get the diagonal oriented along this face and always starts on the left."""
        tol_pt = Vector3D(1.0e-7, 1.0e-7, 1.0e-7)  # closer than float tolerance
        diagonal = LineSegment3D.from_end_points(self.min + tol_pt, self.max - tol_pt)
        # invert the diagonal XY if it is not oriented with the face plane
        if self._plane.distance_to_point(diagonal.p) > tolerance:
            start = Point3D(diagonal.p1.x, diagonal.p2.y, diagonal.p1.z)
            end = Point3D(diagonal.p2.x, diagonal.p1.y, diagonal.p2.z)
            diagonal = LineSegment3D.from_end_points(start, end)
        # flip if there's a horizontal direction_vector to ensure always starts on left
        if direction_vector.x != 0 and self.normal.y > 0:
            diagonal = diagonal.flip()
        return diagonal

    def _get_fin_extrusion_vector(self, depth, angle, contour_vector):
        """Get the vector with which to extrude fins."""
        extru_vec = self.plane.n * depth
        if angle != 0:
            # interpret the complement of the 2D contour_vector into a 3D axis
            cont_vec_complement = Vector2D(contour_vector.y, -contour_vector.x)
            ref_plane = Plane(self._plane.n, Point3D(0, 0, 0), self._plane.x)
            if ref_plane.y.z < 0:
                ref_plane = ref_plane.rotate(ref_plane.n, math.pi, ref_plane.o)
            axis = ref_plane.xy_to_xyz(cont_vec_complement).normalize()
            # rotate the extrusion vector around the axis
            extru_vec = extru_vec.rotate(axis, angle)
        return extru_vec

    def _get_extrusion_fins(self, contours, extru_vec, offset):
        """Get fins from the contours and extrusion vector."""
        if offset != 0:
            off_vec = self.plane.n * offset
            contours = tuple(seg.move(off_vec) for seg in contours)
        return tuple(Face3D.from_extrusion(seg, extru_vec) for seg in contours)

    def _split_with_rectangle(self, edge_1, edge_2, tolerance):
        """Split this shape using two parallel edges of the face.

        Result will be None if no rectangle can be obtained.

        Returns:
            rectangle_points: A tuple of 4 points that make the rectangle.
            other_faces: A list of faces for the other parts of this Face that
                are not a part of the rectangle.
        """
        # compute the 4 points defining the rectangle
        close_pt_1 = closest_point3d_on_line3d(edge_1.p1, edge_2)
        close_pt_2 = closest_point3d_on_line3d(edge_2.p2, edge_1)
        close_pt_3 = closest_point3d_on_line3d(edge_1.p2, edge_2)
        close_pt_4 = closest_point3d_on_line3d(edge_2.p1, edge_1)

        # check that there is overlap between the top and bottom curves
        if close_pt_1.is_equivalent(edge_2.p1, tolerance) or \
                close_pt_3.is_equivalent(edge_2.p2, tolerance):
            return None

        # check that the two sides of the rectangle are inside the polygon.
        mid_pt_1 = self.plane.xyz_to_xy(
            LineSegment3D.from_end_points(close_pt_1, close_pt_2).midpoint)
        mid_pt_2 = self.plane.xyz_to_xy(
            LineSegment3D.from_end_points(close_pt_3, close_pt_4).midpoint)
        if self.polygon2d.point_relationship(mid_pt_1, tolerance) == -1 or \
                self.polygon2d.point_relationship(mid_pt_2, tolerance) == -1:
            return None

        # get extra faces outside of the rectangle
        other_faces = []
        edge_pts_1 = self._vertices_between_points(edge_1.p1, edge_2.p2, tolerance)
        if close_pt_1.is_equivalent(edge_2.p2, tolerance) is False:
            edge_pts_1.append(close_pt_1)
            other_faces.append(Face3D(edge_pts_1, self.plane))
        elif close_pt_2.is_equivalent(edge_1.p1, tolerance) is False:
            edge_pts_1.append(close_pt_2)
            other_faces.append(Face3D(edge_pts_1, self.plane))
        elif len(edge_pts_1) > 2:
            other_faces.append(Face3D(edge_pts_1, self.plane))

        edge_pts_2 = self._vertices_between_points(edge_2.p1, edge_1.p2, tolerance)
        if close_pt_3.is_equivalent(edge_2.p1, tolerance) is False:
            edge_pts_2.append(close_pt_3)
            other_faces.append(Face3D(edge_pts_2, self.plane))
        elif close_pt_4.is_equivalent(edge_1.p2, tolerance) is False:
            edge_pts_2.append(close_pt_4)
            other_faces.append(Face3D(edge_pts_2, self.plane))
        elif len(edge_pts_2) > 2:
            other_faces.append(Face3D(edge_pts_2, self.plane))

        # check that any new faces are not self intersecting
        for new_face in other_faces:
            if new_face.is_self_intersecting:
                return None

        # return the rectangle edges and the extra faces
        return (close_pt_1, close_pt_2, close_pt_3, close_pt_4), other_faces

    def _point_on_face(self, tolerance):
        """Get a point that is always reliably on this face.

        The point will be close to the edge of the Face but it will always
        be inside its boundary for all concave and holed geometries. Furthermore,
        it is relatively fast compared with computing the pole_of_inaccessibility.
        """
        try:
            face = self.remove_colinear_vertices(tolerance)
            move_vec = self._inward_pointing_vec(face)
        except (AssertionError, ZeroDivisionError):  # zero area Face3D; use center
            return self.center

        move_vec = move_vec * (tolerance + 0.00001)
        point_on_face = face.boundary[0] + move_vec
        vert2d = face.plane.xyz_to_xy(point_on_face)
        if not face.polygon2d.is_point_inside(vert2d):
            point_on_face = face.boundary[0] - move_vec
        return point_on_face

    def _upper_oriented_plane(self):
        """Get a version of this Face3D's plane where Y is oriented towards positive Z.

        If the Face3D is horizontal, the plane will be the World XY.
        """
        if self._plane.n.z == 1 or self._plane.n.z == -1:  # no vertex is above another
            ref_plane = Plane(self._plane.n, self._plane.o, Vector3D(1, 0, 0))
        else:
            proj_y = Vector3D(0, 0, 1).project(self._plane.n)
            proj_x = proj_y.rotate(self._plane.n, math.pi / -2)
            ref_plane = Plane(self._plane.n, self._plane.o, proj_x)
        return ref_plane

    def _corner_point(self, x_corner='min', y_corner='min'):
        """Get a Point3D that is in a particular corner of this Face3D.

        Args:
            x_corner: Either "min" or "max" depending on the desired corner.
            y_corner: Either "min" or "max" depending on the desired corner.
        """
        # get a correctly-oriented polygon
        ref_plane = self._upper_oriented_plane()
        polygon = Polygon2D(tuple(ref_plane.xyz_to_xy(v) for v in self._boundary))
        # sort points so that they start with the correct corner
        x_pt = getattr(polygon, x_corner)
        y_pt = getattr(polygon, y_corner)
        return ref_plane.xy_to_xyz(Point2D(x_pt.x, y_pt.y))

    def _corner_point_and_polygon(self, points_3d, x_corner='min', y_corner='min'):
        """Get a Point2D and corresponding Polygon in a particular corner of this Face3D.

        Args:
            points_3d: A list of Point3Ds for the output Polygon.
            x_corner: Either "min" or "max" depending on the desired corner.
            y_corner: Either "min" or "max" depending on the desired corner.
        """
        if self.is_horizontal(0.01):  # EnergyPlus tolerance
            polygon = Polygon2D(tuple(Point2D(v.x, v.y) for v in points_3d))
            if self._plane.n.z < 0:  # flip the direction of what counts as "right"
                x_corner = 'max' if x_corner == 'min' else 'min'
            x_pt = getattr(self, x_corner)
            y_pt = getattr(self, y_corner)
        else:
            # get a 2d polygon in the face plane that has a positive Y axis.
            proj_y = Vector3D(0, 0, 1).project(self._plane.n)
            proj_x = proj_y.rotate(self._plane.n, math.pi / -2)
            ref_plane = Plane(self._plane.n, self._plane.o, proj_x)
            polygon = Polygon2D(tuple(ref_plane.xyz_to_xy(v) for v in points_3d))
            x_pt = getattr(polygon, x_corner)
            y_pt = getattr(polygon, y_corner)
        return Point2D(x_pt.x, y_pt.y), polygon

    def _counter_clockwise_verts(self, polygon):
        """Get aligned lists of counter-clockwise 2D and 3D vertices."""
        if self.is_clockwise:
            return tuple(reversed(self.vertices)), tuple(reversed(polygon.vertices))
        else:
            return self.vertices, polygon.vertices

    def _upper_left_counter_clockwise_boundary(self):
        """Get this face's boundary starting from upper left and moving counterclockwise.

        Horizontal faces will treat the positive Y axis as up. All other faces
        treat the positive Z axis as up.

        Unlike the upper_left_counter_clockwise_vertices property, this property
        does not include any holes in the Face3D.
        """
        corner_pt, polygon = self._corner_point_and_polygon(self._boundary, 'min', 'max')
        if self.is_clockwise:
            verts3d, verts2d = \
                tuple(reversed(self.boundary)), tuple(reversed(polygon.vertices))
        else:
            verts3d, verts2d = self.boundary, polygon.vertices
        return self._corner_pt_verts(corner_pt, verts3d, verts2d)

    @staticmethod
    def _inward_pointing_vec(face):
        """Get a unit vector pointing inward/outward from the first vertex of a face."""
        v1 = face.boundary[-1] - face.boundary[0]
        v2 = face.boundary[1] - face.boundary[0]
        if v1.angle(v2) == math.pi:  # colinear vertices; prevent averaging to zero
            return v1.rotate(face.normal, math.pi / 2).normalize()
        else:  # average the two edge vectors together
            avg_coords = ((v1.x + v2.x) / 2), ((v1.y + v2.y) / 2), ((v1.z + v2.z) / 2)
            return Vector3D(*avg_coords).normalize()

    @staticmethod
    def _plane_from_vertices(verts):
        """Get a plane from a list of vertices.

        Args:
            verts: The vertices to be used to extract the normal.
        """
        try:
            # walk around the shape and get cross products
            cprods, base_vert = [], verts[0]
            for i in range(len(verts) - 2):
                verts_3 = (base_vert, verts[i + 1], verts[i + 2])
                cprods.append(Face3D._normal_from_3pts(*verts_3))
            # sum together the cross products
            normal = [0, 0, 0]
            for cprodx in cprods:
                normal[0] += cprodx[0]
                normal[1] += cprodx[1]
                normal[2] += cprodx[2]
            # normalize the vector
            if normal != [0, 0, 0]:
                ds = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
                normal_vec = Vector3D(normal[0] / ds, normal[1] / ds, normal[2] / ds)
            else:  # zero area Face3D; default to positive Z axis
                normal_vec = Vector3D(0, 0, 1)
        except Exception as e:
            raise ValueError('Incorrect vertices input for Face3D:\n\t{}'.format(e))
        return Plane(normal_vec, verts[0])

    @staticmethod
    def _normal_from_3pts(pt1, pt2, pt3):
        """Get a tuple representing a normal vector from 3 vertices.

        The vector will have a magnitude of 0 if vertices are colinear.
        This method effectively performs the cross product of two unit vectors but
        the ladybug_geometry objects are not used in order to remove assertions
        and increase speed.
        """
        # get two vectors for the two edges the 3 points form
        v1 = (pt2.x - pt1.x, pt2.y - pt1.y, pt2.z - pt1.z)
        v2 = (pt3.x - pt1.x, pt3.y - pt1.y, pt3.z - pt1.z)
        # get the cross product of the two edge vectors
        return (v1[1] * v2[2] - v1[2] * v2[1],
                -v1[0] * v2[2] + v1[2] * v2[0],
                v1[0] * v2[1] - v1[1] * v2[0])

    @staticmethod
    def _corner_pt_verts(corner_pt, verts3d, verts2d):
        """Get verts3d starting from the one closes to the corner_pt."""
        first_pt_index = 0
        min_dist = verts2d[0].distance_to_point(corner_pt)
        for pt_index, pt in enumerate(verts2d[1:]):
            new_dist = pt.distance_to_point(corner_pt)
            if new_dist < min_dist:
                first_pt_index = pt_index + 1
                min_dist = new_dist
        if first_pt_index != 0:
            verts3d = verts3d[first_pt_index:] + verts3d[:first_pt_index]
        return verts3d

    def __copy__(self):
        _new_face = Face3D(self.vertices, self.plane)
        self._transfer_properties(_new_face)
        _new_face._holes = self._holes
        _new_face._polygon2d = self._polygon2d
        _new_face._mesh2d = self._mesh2d
        _new_face._mesh3d = self._mesh3d
        return _new_face

    def __key(self):
        """A tuple based on the object properties, useful for hashing."""
        return tuple(hash(pt) for pt in self._vertices) + (hash(self._plane),)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(other, Face3D) and self.__key() == other.__key()

    def __repr__(self):
        return 'Face3D ({} vertices)'.format(len(self))
