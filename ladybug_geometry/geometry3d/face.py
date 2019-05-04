# coding=utf-8
"""Planar Face in 3D Space"""
from __future__ import division

from .pointvector import Point3D, Point3DImmutable, Vector3D, Vector3DImmutable
from .line import LineSegment3D, LineSegment3DImmutable
from .plane import Plane
from .mesh import Mesh3D
from ._2d import Base2DIn3D

from ..geometry2d.pointvector import Vector2D
from ..geometry2d.ray import Ray2D
from ..geometry2d.polygon import Polygon2D
from ..geometry2d.mesh import Mesh2D

import math


class Face3D(Base2DIn3D):
    """Planar Face in 3D space.

    Properties:
        vertices
        plane
        polygon2d
        triangulated_mesh2d
        triangulated_mesh3d
        boundary
        holes
        boundary_segments
        hole_segments
        normal
        min
        max
        center
        perimeter
        area
        centroid
        is_clockwise
        is_convex
        is_self_intersecting
    """
    __slots__ = ('_vertices', '_plane', '_polygon2d', '_triangulated_mesh2d',
                 '_triangulated_mesh3d', '_boundary', '_holes',
                 '_boundary_segments', '_hole_segments'
                 '_min', '_max', '_center', '_perimeter', '_area', '_centroid',
                 '_is_clockwise', '_is_convex', '_is_complex')

    def __init__(self, vertices, plane):
        """Initilize Face3D.

        Args:
            vertices: A list of Point3D objects representing the vertices of the sface.
            plane: A Plane object indicating the plane in which the face exists.
        """
        self._check_vertices_input(vertices)
        assert isinstance(plane, Plane), 'Expected Plane for Face3D.' \
            ' Got {}.'.format(type(plane))
        self._plane = plane

        self._polygon2d = None
        self._mesh2d = None
        self._mesh3d = None
        self._boundary = self._vertices
        self._holes = None
        self._boundary_segments = None
        self._hole_segments = None

        self._min = None
        self._max = None
        self._center = None
        self._perimeter = None
        self._area = None
        self._centroid = None
        self._is_clockwise = None
        self._is_convex = None
        self._is_complex = None

    @classmethod
    def from_vertices(cls, vertices):
        """Initialize Face3D from only a list of vertices.

        The Plane normal will automatically be calculated by analyzing the first
        three vertices and the origin of the plane will be the first vertex of
        the input vertices.
        """
        plane = cls._plane_from_vertices(vertices)
        return cls(vertices, plane)

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
        assert isinstance(line_segment, (LineSegment3D, LineSegment3DImmutable)), \
            'line_segment must be LineSegment3D. Got {}.'.format(
                type(line_segment))
        assert isinstance(extrusion_vector, (Vector3D, Vector3DImmutable)), \
            'extrusion_vector must be Vector3D. Got {}.'.format(
                type(extrusion_vector))
        _p1 = line_segment.p1
        _p2 = line_segment.p2
        _verts = (_p1, _p1 + extrusion_vector, _p2 + extrusion_vector, _p2)
        _plane = Plane(line_segment.v.cross(extrusion_vector), _p1)
        face = cls(_verts, _plane)
        _base = line_segment.length
        _dist = extrusion_vector.magnitude
        _height = _dist * math.sin(extrusion_vector.angle(line_segment.v))
        face._perimeter = _base * 2 + _dist * 2
        face._area = _base * _height
        _cent = _p1 + (line_segment.v * 0.5) + (extrusion_vector * 0.5)
        face._centroid = _cent.to_immutable()
        face._is_clockwise = True
        face._is_convex = False
        face._is_complex = False
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
            base_plane = Plane(Vector3D(0, 0, 1), Point3D())
        _o = base_plane.o
        _b_vec = base_plane.x * base
        _h_vec = base_plane.y * height
        _verts = (_o, _o + _h_vec, _o + _h_vec + _b_vec, _o + _b_vec)
        face = cls(_verts, base_plane)
        face._perimeter = base * 2 + height * 2
        face._area = base * height
        _cent = _o + (_b_vec * 0.5) + (_h_vec * 0.5)
        face._centroid = _cent.to_immutable()
        face._is_clockwise = True
        face._is_convex = False
        face._is_complex = False
        return face

    @classmethod
    def from_shape_with_holes(cls, boundary, holes, plane=None):
        """Initialize a Face3D from a boundary vertex list with holes inside of it.

        This method will separately store the list of Point2Ds representing the
        boundary and holes on the `boundary` and `holes` properties of this object.
        However, under the hood, the polygon_2d for the face will be a single
        concave polygon made by drawing lines from the holes to the outer boundary.
        This allows for ease of use with the other methods including intersections,
        generating triangulated meshes, and generating grid meshes.

        Args:
            boundary: A list of Point3D objects for the outer boundary of the face
                inside of which all of the holes are contained.
            holes: A list of lists with one list for each hole in the face. Each hole
                should be a list of at least 3 Point3D objects.
            plane: A Plane object indicating the plane in which the face exists.
                If left as none, the Plane normal will automatically be calculated
                by analyzing the first three vertices of the boundary and the origin
                of the plane will be the first vertex of the boundary vertices.
        """
        # check the inputs
        assert isinstance(boundary, list), \
            'boundary should be a list. Got {}'.format(type(boundary))
        assert isinstance(holes, list), \
            'holes should be a list. Got {}'.format(type(holes))
        for hole in holes:
            assert isinstance(hole, list), \
                'hole should be a list. Got {}'.format(type(hole))
            assert len(hole) >= 3, \
                'hole should have at least 3 vertices. Got {}'.format(len(hole))
        if plane is None:
            plane = cls._plane_from_vertices(boundary)

        # create a Polygon2D from the vertices
        _boundary2d = [plane.xyz_to_xy_immutable(_v) for _v in boundary]
        _holes2d = [[plane.xyz_to_xy_immutable(_v) for _v in hole] for hole in holes]
        _polygon2d = Polygon2D.from_shape_with_holes(_boundary2d, _holes2d)

        # convert Polygon2D vertices to 3D to become the vertices of the face.
        _vert3d = tuple(plane.xy_to_xyz_immutable(_v) for _v in _polygon2d.vertices)
        _face = cls(_vert3d, plane)

        # assign extra properties that we know to the face
        _face._polygon2d = _polygon2d
        _face._boundary = tuple(pt.to_immutable() for pt in boundary)
        _face._holes = tuple(tuple(pt.to_immutable() for pt in hole) for hole in holes)
        return _face

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
            _vert2d = tuple(self._plane.xyz_to_xy_immutable(_v) for _v in self.vertices)
            self._polygon2d = Polygon2D(_vert2d)
            if self._is_clockwise is not None:
                self._polygon2d._is_clockwise = self._is_clockwise
        return self._polygon2d

    @property
    def triangulated_mesh2d(self):
        """A triagulated Mesh2D in the 2D space of the face's plane."""
        if self._mesh2d is None:
            self._mesh2d = Mesh2D.from_polygon_triangulated(self.polygon2d)
        return self._mesh2d

    @property
    def triangulated_mesh3d(self):
        """A triagulated Mesh3D of this face."""
        if self._mesh3d is None:
            _vert3d = [self._plane.xy_to_xyz_immutable(_v) for _v in
                       self.triangulated_mesh2d.vertices]
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
                _seg = LineSegment3DImmutable.from_end_points(self.boundary[i - 1], vert)
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
                    _seg = LineSegment3DImmutable.from_end_points(hole[i - 1], vert)
                    _segs.append(_seg)
                _segs.append(_segs.pop(0))  # segments will start from the first vertex
                _all_segs.append(_segs)
            self._hole_segments = tuple(tuple(_s) for _s in _all_segs)
        return self._hole_segments

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
            self._centroid = self._plane.xy_to_xyz_immutable(_cent2d)
        return self._centroid

    @property
    def is_clockwise(self):
        """Boolean for whether the face vertices and boundary are in clockwise order.

        Note that this does not describe the orientation of any holes in the face.
        """
        if self._is_clockwise is None:
            self._is_clockwise = self.polygon2d.is_clockwise
        return self._is_clockwise

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
        if self._is_complex is None:
            self._is_complex = self.polygon2d.is_self_intersecting
        return self._is_complex

    @property
    def has_holes(self):
        """Boolean noting whther the face has holes within it."""
        return self._holes is not None

    def validate_planarity(self, tolerance, raise_exception=True):
        """Validate that all of the face's vertices lie within the face's plane.

        This check is not done by default when creating the face since
        it is assumed that there is likely a check for planarity before the face
        is created (ie. in CAD software where the face likely originates from).
        This method is intended for quality control checking when the origin of
        face geometry is unkown or is known to come from a place where no
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
        """
        for _v in self.vertices:
            if self._plane.distance_to_point(_v) > tolerance:
                if raise_exception is True:
                    raise AttributeError(
                        'Vertex {} is out of plane with its parent face.\nDistance '
                        'to plane is {}'.format(_v, self._plane.distance_to_point(_v)))
                else:
                    return False
        return True

    def flip(self):
        """Get a face with a flipped direction from this one.

        Note that this only flips the plane of the face and does not change the vertices.
        """
        _new_face = Face3D(self.vertices, self.plane.flip())
        self._transfer_properties(_new_face)
        _new_face._boundary = self._boundary
        _new_face._holes = self._holes
        if self._is_clockwise is not None:
            _new_face._is_clockwise = not self._is_clockwise
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
            _new_face._holes = self._move(self._holes, moving_vec)
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
            _new_face._holes = self._rotate(self._holes, axis, angle, origin)
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
            _new_face._holes = self._rotate_xy(self._holes, angle, origin)
        return _new_face

    def reflect(self, normal, origin):
        """Get a face reflected across a plane with the input normal vector and origin.

        Args:
            normal: A Vector3D representing the normal vector for the plane across
                which the face will be reflected. THIS VECTOR MUST BE NORMALIZED.
            origin: A Point3D representing the origin from which to reflect.
        """
        _verts = self._reflect(self.vertices, normal, origin)
        _new_face = self._face_transform(_verts, self.plane.reflect(normal, origin))
        if self._holes is not None:
            _new_face._boundary = self._reflect(self._boundary, normal, origin)
            _new_face._holes = self._reflect(self._holes, normal, origin)
        return _new_face

    def scale(self, factor, origin):
        """Scale a face by a factor from an origin point.

        Args:
            factor: A number representing how much the face should be scaled.
            origin: A Point3D representing the origin from which to scale.
        """
        _verts = self._scale(self.vertices, factor, origin)
        _new_face = self._face_transform_scale(
            _verts, self.plane.scale(factor, origin), factor)
        if self._holes is not None:
            _new_face._boundary = self._scale(self._boundary, factor, origin)
            _new_face._holes = self._scale(self._holes, factor, origin)
        return _new_face

    def scale_world_origin(self, factor):
        """Scale a face by a factor from the world origin. Faster than Face3D.scale.

        Args:
            factor: A number representing how much the line segment should be scaled.
        """
        _verts = self._scale_world_origin(self.vertices, factor)
        _new_face = self._face_transform_scale(
            _verts, self.plane.scale_world_origin(factor), factor)
        if self._holes is not None:
            _new_face._boundary = self._scale_world_origin(self._boundary, factor)
            _new_face._holes = self._scale_world_origin(self._holes, factor)
        return _new_face

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
            _v2d = self._plane.xyz_to_xy(_plane_int_ray.v)
            _int_ray2d = Ray2D(self._plane.xyz_to_xy(_plane_int_ray.p),
                               Vector2D(_v2d.x, _v2d.y))
            _int_pt2d = self.polygon2d.intersect_line_infinite(_int_ray2d)
            if len(_int_pt2d) != 0:
                if len(_int_pt2d) > 2:  # sort the points along the intersection line
                    _int_pt2d.sort(key=lambda pt: pt.x)
                _int_pt3d = [self._plane.xy_to_xyz(pt) for pt in _int_pt2d]
                _int_seg3d = []
                for i in range(0, len(_int_pt3d) - 1, 2):
                    _int_seg3d.append(LineSegment3D(_int_pt3d[i], _int_pt3d[i + 1]))
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
        _plane_int = self._plane.project_point(point)
        _plane_int2d = self._plane.xyz_to_xy(_plane_int)
        if self.polygon2d.is_point_inside_bound_rect(_plane_int2d):
            return _plane_int
        return None

    def get_mesh_grid(self, x_dim, y_dim=None, offset=None, flip=False,
                      generate_centroids=True):
        """Get a gridded Mesh3D from over this face.

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
                have no need for the face centroids, you would save memory by setting
                this to False. Default is True.
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
        vert_3d = tuple(self._plane.xy_to_xyz_immutable(pt)
                        for pt in grid_mesh2d.vertices)
        if offset is not None:
            _off_num = -offset if flip is True else offset
            _off_vec = self.plane.n * _off_num
            vert_3d = tuple(pt.move(_off_vec).to_immutable() for pt in vert_3d)
        grid_mesh3d = Mesh3D(vert_3d, grid_mesh2d.faces)
        grid_mesh3d._face_areas = grid_mesh2d._face_areas

        # assign the face plane normal to the mesh normals
        if flip is True:
            grid_mesh3d._face_normals = self._plane.n.reversed().to_immutable()
            grid_mesh3d._vertex_normals = self._plane.n.reversed().to_immutable()
        else:
            grid_mesh3d._face_normals = grid_mesh3d._vertex_normals = self._plane.n

        # transform the centroids to 3D space if they were generated
        if generate_centroids is True:
            cent_3d = tuple(self._plane.xy_to_xyz_immutable(pt)
                            for pt in grid_mesh2d.face_centroids)
            if offset is not None:
                cent_3d = tuple(pt.move(_off_vec).to_immutable() for pt in cent_3d)
            grid_mesh3d._face_centroids = cent_3d

        return grid_mesh3d

    def _check_vertices_input(self, vertices):
        assert isinstance(vertices, (list, tuple)), \
            'vertices should be a list or tuple. Got {}'.format(type(vertices))
        assert len(vertices) >= 3, 'There must be at least 3 vertices for a Face3D.' \
            ' Got {}'.format(len(vertices))
        assert isinstance(vertices[0], (Point3D, Point3DImmutable)), \
            'Expected Point3D for Face3D vertex. Got {}.'.format(type(vertices[0]))
        self._vertices = tuple(p.to_immutable() for p in vertices)

    def _check_number_mesh_grid(self, input, name):
        assert isinstance(input, (float, int)), '{} for Face3D.get_mesh_grid' \
            ' must be a number. Got {}.'.format(name, type(input))

    def _move(self, vertices, mov_vec):
        return tuple(Point3DImmutable(
            pt.x + mov_vec.x, pt.y + mov_vec.y, pt.z + mov_vec.z) for pt in vertices)

    def _rotate(self, vertices, axis, angle, origin):
        return tuple([pt.rotate(axis, angle, origin).to_immutable() for pt in vertices])

    def _rotate_xy(self, vertices, angle, origin):
        return tuple([pt.rotate_xy(angle, origin).to_immutable() for pt in vertices])

    def _reflect(self, vertices, normal, origin):
        return tuple([pt.reflect(normal, origin).to_immutable() for pt in vertices])

    def _scale(self, vertices, factor, origin):
        return tuple([pt.scale(factor, origin).to_immutable() for pt in vertices])

    def _scale_world_origin(self, vertices, factor):
        return tuple([pt.scale_world_origin(factor).to_immutable() for pt in vertices])

    def _face_transform(self, verts, plane):
        """Transform face in a way that transfers properties and avoids checks."""
        _new_face = Face3D(verts, plane)
        self._transfer_properties(_new_face)
        _new_face._polygon2d = self._polygon2d
        _new_face._mesh2d = self._mesh2d
        return _new_face

    def _face_transform_scale(self, verts, plane, factor):
        """Scale face in a way that transfers properties and avoids checks."""
        _new_face = Face3D(verts, plane)
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
        new_face._is_complex = self._is_complex
        new_face._is_clockwise = self._is_clockwise

    def _transfer_properties_scale(self, new_face, factor):
        """Transfer properties from this face to a new face.

        This is used by the methods that scale the face.
        """
        new_face._is_convex = self._is_convex
        new_face._is_complex = self._is_complex
        new_face._is_clockwise = self._is_clockwise
        if self._perimeter is not None:
            new_face._perimeter = self._perimeter * factor
        if self._area is not None:
            new_face._area = self._area * factor

    @staticmethod
    def _plane_from_vertices(vertices):
        """Get a plane from a list of vertices.

        The first 3 vertices will be used to make the plane.
        """
        try:
            pt1, pt2, pt3 = vertices[:3]
            v1 = pt2 - pt1
            v2 = pt3 - pt1
            n = v1.cross(v2)
        except Exception as e:
            raise ValueError('Incorrect vertices input for Face3D:\n\t{}'.format(e))
        return Plane(n, pt1)

    def __copy__(self):
        _new_poly = Face3D(self.vertices, self.plane)
        return _new_poly

    def __repr__(self):
        return 'Face3D ({} vertices)'.format(len(self))
