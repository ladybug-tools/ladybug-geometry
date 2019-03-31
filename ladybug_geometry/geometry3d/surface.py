# coding=utf-8
"""Surface in 3D Space"""
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


class Surface3D(Base2DIn3D):
    """Surface in 3D space.

    Properties:
        vertices
        plane
        normal
        min
        max
        center
        segments
        perimeter
        area
        centroid
        is_clockwise
        is_convex
        is_self_intersecting
        polygon2d
        triangulated_mesh2d
        triangulated_mesh3d
    """
    _check_required = True
    _segments = None
    _polygon2d = None
    _mesh2d = None
    _mesh3d = None
    _perimeter = None
    _area = None
    _centroid = None
    _is_clockwise = None
    _is_convex = None
    _is_complex = None

    def __init__(self, vertices, plane):
        """Initilize Surface3D.

        Args:
            vertices: A list of Point3D objects representing the vertices of the surface.
            plane: A Plane object indicating the plane in which the surface exists.
        """
        if self._check_required:
            self._check_vertices_input(vertices)
            assert isinstance(plane, Plane), 'Expected Plane for Surface3D.' \
                ' Got {}.'.format(type(plane))
        else:
            self._vertices = vertices
        self._plane = plane

    @classmethod
    def from_vertices(cls, vertices):
        """Initialize Surface3D from only a list of vertices.

        The Plane normal will automatically be calculated by analyzing the first
        three vertices and the origin of the plane will be the first vertex of
        the input vertices.
        """
        try:
            pt1, pt2, pt3 = vertices[:3]
            v1 = pt2 - pt1
            v2 = pt3 - pt1
            n = v1.cross(v2)
        except Exception as e:
            raise ValueError('Incorrect vertices input for Surface3D:\n\t{}'.format(e))
        plane = Plane(n, pt1)
        return cls(vertices, plane)

    @classmethod
    def from_extrusion(cls, line_segment, extrusion_vector):
        """Initialize Surface3D by extruding a line segment.

        Initializing a surface this way has the added benefit of having its
        properties quickly computed.

        Args:
            line_segment: A LineSegment3D to be extruded.
            extrusion_vector: A vector denoting the direction and distance to
                extrude the line segment.
        """
        assert isinstance(line_segment, (LineSegment3D, LineSegment3DImmutable)), \
            'line_segment must be LineSegment3D. Got {}.'.format(type(line_segment))
        assert isinstance(extrusion_vector, (Vector3D, Vector3DImmutable)), \
            'extrusion_vector must be Vector3D. Got {}.'.format(type(extrusion_vector))
        _p1 = line_segment.p1
        _p2 = line_segment.p2
        _verts = (_p1, _p1 + extrusion_vector, _p2 + extrusion_vector, _p2)
        _plane = Plane(line_segment.v.cross(extrusion_vector), _p1)
        Surface3D._check_required = False  # Turn off check since input is valid
        surface = cls(_verts, _plane)
        Surface3D._check_required = True  # Turn the checks back on
        _base = line_segment.length
        _dist = extrusion_vector.magnitude
        _height = _dist * math.sin(extrusion_vector.angle(line_segment.v))
        surface._perimeter = _base * 2 + _dist * 2
        surface._area = _base * _height
        _cent = _p1 + (line_segment.v * 0.5) + (extrusion_vector * 0.5)
        surface._centroid = _cent.to_immutable()
        surface._is_clockwise = True
        surface._is_convex = False
        surface._is_complex = False
        return surface

    @classmethod
    def from_rectangle(cls, base, height, base_plane=None):
        """Initialize Surface3D from rectangle parameters (base + height) and a base plane.

        Initializing a surface this way has the added benefit of having its
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
        Surface3D._check_required = False  # Turn off check since input is valid
        surface = cls(_verts, base_plane)
        Surface3D._check_required = True  # Turn the checks back on
        surface._perimeter = base * 2 + height * 2
        surface._area = base * height
        _cent = _o + (_b_vec * 0.5) + (_h_vec * 0.5)
        surface._centroid = _cent.to_immutable()
        surface._is_clockwise = True
        surface._is_convex = False
        surface._is_complex = False
        return surface

    @property
    def vertices(self):
        """Tuple of all vertices in this surface."""
        return self._vertices

    @property
    def plane(self):
        """Tuple of all vertices in this geometry."""
        return self._plane

    @property
    def normal(self):
        """Normal vector for the plane in which the surface exists."""
        return self._plane.n

    @property
    def segments(self):
        """Tuple of all line segments in the polygon."""
        if self._segments is None:
            _segs = []
            for i, vert in enumerate(self.vertices):
                _seg = LineSegment3DImmutable.from_end_points(self.vertices[i - 1], vert)
                _segs.append(_seg)
            _segs.append(_segs.pop(0))  # segments will start from the first vertex
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
        """The area of the surface."""
        if self._area is None:
            self._area = self.polygon2d.area
        return self._area

    @property
    def centroid(self):
        """The centroid of the surface as a Point3D (aka. center of mass).

        Note that the centroid is more time consuming to compute than the center
        (or the middle point of the surface bounding box). So the center might be
        preferred over the centroid if you just need a rough point for the middle
        of the surface. Also note that, if the surface is convex, the center will
        always lie within or on the edge of the surface.
        """
        if self._centroid is None:
            _cent2d = self.triangulated_mesh2d.centroid
            self._centroid = self._plane.xy_to_xyz_immutable(_cent2d)
        return self._centroid

    @property
    def is_clockwise(self):
        """Boolean for whether the surface vertices are in clockwise order."""
        if self._is_clockwise is None:
            self._is_clockwise = self.polygon2d.is_clockwise
        return self._is_clockwise

    @property
    def is_convex(self):
        """Boolean noting whether the polygon is convex (True) or non-convex (False)."""
        if self._is_convex is None:
            self._is_convex = self.polygon2d.is_convex
        return self._is_convex

    @property
    def is_self_intersecting(self):
        """Boolean noting whether the surface has self-intersecting edges.

        Note that this property is relatively computationally intense to obtain and
        most CAD programs forbid surfaces with self-intersecting edges.
        So this property should only be used in quality control scripts where the
        origin of the geometry is unknown.
        """
        if self._is_complex is None:
            self._is_complex = self.polygon2d.is_self_intersecting
        return self._is_complex

    @property
    def polygon2d(self):
        """A Polygon2D of this surface in the 2D space of the surface's plane."""
        if self._polygon2d is None:
            _vert2d = [self._plane.xyz_to_xy_immutable(_v) for _v in self.vertices]
            Polygon2D._check_required = False  # Turn off check since input is valid
            self._polygon2d = Polygon2D(tuple(_vert2d))
            Polygon2D._check_required = True  # Turn the checks back on
            if self._is_clockwise is not None:
                self._polygon2d._is_clockwise = self._is_clockwise
        return self._polygon2d

    @property
    def triangulated_mesh2d(self):
        """A triagulated Mesh2D in the 2D space of the surface's plane."""
        if self._mesh2d is None:
            self._mesh2d = Mesh2D.from_polygon_triangulated(self.polygon2d)
        return self._mesh2d

    @property
    def triangulated_mesh3d(self):
        """A triagulated Mesh3D of this surface."""
        if self._mesh3d is None:
            _vert3d = [self._plane.xy_to_xyz_immutable(_v) for _v in
                       self.triangulated_mesh2d.vertices]
            self._mesh3d = Mesh3D(_vert3d, self.triangulated_mesh2d.faces)
        return self._mesh3d

    def validate_planarity(self, tolerance, raise_exception=True):
        """Validate that all of the surface's vertices lie within the surface's plane.

        This check is not done by default when creating the surface since
        it is assumed that there is likely a check for planarity before the surface
        is created (ie. in CAD software where the surface might originate from).
        This method is intended for quality control checking when the origin of
        surface geometry is unkown or is known to come from a place where no
        planarity check was performed.

        Args:
            tolerance: The minimum distance between a given vertex and a the
                surface's plane at which the vertex is said to lie in the plane.
            raise_exception: Boolean to note whether an exception should be raised
                if a vertex does not lie within the surface's plane. If True, an
                exception message will be given in such cases, which notes the
                vertex and its distance from the plane. If False, this method will
                simply return a False boolean if a vertex is found that is out
                of plane. Default is True to raise an exception.
        """
        for _v in self.vertices:
            if self._plane.distance_to_point(_v) > tolerance:
                if raise_exception is True:
                    raise AttributeError(
                        'Vertex {} is out of plane with its parent surface.\nDistance '
                        'to plane is {}'.format(_v, self._plane.distance_to_point(_v)))
                else:
                    return False
        return True

    def flip(self):
        """Get a surface with a flipped direction from this one."""
        Surface3D._check_required = False  # Turn off check since input is valid
        _new_srf = Surface3D(self.vertices, self.plane.flip())
        Surface3D._check_required = True  # Turn the checks back on
        self._transfer_properties(_new_srf)
        if self._is_clockwise is not None:
            _new_srf._is_clockwise = not self._is_clockwise
        return _new_srf

    def move(self, moving_vec):
        """Get a surface that has been moved along a vector.

        Args:
            moving_vec: A Vector3D with the direction and distance to move the surface.
        """
        _verts = tuple([Point3DImmutable(
            pt.x + moving_vec.x, pt.y + moving_vec.y, pt.z + moving_vec.z)
                        for pt in self.vertices])
        return self._surface_transform(_verts, self.plane.move(moving_vec))

    def rotate(self, axis, angle, origin):
        """Rotate a surface by a certain angle around an axis and origin.

        Right hand rule applies:
        If axis has a positive orientation, rotation will be clockwise.
        If axis has a negative orientation, rotation will be counterclockwise.

        Args:
            axis: A Vector3D axis representing the axis of rotation.
            angle: An angle for rotation in radians.
            origin: A Point3D for the origin around which the object will be rotated.
        """
        _verts = tuple([pt.rotate(axis, angle, origin).to_immutable()
                        for pt in self.vertices])
        return self._surface_transform(_verts, self.plane.rotate(axis, angle, origin))

    def rotate_xy(self, angle, origin):
        """Get a surface rotated counterclockwise in the world XY plane by a certain angle.

        Args:
            angle: An angle in radians.
            origin: A Point3D for the origin around which the object will be rotated.
        """
        _verts = tuple([pt.rotate_xy(angle, origin).to_immutable()
                        for pt in self.vertices])
        return self._surface_transform(_verts, self.plane.rotate_xy(angle, origin))

    def reflect(self, normal, origin):
        """Get a surface reflected across a plane with the input normal vector and origin.

        Args:
            normal: A Vector3D representing the normal vector for the plane across
                which the surface will be reflected. THIS VECTOR MUST BE NORMALIZED.
            origin: A Point3D representing the origin from which to reflect.
        """
        _verts = tuple([pt.reflect(normal, origin).to_immutable()
                        for pt in self.vertices])
        return self._surface_transform(_verts, self.plane.reflect(normal, origin))

    def scale(self, factor, origin):
        """Scale a surface by a factor from an origin point.

        Args:
            factor: A number representing how much the surface should be scaled.
            origin: A Point3D representing the origin from which to scale.
        """
        _verts = tuple([pt.scale(factor, origin).to_immutable() for pt in self.vertices])
        return self._surface_transform_scale(
            _verts, self.plane.scale(factor, origin), factor)

    def scale_world_origin(self, factor):
        """Scale a surface by a factor from the world origin. Faster than Surface3D.scale.

        Args:
            factor: A number representing how much the line segment should be scaled.
        """
        _verts = tuple([pt.scale_world_origin(factor).to_immutable()
                        for pt in self.vertices])
        return self._surface_transform_scale(
            _verts, self.plane.scale_world_origin(factor), factor)

    def intersect_line_ray(self, line_ray):
        """Get the intersection between this surface and the input Line3D or Ray3D.

        Args:
            line_ray: A Line3D or Ray3D object for which intersection will be computed.

        Returns:
            Point3D for the intersection. Will be None if no intersection exists.
        """
        _plane_int = self._plane.intersect_line(line_ray)
        if _plane_int is not None:
            _int2d = self._plane.xyz_to_xy(_plane_int)
            if self.polygon2d.is_point_inside_bound_rect(_int2d):
                return _plane_int
        return None

    def intersect_plane(self, plane):
        """Get the intersection between this surface and the input plane.

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
        """Project a Point3D onto this surface.

        Note that this method does a check to see if the point can be projected to
        within this surface's boundary. If all that is needed is a point projected
        into the plane of this surface, the Plane.project_point() method should be
        used with this surface's plane property.

        Args:
            point: A Point3D object to project.

        Returns:
            Point3D for the point projected onto this surface. Will be None if the
                point cannot be projected to within the boundary of the surface.
        """
        _plane_int = self._plane.project_point(point)
        _plane_int2d = self._plane.xyz_to_xy(_plane_int)
        if self.polygon2d.is_point_inside_bound_rect(_plane_int2d):
            return _plane_int
        return None

    def get_mesh_grid(self, x_dim, y_dim=None, offset=None, flip=False,
                      generate_centroids=True):
        """Get a gridded Mesh3D from over this surface.

        Note that this gridded mesh will likely not completely fill the surface.
        Essentially, this method generates a grid over the domain of the surface
        and then removes any points that do not lie within it.

        Args:
            x_dim: The x dimension of the grid cells as a number.
            y_dim: The y dimension of the grid cells as a number. Default is None,
                which will assume the same cell dimension for y as is set for x.
            offset: A number for how far to offset the grid from the base surface.
                Default is None, which will not offset the grid at all.
            flip: Set to True to have the mesh normals reversed from the direction
                of this surface and to have the offset input move the mesh in the
                opposite direction from this surface's normal.
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
        Mesh3D._check_required = False  # Turn off check since input is valid
        grid_mesh3d = Mesh3D(vert_3d, grid_mesh2d.faces)
        Mesh3D._check_required = True  # Turn the checks back on
        grid_mesh3d._face_areas = grid_mesh2d._face_areas

        # assign the surface plane normal to the mesh normals
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
        assert len(vertices) > 2, 'There must be at least 3 vertices for a Surface.' \
            ' Got {}'.format(len(vertices))
        _verts_immutable = []
        for p in vertices:
            assert isinstance(p, (Point3D, Point3DImmutable)), \
                'Expected Point3D. Got {}.'.format(type(p))
            _verts_immutable.append(p.to_immutable())
        self._vertices = tuple(_verts_immutable)

    def _check_number_mesh_grid(self, input, name):
        assert isinstance(input, (float, int)), '{} for Surface3D.get_mesh_grid' \
            ' must be a number. Got {}.'.format(name, type(input))

    def _surface_transform(self, verts, plane):
        """Transform surface in a way that transfers properties and avoids checks."""
        Surface3D._check_required = False  # Turn off check since input is valid
        _new_srf = Surface3D(verts, plane)
        Surface3D._check_required = True  # Turn the checks back on
        self._transfer_properties(_new_srf)
        return _new_srf

    def _surface_transform_scale(self, verts, plane, factor):
        """Scale surface in a way that transfers properties and avoids checks."""
        Surface3D._check_required = False  # Turn off check since input is valid
        _new_srf = Surface3D(verts, plane)
        Surface3D._check_required = True  # Turn the checks back on
        self._transfer_properties_scale(_new_srf, factor)
        return _new_srf

    def _transfer_properties(self, new_srf):
        """Transfer properties from this surface to a new surface.

        This is used by the transform methods that don't alter the relationship of
        surface vertices to one another (move, rotate, reflect).
        """
        new_srf._perimeter = self._perimeter
        new_srf._area = self._area
        new_srf._is_convex = self._is_convex
        new_srf._is_complex = self._is_complex
        new_srf._is_clockwise = self._is_clockwise
        new_srf._polygon2d = self._polygon2d
        new_srf._mesh2d = self._mesh2d

    def _transfer_properties_scale(self, new_srf, factor):
        """Transfer properties from this surface to a new surface.

        This is used by the methods that scale the surface.
        """
        new_srf._is_convex = self._is_convex
        new_srf._is_complex = self._is_complex
        new_srf._is_clockwise = self._is_clockwise
        if self._perimeter is not None:
            new_srf._perimeter = self._perimeter * factor
        if self._area is not None:
            new_srf._area = self._area * factor

    def __copy__(self):
        Surface3D._check_required = False  # Turn off check since we know input is valid
        _new_poly = Surface3D(self.vertices, self.plane)
        Surface3D._check_required = True  # Turn the checks back on
        return _new_poly

    def __repr__(self):
        return 'Ladybug Surface3D ({} vertices)'.format(len(self))
