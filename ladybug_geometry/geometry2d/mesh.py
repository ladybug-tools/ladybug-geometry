# coding=utf-8
"""2D Mesh"""
from __future__ import division

from .._mesh import MeshBase

from .pointvector import Point2D
from .line import LineSegment2D
from .polygon import Polygon2D

try:
    from itertools import izip as zip  # python 2
except ImportError:
    xrange = range  # python 3


class Mesh2D(MeshBase):
    """2D Mesh object.

    Args:
        vertices: A list or tuple of Point2D objects for vertices.
        faces: A list of tuples with each tuple having either 3 or 4 integers.
            These integers correspond to indices within the list of vertices.
        colors: An optional list of colors that correspond to either the faces
            of the mesh or the vertices of the mesh. Default is None.

    Properties:
        * vertices
        * faces
        * colors
        * is_color_by_face
        * min
        * max
        * center
        * area
        * centroid
        * face_areas
        * face_centroids
        * vertex_connected_faces
    """
    __slots__ = ('_min', '_max', '_center', '_centroid')

    def __init__(self, vertices, faces, colors=None):
        """Initilize Mesh2D.
        """
        self._vertices = self._check_vertices_input(vertices)
        self._faces = self._check_faces_input(faces)
        self._is_color_by_face = False  # default if colors is None
        self.colors = colors

        self._min = None
        self._max = None
        self._center = None
        self._area = None
        self._centroid = None
        self._face_areas = None
        self._face_centroids = None
        self._vertex_connected_faces = None

    @classmethod
    def from_dict(cls, data):
        """Create a Mesh2D from a dictionary.

        Args:
            data: A python dictionary in the following format

        .. code-block:: python

            {
            "type": "Mesh2D",
            "vertices": [(0, 0), (10, 0), (0, 10)],
            "faces": [(0, 1, 2)],
            "colors": [{"r": 255, "g": 0, "b": 0}]
            }
        """
        colors = None
        if 'colors' in data and data['colors'] is not None:
            try:
                from ladybug.color import Color
            except ImportError:
                raise ImportError('Colors are specified in input Mesh2D dictionary '
                                  'but failed to import ladybug.color')
            colors = tuple(Color.from_dict(col) for col in data['colors'])
        return cls(tuple(Point2D.from_array(pt) for pt in data['vertices']),
                   data['faces'], colors)

    @classmethod
    def from_face_vertices(cls, faces, purge=True):
        """Create a mesh from a list of faces with each face defined by a list of Point2Ds.

        Args:
            faces: A list of faces with each face defined as a list of 3 or 4 Point2D.
            purge: A boolean to indicate if duplicate vertices should be shared between
                faces. Default is True to purge duplicate vertices, which can be slow
                for large lists of faces but results in a higher-quality mesh with
                a smaller size in memory. Default is True.
        """
        vertices, face_collector = cls._interpret_input_from_face_vertices(faces, purge)
        return cls(tuple(vertices), tuple(face_collector))

    @classmethod
    def from_polygon_triangulated(cls, polygon, purge=True):
        """Initialize a triangulated Mesh2D from a Polygon2D.

        Such a triangulated mesh will completely fill the polygon since any
        polygon can be subdivided into a set of triangles.

        Args:
            polygon: A Polygon2D.
            purge: A boolean to indicate if duplicate vertices should be shared between
                faces. This has no bearing on the triagnulation of convex polygons
                and only affects concave polygons with more than 4 sides.
                Default is True to purge duplicate vertices, which can be slow
                for convace polygons composed of a large number of triangles but
                ultimately results in a higher-quality mesh. Default is True.
        """
        assert isinstance(polygon, Polygon2D), 'polygon must be a Polygon2D to use ' \
            'from_polygon_triangulated. Got {}.'.format(type(polygon))

        if polygon.is_convex:
            # super-fast fan triangulation!
            _faces = []
            for i in xrange(1, len(polygon) - 1):
                _faces.append((0, i, i + 1))
            _new_mesh = cls(polygon.vertices, _faces)
        elif len(polygon) == 4:
            _faces = Mesh2D._concave_quad_to_triangles(polygon.vertices)
            _new_mesh = cls(polygon.vertices, _faces)
        else:
            # slow ear-clipping method
            _faces = Mesh2D._ear_clipping_triangulation(polygon)
            _new_mesh = cls.from_face_vertices(_faces, purge)

        return _new_mesh

    @classmethod
    def from_polygon_grid(cls, polygon, x_dim, y_dim, generate_centroids=True):
        """Initialize a gridded Mesh2D from a Polygon2D.

        Note that this gridded mesh will likely not completely fill the polygon.
        Essentially, this method generates a grid over the domain of the polygon
        and then removes any points that do not lie within the polygon.

        Args:
            polygon: A Polygon2D object.
            x_dim: The x dimension of the grid cells as a number.
            y_dim: The y dimension of the grid cells as a number.
            generate_centroids: Set to True to have the face centroids generated
                alongside the grid of vertices, which is much faster than having
                them generated upon request as they typically are. However, if you
                have no need for the face centroids, you would save memory by setting
                this to False. Default is True.
        """
        assert isinstance(polygon, Polygon2D), 'Expected Polygon2D for' \
            ' Mesh2D.from_polygon_grid. Got {}'.format(type(polygon))
        # figure out how many x and y cells to make
        _x_dim, _num_x = Mesh2D._domain_dimensions(polygon.max.x - polygon.min.x, x_dim)
        _y_dim, _num_y = Mesh2D._domain_dimensions(polygon.max.y - polygon.min.y, y_dim)
        _poly_min = polygon.min

        # generate the gid of points and faces
        _verts = Mesh2D._grid_vertices(_poly_min, _num_x, _num_y, _x_dim, _y_dim)
        _faces = Mesh2D._grid_faces(_num_x, _num_y)
        _centroids = None
        if generate_centroids is True:  # calculate centroids if requested
            _centroids = Mesh2D._grid_centroids(
                _poly_min, _num_x, _num_y, _x_dim, _y_dim)

        # figure out which vertices lie inside the polygon
        # for tolerance reasons, we scale the polygon by a very small amount
        # this avoids the fringe cases noted in the Polygon2d.is_point_inside description
        tol_pt = Point2D(0.0000001, 0.0000001)
        scaled_poly = Polygon2D(
            tuple(pt.scale(1.000001, _poly_min) - tol_pt for pt in polygon.vertices))
        _pattern = [scaled_poly.is_point_inside(_v) for _v in _verts]

        # build the mesh
        _mesh_init = cls(_verts, _faces)
        _mesh_init._face_centroids = _centroids
        _new_mesh, _face_pattern = _mesh_init.remove_vertices(_pattern)
        _new_mesh._face_areas = x_dim * y_dim
        return _new_mesh

    @classmethod
    def from_grid(cls, base_point=Point2D(), num_x=1, num_y=1, x_dim=1, y_dim=1,
                  generate_centroids=True):
        """Initialize a Mesh2D from parameters that define a grid.

        Args:
            base_point: The base point from which the mesh grid will be generated.
                Default is (0, 0).
            num_x: An integer for the number of mesh cells to generate in the
                x direction. Default is 1.
            num_y: An integer for the number of mesh cells to generate in the
                y direction. Default is 1.
            x_dim: The x dimension of the grid cells as a number. Default is 1.
            y_dim: The y dimension of the grid cells as a number. Default is 1.
            generate_centroids: Set to True to have the face centroids generated
                alongside the grid of vertices, which is much faster than having
                them generated upon request as they typically are. However, if you
                have no need for the face centroids, you would save memory by setting
                this to False. Default is True.
        """
        _verts = Mesh2D._grid_vertices(base_point, num_x, num_y, x_dim, y_dim)
        _faces = Mesh2D._grid_faces(num_x, num_y)
        _centroids = None
        if generate_centroids is True:
            _centroids = Mesh2D._grid_centroids(base_point, num_x, num_y, x_dim, y_dim)

        _new_mesh = cls(tuple(_verts), tuple(_faces))
        _new_mesh._face_areas = x_dim * y_dim
        _new_mesh._face_centroids = _centroids
        return _new_mesh

    @property
    def min(self):
        """A Point2D for the minimum bounding rectangle vertex around this geometry."""
        if self._min is None:
            self._calculate_min_max()
        return self._min

    @property
    def max(self):
        """A Point2D for the maximum bounding rectangle vertex around this geometry."""
        if self._max is None:
            self._calculate_min_max()
        return self._max

    @property
    def center(self):
        """A Point2D for the center of the bounding rectangle around this geometry."""
        if self._center is None:
            min, max = self.min, self.max
            self._center = Point2D((min.x + max.x) / 2, (min.y + max.y) / 2)
        return self._center

    @property
    def face_areas(self):
        """A tuple of face areas that parallels the faces property."""
        if self._face_areas is None:
            self._face_areas = tuple(self._face_area(face) for face in self.faces)
        elif isinstance(self._face_areas, (float, int)):  # grid of faces with same area
            self._face_areas = tuple(self._face_areas for face in self.faces)
        return self._face_areas

    @property
    def centroid(self):
        """The centroid of the mesh as a Point2D (aka. center of mass).

        Note that the centroid is more time consuming to compute than the center
        (or the middle point of the bounding rectangle). So the center might be
        preferred over the centroid if you just need a rough point for the middle
        of the mesh.
        """
        if self._centroid is None:
            _weight_x = 0
            _weight_y = 0
            for _c, _a in zip(self.face_centroids, self.face_areas):
                _weight_x += _c.x * _a
                _weight_y += _c.y * _a
            self._centroid = Point2D(_weight_x / self.area, _weight_y / self.area)
        return self._centroid

    def triangulated(self):
        """Get a version of this Mesh2D where all quads have been triangulated."""
        _new_faces = []
        for face in self.faces:
            if len(face) == 3:
                _new_faces.append(face)
            else:
                _triangles = Mesh2D._quad_to_triangles([self._vertices[i] for i in face])
                _new_faces.extend(_triangles)
        _new_faces = tuple(_new_faces)

        _new_colors = self.colors
        if self.is_color_by_face is True:
            _new_colors = []
            for i, face in enumerate(self.faces):
                if len(face) == 3:
                    _new_colors.append(self.colors[i])
                else:
                    _new_colors.extend([self.colors[i]] * 2)
            _new_colors = tuple(_new_colors)

        _new_mesh = Mesh2D(self.vertices, _new_faces, _new_colors)
        return _new_mesh

    def remove_vertices(self, pattern):
        """Get a version of this mesh where vertices are removed according to a pattern.

        Args:
            pattern: A list of boolean values denoting whether a vertex should
                remain in the mesh (True) or be removed from the mesh (False).
                The length of this list must match the number of this mesh's vertices.

        Returns:
            A tuple with two elements

            -   new_mesh:
                A mesh where the vertices have been removed according
                to the input pattern.

            -   face_pattern:
                A list of boolean values that corresponds to the
                original mesh faces noting whether the face is in the new mesh (True)
                or has been removed from the new mesh (False).
        """
        _new_verts, _new_faces, _new_colors, _new_f_cent, _new_f_area, face_pattern = \
            self._remove_vertices(pattern)

        new_mesh = Mesh2D(_new_verts, _new_faces, _new_colors)
        new_mesh._face_centroids = _new_f_cent
        new_mesh._face_areas = _new_f_area
        return new_mesh, face_pattern

    def remove_faces(self, pattern):
        """Get a version of this mesh where faces are removed according to a pattern.

        Args:
            pattern: A list of boolean values denoting whether a face should
                remain in the mesh (True) or be removed from the mesh (False).
                The length of this list must match the number of this mesh's faces.

        Returns:
            A tuple with two elements

            -   new_mesh:
                A mesh where the faces have been removed according
                to the input pattern.

            -   vertex_pattern:
                A list of boolean values that corresponds to the
                original mesh vertices noting whether the vertex is in the new mesh
                (True) or has been removed from the new mesh (False).
        """
        vertex_pattern = self._vertex_pattern_from_remove_faces(pattern)
        _new_verts, _new_faces, _new_colors, _new_f_cent, _new_f_area, face_pattern = \
            self._remove_vertices(vertex_pattern, pattern)

        new_mesh = Mesh2D(_new_verts, _new_faces, _new_colors)
        new_mesh._face_centroids = _new_f_cent
        new_mesh._face_areas = _new_f_area
        return new_mesh, vertex_pattern

    def remove_faces_only(self, pattern):
        """Get a version of this mesh where faces are removed and vertices are not altered.

        This is faster than the Mesh2D.remove_faces method but will likely result
        a lower-quality mesh where several vertices exist in the mesh that are not
        referenced by any face. This may be preferred if pure speed of removing
        faces is a priorty over smallest size of the mesh in memory.

        Args:
            pattern: A list of boolean values denoting whether a face should
                remain in the mesh (True) or be removed from the mesh (False).
                The length of this list must match the number of this mesh's faces.

        Returns:
            new_mesh -- A mesh where the faces have been removed according
            to the input pattern.
        """
        _new_faces, _new_colors, _new_f_cent, _new_f_area = \
            self._remove_faces_only(pattern)

        new_mesh = Mesh2D(self.vertices, _new_faces, _new_colors)
        new_mesh._face_centroids = _new_f_cent
        new_mesh._face_areas = _new_f_area
        return new_mesh

    def rotate(self, angle, origin):
        """Get a mesh that is rotated counterclockwise by a certain angle.

        Args:
            angle: An angle for rotation in radians.
            origin: A Point2D for the origin around which the point will be rotated.
        """
        _verts = tuple([pt.rotate(angle, origin) for pt in self.vertices])
        return self._mesh_transform(_verts)

    def scale(self, factor, origin=None):
        """Scale a mesh by a factor from an origin point.

        Args:
            factor: A number representing how much the mesh should be scaled.
            origin: A Point representing the origin from which to scale.
                If None, it will be scaled from the World origin (0, 0).
        """
        if origin is None:
            _verts = tuple(
                Point2D(pt.x * factor, pt.y * factor) for pt in self.vertices)
        else:
            _verts = tuple(pt.scale(factor, origin) for pt in self.vertices)
        return self._mesh_scale(_verts, factor)

    def to_dict(self):
        """Get Mesh2D as a dictionary."""
        colors = None
        if self.colors is not None:
            colors = [col.to_dict() for col in self.colors]
        return {'type': 'Mesh2D',
                'vertices': [pt.to_array() for pt in self.vertices],
                'faces': self.faces, 'colors': colors}

    def _calculate_min_max(self):
        """Calculate maximum and minimum Point2D for this object."""
        min_pt = [self.vertices[0].x, self.vertices[0].y]
        max_pt = [self.vertices[0].x, self.vertices[0].y]

        for v in self.vertices[1:]:
            if v.x < min_pt[0]:
                min_pt[0] = v.x
            elif v.x > max_pt[0]:
                max_pt[0] = v.x
            if v.y < min_pt[1]:
                min_pt[1] = v.y
            elif v.y > max_pt[1]:
                max_pt[1] = v.y

        self._min = Point2D(min_pt[0], min_pt[1])
        self._max = Point2D(max_pt[0], max_pt[1])

    def _face_area(self, face):
        """Return the area of a face."""
        return Mesh2D._get_area(tuple(self._vertices[i] for i in face))

    def _tri_face_centroid(self, face):
        """Compute the centroid of a triangular face."""
        return Mesh2D._tri_centroid(tuple(self._vertices[i] for i in face))

    def _quad_face_centroid(self, face):
        """Compute the centroid of a quadrilateral face."""
        return Mesh2D._quad_centroid(tuple(self._vertices[i] for i in face))

    def _mesh_transform(self, verts):
        """Transform mesh in a way that transfers properties and avoids extra checks."""
        _new_mesh = Mesh2D(verts, self.faces)
        self._transfer_properties(_new_mesh)
        return _new_mesh

    def _mesh_scale(self, verts, factor):
        """Scale mesh in a way that transfers properties and avoids extra checks."""
        _new_mesh = Mesh2D(verts, self.faces)
        self._transfer_properties_scale(_new_mesh, factor)
        return _new_mesh

    def _check_vertices_input(self, vertices):
        if not isinstance(vertices, tuple):
            vertices = tuple(vertices)
        for vert in vertices:
            assert isinstance(vert, Point2D), \
                'Expected Point2D for {} vertex. Got {}.'.format(
                    self.__class__.__name__, type(vert))
        return vertices

    @staticmethod
    def _ear_clipping_triangulation(polygon):
        """Triangulate a polygon using the ear clipping method."""
        _faces = []
        _clipping_poly = polygon.duplicate()
        while len(_clipping_poly) > 3:
            _ear, _index = Mesh2D._find_ear(_clipping_poly)
            _faces.append(_ear)
            new_verts = list(_clipping_poly.vertices)
            del new_verts[_index]
            _clipping_poly = Polygon2D(new_verts)
        _faces.append(_clipping_poly.vertices)
        return _faces

    @staticmethod
    def _find_ear(polygon):
        """Find an ear of a polygon.

        An ear is a triangle with two sides as edges of the polygon and the third
        side is a diagonal that lies completely inside the polygon. Whether a polygon
        is convex or concave, it always has at least two ears that can be "clipped"
        to yield a simpler polygon.
        """
        for i in xrange(0, len(polygon) - 1):
            diagonal = LineSegment2D.from_end_points(polygon[i - 1], polygon[i + 1])
            if polygon.is_point_inside(diagonal.midpoint):
                if len(polygon.intersect_line_ray(diagonal)) < 5:
                    ear = (polygon[i - 1], polygon[i], polygon[i + 1])  # found an ear!
                    break
        try:
            return ear, i
        except UnboundLocalError:
            raise ValueError(
                'Polygon {} is not in a valid format for triangulation.'
                ' The polygon likely has self-intersecting edges.'.format(polygon))

    @staticmethod
    def _quad_to_triangles(verts):
        """Return two triangles that represent any quadrilateral."""
        # check if the quad is convex
        convex = True
        pt1, pt2, pt3 = verts[1], verts[2], verts[3]
        start_val = True if (pt2.x - pt1.x) * (pt3.y - pt2.y) - \
            (pt2.y - pt1.y) * (pt3.x - pt2.x) > 0 else False
        for i, pt3 in enumerate(verts[:3]):
            pt1 = verts[i - 2]
            pt2 = verts[i - 1]
            val = True if (pt2.x - pt1.x) * (pt3.y - pt2.y) - \
                (pt2.y - pt1.y) * (pt3.x - pt2.x) > 0 else False
            if val is not start_val:
                convex = False
                break
        if convex is True:
            # if the quad is convex, either diagonal splits it into triangles
            return [(0, 1, 2), (2, 3, 0)]
        else:
            # if it is concave, we need to select the right diagonal of the two
            return Mesh2D._concave_quad_to_triangles(verts)

    @staticmethod
    def _concave_quad_to_triangles(verts):
        """Return two triangles that represent a concave quadrilateral."""
        quad_poly = Polygon2D(verts)
        diagonal = LineSegment2D.from_end_points(quad_poly[0], quad_poly[2])
        if quad_poly.is_point_inside(diagonal.midpoint):
            # if the diagonal midpoint is inside the quad, it splits it into two ears
            return [(0, 1, 2), (2, 3, 0)]
        else:
            # if not, then the other diagonal splits it into two ears
            return [(1, 2, 3), (3, 0, 1)]

    @staticmethod
    def _quad_centroid(verts):
        """Get the centroid of a list of 4 Point2D vertices."""
        _tri_i = Mesh2D._quad_to_triangles(verts)
        _tri_verts = ([verts[i] for i in _tri_i[0]], [verts[i] for i in _tri_i[1]])
        _tri_c = [Mesh2D._tri_centroid(tri) for tri in _tri_verts]
        _tri_a = [Mesh2D._get_area(tri) for tri in _tri_verts]
        _tot_a = sum(_tri_a)
        _cent_x = (_tri_c[0].x * _tri_a[0] + _tri_c[1].x * _tri_a[1]) / _tot_a
        _cent_y = (_tri_c[0].y * _tri_a[0] + _tri_c[1].y * _tri_a[1]) / _tot_a
        return Point2D(_cent_x, _cent_y)

    @staticmethod
    def _tri_centroid(verts):
        """Get the centroid of a list of 3 Point2D vertices."""
        _cent_x = sum([v.x for v in verts])
        _cent_y = sum([v.y for v in verts])
        return Point2D(_cent_x / 3, _cent_y / 3)

    @staticmethod
    def _get_area(verts):
        """Return the area of a list of Point2D vertices."""
        _a = 0
        for i, pt in enumerate(verts):
            _a += verts[i - 1].x * pt.y - verts[i - 1].y * pt.x
        return abs(_a / 2)

    @staticmethod
    def _domain_dimensions(_dom, _dim):
        """Get corrected dimensions and number of cells over a domain."""
        _num = int(_dom / _dim)
        _num = 1 if _num == 0 else _num
        _dim = _dom / _num
        return _dim, _num

    @staticmethod
    def _grid_vertices(base_point, num_x, num_y, x_dim, y_dim):
        """Generate Point2D vertices for a grid."""
        _verts = []
        _x = base_point.x
        for i in xrange(num_x + 1):
            _y = base_point.y
            for j in xrange(num_y + 1):
                _verts.append(Point2D(_x, _y))
                _y += y_dim
            _x += x_dim
        return _verts

    @staticmethod
    def _grid_faces(num_x, num_y):
        """Generate face tuples for a grid."""
        _faces = []
        _c = 0
        for i in xrange(num_x):
            for j in xrange(num_y):
                _faces.append((_c, _c + num_y + 1, _c + num_y + 2, _c + 1))
                _c += 1
            _c += 1
        return _faces

    @staticmethod
    def _grid_centroids(base_point, num_x, num_y, x_dim, y_dim):
        """Generate Point2D centroids for a grid."""
        _centroids = []
        _x_half = x_dim / 2
        _y_half = y_dim / 2
        _x = base_point.x
        for i in xrange(num_x):
            _y = base_point.y
            for j in xrange(num_y):
                _centroids.append(Point2D(_x + _x_half, _y + _y_half))
                _y += y_dim
            _x += x_dim
        return tuple(_centroids)

    def __copy__(self):
        _new_mesh = Mesh2D(self.vertices, self.faces)
        self._transfer_properties(_new_mesh)
        _new_mesh._face_centroids = self._face_centroids
        _new_mesh._centroid = self._centroid
        return _new_mesh

    def __key(self):
        """A tuple based on the object properties, useful for hashing."""
        return tuple(hash(pt) for pt in self._vertices) + \
            tuple(hash(face) for face in self._faces)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(other, Mesh2D) and self.__key() == other.__key()

    def __repr__(self):
        return 'Ladybug Mesh2D ({} faces) ({} vertices)'.format(
            len(self.faces), len(self))
