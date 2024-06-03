# coding=utf-8
"""3D Mesh"""
from __future__ import division

from .._mesh import MeshBase
from ..geometry2d.mesh import Mesh2D
from .pointvector import Point3D, Vector3D
from .line import LineSegment3D
from .polyline import Polyline3D
from .plane import Plane

try:
    from itertools import izip as zip  # python 2
except ImportError:
    xrange = range  # python 3


class Mesh3D(MeshBase):
    """3D Mesh object.

    Args:
        vertices: A list or tuple of Point3D objects for vertices.
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
        * face_areas
        * face_centroids
        * face_area_centroids
        * face_vertices
        * face_normals
        * vertex_normals
        * vertex_connected_faces
        * face_edges
        * edges
        * naked_edges
        * internal_edges
        * non_manifold_edges
    """
    __slots__ = ('_min', '_max', '_center', '_face_normals', '_vertex_normals')

    def __init__(self, vertices, faces, colors=None):
        """Initialize Mesh3D."""
        self._vertices = self._check_vertices_input(vertices)
        self._faces = self._check_faces_input(faces)

        self._is_color_by_face = False  # default if colors is None
        self.colors = colors
        self._min = None
        self._max = None
        self._center = None
        self._area = None
        self._face_areas = None
        self._face_centroids = None
        self._face_area_centroids = None
        self._face_normals = None
        self._vertex_normals = None
        self._vertex_connected_faces = None
        self._edge_indices = None
        self._edge_types = None
        self._edges = None
        self._naked_edges = None
        self._internal_edges = None
        self._non_manifold_edges = None

    @classmethod
    def from_dict(cls, data):
        """Create a Mesh3D from a dictionary.

        Args:
            data: A python dictionary in the following format

        .. code-block:: python

            {
                "type": "Mesh3D",
                "vertices": [(0, 0, 0), (10, 0, 0), (0, 10, 0)],
                "faces": [(0, 1, 2)],
                "colors": [{"r": 255, "g": 0, "b": 0}]
            }
        """
        colors = None
        if 'colors' in data and data['colors'] is not None and len(data['colors']) != 0:
            try:
                from ladybug.color import Color
            except ImportError:
                raise ImportError('Colors are specified in input Mesh2D dictionary '
                                  'but failed to import ladybug.color')
            colors = tuple(Color.from_dict(col) for col in data['colors'])
        fcs = tuple(tuple(f) for f in data['faces'])  # cast to immutable type
        return cls(tuple(Point3D.from_array(pt) for pt in data['vertices']), fcs, colors)

    @classmethod
    def from_face_vertices(cls, faces, purge=True):
        """Create a mesh from a list of faces with each face defined by Point3Ds.

        Args:
            faces: A list of faces with each face defined as a list of 3 or 4 Point3D.
            purge: A boolean to indicate if duplicate vertices should be shared between
                faces. Default is True to purge duplicate vertices, which can be slow
                for large lists of faces but results in a higher-quality mesh with
                a smaller size in memory. Note that vertices are only considered
                duplicate if the coordinate values are equal to one another
                within floating point tolerance. To remove duplicate vertices
                within a specified tolerance other than floating point, the
                from_purged_face_vertices method should be used instead.
        """
        vertices, face_collector = cls._interpret_input_from_face_vertices(faces, purge)
        return cls(tuple(vertices), tuple(face_collector))

    @classmethod
    def from_purged_face_vertices(cls, faces, tolerance):
        """Create a mesh from a list of faces with each face defined by Point3Ds.

        This method is slower than 'from_face_vertices' but will result in a mesh
        with fewer vertices and a smaller size in memory. This method is similar to
        using the 'purge' option in 'from_face_vertices' but will result in more shared
        vertices since it uses a tolerance to check equivalent vertices rather than
        comparing within floating point tolerance.

        Args:
            faces: A list of faces with each face defined as a list of 3 or 4 Point3D.
            tolerance: A number for the minimum difference between coordinate
                values at which point vertices are considered equal to one another.
        """
        vertices, faces = cls._interpret_input_from_face_vertices_with_tolerance(
            faces, tolerance)
        return cls(tuple(vertices), tuple(faces))

    @classmethod
    def from_mesh2d(cls, mesh_2d, plane=None):
        """Create a Mesh3D from a Mesh2D and a Plane in which the mesh exists.

        Args:
            mesh_2d: A Mesh2D object.
            plane: A Plane object to represent the plane in which the Mesh2D exists
                within 3D space. If None, the WorldXY plane will be used.
        """
        assert isinstance(mesh_2d, Mesh2D), 'Expected Mesh2D for from_mesh_2d. ' \
            'Got {}.'.format(type(mesh_2d))
        if plane is None:
            return cls(tuple(Point3D(pt.x, pt.y, 0) for pt in mesh_2d.vertices),
                       mesh_2d.faces, mesh_2d.colors)
        else:
            assert isinstance(plane, Plane), 'Expected Plane. Got {}'.format(type(plane))
            _verts3d = tuple(plane.xy_to_xyz(_v) for _v in mesh_2d.vertices)
            return cls(_verts3d, mesh_2d.faces, mesh_2d.colors)

    @classmethod
    def from_stl(cls, file_path):
        """Create a Mesh3D from an STL file.

        Args:
            file_path: Path to an STL file as a text string. The STL file can be
                in either ASCII or binary format.
        """
        from ladybug_geometry.interop.stl import STL  # avoid circular import
        face_vertices = STL.from_file(file_path).face_vertices
        return cls.from_face_vertices(face_vertices)

    @classmethod
    def from_obj(cls, file_path):
        """Create a Mesh3D from an OBJ file.

        Args:
            file_path: Path to an OBJ file as a text string.
        """
        from ladybug_geometry.interop.obj import OBJ  # avoid circular import
        transl_obj = OBJ.from_file(file_path)
        return cls(transl_obj.vertices, transl_obj.faces, transl_obj.vertex_colors)

    @property
    def min(self):
        """A Point3D for the minimum bounding box vertex around this mesh."""
        if self._min is None:
            self._calculate_min_max()
        return self._min

    @property
    def max(self):
        """A Point3D for the maximum bounding box vertex around this mesh."""
        if self._max is None:
            self._calculate_min_max()
        return self._max

    @property
    def center(self):
        """A Point3D for the center of the bounding box around this mesh."""
        if self._center is None:
            min, max = self.min, self.max
            self._center = Point3D(
                (min.x + max.x) / 2, (min.y + max.y) / 2, (min.z + max.z) / 2)
        return self._center

    @property
    def face_areas(self):
        """A tuple of face areas that parallels the faces property."""
        if self._face_normals is None:
            self._calculate_face_areas_and_normals()
        elif isinstance(self._face_areas, (float, int)):  # same area for each face
            self._face_areas = tuple(self._face_areas for face in self.faces)
        return self._face_areas

    @property
    def face_normals(self):
        """Tuple of Vector3D objects for all face normals."""
        if self._face_normals is None:
            self._calculate_face_areas_and_normals()
        elif isinstance(self._face_normals, Vector3D):  # same normal for each face
            self._face_normals = tuple(self._face_normals for face in self.faces)
        return self._face_normals

    @property
    def vertex_normals(self):
        """Tuple of Vector3D objects for all vertex normals."""
        if not self._vertex_normals:
            self._calculate_vertex_normals()
        elif isinstance(self._vertex_normals, Vector3D):  # same normal for each vertex
            self._vertex_normals = tuple(self._vertex_normals for face in self.vertices)
        return self._vertex_normals

    @property
    def face_edges(self):
        """List of polylines with one Polyline3D for each face.
        
        This is faster to compute compared to the edges and results in effectively
        the same type of wireframe visualization.
        """
        _all_verts = self._vertices
        f_edges = []
        for face in self._faces:
            verts = tuple(_all_verts[v] for v in face) + (_all_verts[face[0]],)
            f_edges.append(Polyline3D(verts))
        return f_edges

    @property
    def edges(self):
        """"Tuple of all edges in this Mesh3D as LineSegment3D objects.
        
        Note that this method will return only the unique edges in the mesh without
        any duplicates. This is sometimes desirable but can take a lot of time
        to compute for large meshes. For a faster property, use face_edges."""
        if self._edges is None:
            if self._edge_indices is None:
                self._compute_edge_info()
            self._edges = tuple(LineSegment3D.from_end_points(
                self.vertices[seg[0]], self.vertices[seg[1]])
                for seg in self._edge_indices)
        return self._edges

    @property
    def naked_edges(self):
        """"Tuple of all naked edges in this Mesh3D as LineSegment3D objects.

        Naked edges belong to only one face in the mesh (they are not
        shared between faces).
        """
        if self._naked_edges is None:
            self._naked_edges = self._get_edge_type(0)
        return self._naked_edges

    @property
    def internal_edges(self):
        """"Tuple of all internal edges in this Mesh3D as LineSegment3D objects.

        Internal edges are shared between two faces in the mesh.
        """
        if self._internal_edges is None:
            self._internal_edges = self._get_edge_type(1)
        return self._internal_edges

    @property
    def non_manifold_edges(self):
        """"Tuple of all non-manifold edges in this mesh as LineSegment3D objects.

        Non-manifold edges are shared between three or more faces.
        """
        if self._non_manifold_edges is None:
            if self._edges is None:
                self.edges
            nm_edges = []
            for i, type in enumerate(self._edge_types):
                if type > 1:
                    nm_edges.append(self._edges[i])
            self._non_manifold_edges = tuple(nm_edges)
        return self._non_manifold_edges

    def remove_vertices(self, pattern):
        """Get a version of this mesh where vertices are removed according to a pattern.

        Args:
            pattern: A list of boolean values denoting whether a vertex should
                remain in the mesh (True) or be removed from the mesh (False).
                The length of this list must match the number of this mesh's vertices.

        Returns:
            A tuple with two elements.

            -   new_mesh:
                A mesh where the vertices have been removed according
                to the input pattern.

            -   face_pattern:
                A list of boolean values that corresponds to the
                original mesh faces noting whether the face is in the new mesh
                (True) or has been removed from the new mesh (False).
        """
        _new_verts, _new_faces, _new_colors, _new_f_cent, _new_f_area, face_pattern = \
            self._remove_vertices(pattern)

        new_mesh = Mesh3D(_new_verts, _new_faces, _new_colors)
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
            A tuple with two elements.

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

        new_mesh = Mesh3D(_new_verts, _new_faces, _new_colors)
        new_mesh._face_centroids = _new_f_cent
        new_mesh._face_areas = _new_f_area
        return new_mesh, vertex_pattern

    def remove_faces_only(self, pattern):
        """Get a version of this mesh where faces are removed and vertices are unaltered.

        This is faster than the Mesh3D.remove_faces method but will likely result
        a lower-quality mesh where several vertices exist in the mesh that are not
        referenced by any face. This may be preferred if pure speed of removing
        faces is a priority over smallest size of the mesh in memory.

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

        new_mesh = Mesh3D(self.vertices, _new_faces, _new_colors)
        new_mesh._face_centroids = _new_f_cent
        new_mesh._face_areas = _new_f_area
        return new_mesh

    def rotate(self, axis, angle, origin):
        """Rotate a mesh by a certain angle around an axis and origin.

        Right hand rule applies:
        If axis has a positive orientation, rotation will be clockwise.
        If axis has a negative orientation, rotation will be counterclockwise.

        Args:
            axis: A Vector3D axis representing the axis of rotation.
            angle: An angle for rotation in radians.
            origin: A Point3D for the origin around which the point will be rotated.
        """
        _verts = tuple(pt.rotate(axis, angle, origin) for pt in self.vertices)
        return self._mesh_transform(_verts)

    def rotate_xy(self, angle, origin):
        """Get a mesh rotated counterclockwise in the XY plane by a certain angle.

        Args:
            angle: An angle for rotation in radians.
            origin: A Point3D for the origin around which the point will be rotated.
        """
        _verts = tuple(pt.rotate_xy(angle, origin) for pt in self.vertices)
        return self._mesh_transform(_verts)

    def scale(self, factor, origin=None):
        """Scale a mesh by a factor from an origin point.

        Args:
            factor: A number representing how much the mesh should be scaled.
            origin: A Point representing the origin from which to scale.
                If None, it will be scaled from the World origin (0, 0, 0).
        """
        if origin is None:
            _verts = tuple(
                Point3D(pt.x * factor, pt.y * factor, pt.z * factor)
                for pt in self.vertices)
        else:
            _verts = tuple(pt.scale(factor, origin) for pt in self.vertices)
        return self._mesh_scale(_verts, factor)

    def offset_mesh(self, distance):
        """Get a Mesh3D that has been offset from this one by a certain difference.

        Effectively, this method moves each mesh vertex along the vertex normal
        by the offset distance.

        Args:
            distance: A number for the distance to offset the mesh.
        """
        new_verts = tuple(pt.move(norm * distance) for pt, norm in
                          zip(self.vertices, self.vertex_normals))
        return Mesh3D(new_verts, self.faces, self._colors)

    def height_field_mesh(self, values, domain):
        """Get a Mesh3D that has faces or vertices offset according to a list of values.

        Args:
            values: A list of values that has a length matching the number of faces
                or vertices in this mesh.
            domain: A tuple or list of two numbers for the upper and lower distances
                that the mesh vertices should be offset. (ie. (0, 3))
        """
        assert isinstance(domain, (tuple, list)), 'Expected tuple for domain. '\
            'Got {}.'.format(type(domain))
        assert len(domain) == 2, 'Expected domain to be in the format (min, max). ' \
            'Got {}.'.format(domain)

        if len(values) == len(self.faces):
            remap_vals = Mesh3D._remap_values(values, domain[0], domain[-1])
            vert_remap_vals = []
            for vf in self.vertex_connected_faces:
                v = 0
                for j in vf:
                    v += remap_vals[j]
                try:
                    v /= len(vf)  # average the vertex value over its connected faces
                except ZeroDivisionError:
                    pass  # lone vertex without any faces
                vert_remap_vals.append(v)
            new_verts = tuple(pt.move(norm * dist) for pt, norm, dist in
                              zip(self.vertices, self.vertex_normals, vert_remap_vals))
        elif len(values) == len(self.vertices):
            remap_vals = Mesh3D._remap_values(values, domain[0], domain[-1])
            new_verts = tuple(pt.move(norm * dist) for pt, norm, dist in
                              zip(self.vertices, self.vertex_normals, remap_vals))
        else:
            raise ValueError(
                'Input values for height_field_mesh ({}) does not match the number of'
                ' mesh faces ({}) nor the number of vertices ({}).'
                .format(len(values), len(self.faces), len(self.vertices)))
        return Mesh3D(new_verts, self.faces, self._colors)

    def to_dict(self):
        """Get Mesh3D as a dictionary."""
        base = {'type': 'Mesh3D',
                'vertices': [pt.to_array() for pt in self.vertices],
                'faces': self.faces}
        if self.colors is not None:
            base['colors'] = [col.to_dict() for col in self.colors]
        return base

    def to_stl(self, folder, name=None):
        """Write the Mesh3D to an ASCII STL file.

        Args:
            folder: A text string for the directory where the STL will be written.
            name: A text string for the name of the STL file.
        """
        from ladybug_geometry.interop.stl import STL  # avoid circular import
        stl_obj = STL(self.face_vertices, self.face_normals)
        return stl_obj.to_file(folder, name)

    def to_obj(self, folder, name, include_colors=True, include_normals=False,
               triangulate_quads=False, include_mtl=False):
        """Write the Mesh3D to an ASCII OBJ file.

        Args:
            folder: A text string for the directory where the OBJ will be written.
            name: A text string for the name of the OBJ file.
            include_colors: Boolean to note whether the Mesh3D colors should be
                included in the OBJ file. (Default: True).
            include_normals: Boolean to note whether the vertex normals should be
                included in the OBJ file. (Default: False).
            triangulate_quads: Boolean to note whether quad faces should be
                triangulated upon export to OBJ. This may be needed for certain
                software platforms that require the mesh to be composed entirely
                of triangles (eg. Radiance). (Default: False).
            include_mtl: Boolean to note whether an .mtl file should be automatically
                generated next to the .obj file in the output folder. All materials
                in the mtl file will be diffuse white, with the assumption that
                these will be customized later. (Default: False).
        """
        from ladybug_geometry.interop.obj import OBJ  # avoid circular import
        transl_obj = OBJ.from_mesh3d(self, include_colors, include_normals)
        return transl_obj.to_file(folder, name, triangulate_quads, include_mtl)

    @staticmethod
    def join_meshes(meshes):
        """Join an array of Mesh3Ds into a single Mesh3D.

        Args:
            meshes: An array of meshes to be joined into one.

        Returns:
            A single Mesh3D object derived from the input meshes.
        """
        # set up empty lists of objects to be filled
        verts = []
        faces = []
        colors = []

        # loop through all of the meshes and get new faces
        total_v_i = 0
        for mesh in meshes:
            verts.extend(mesh._vertices)
            for fc in mesh._faces:
                faces.append(tuple(v_i + total_v_i for v_i in fc))
            total_v_i += len(mesh._vertices)
            if mesh._colors:
                colors.extend(mesh._colors)

        # create the new mesh
        if len(colors) != 0:
            new_mesh = Mesh3D(verts, faces, colors)
        else:
            new_mesh = Mesh3D(verts, faces)
        return new_mesh

    def _calculate_min_max(self):
        """Calculate maximum and minimum Point3D for this object."""
        min_pt = [self.vertices[0].x, self.vertices[0].y, self.vertices[0].z]
        max_pt = [self.vertices[0].x, self.vertices[0].y, self.vertices[0].z]

        for v in self.vertices[1:]:
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

    def _calculate_face_areas_and_normals(self):
        """Calculate face areas and normals from vertices."""
        _f_norm = []
        _f_area = []
        for face in self.faces:
            pts = tuple(self._vertices[i] for i in face)
            if len(face) == 3:
                n, a = self._calculate_normal_and_area_for_triangle(pts)
            else:
                n, a = self._calculate_normal_and_area_for_quad(pts)
            _f_norm.append(n)
            _f_area.append(a)
        self._face_normals = tuple(_f_norm)
        self._face_areas = tuple(_f_area)

    def _calculate_vertex_normals(self):
        """Calculate vertex normals.

        This is accomplished by normalizing the average of the surface normals
        of the faces that contain that vertex.  This particular method weights
        this average by the area of each face, though this does not always need
        to be the case as noted here:
        https://en.wikipedia.org/wiki/Vertex_normal
        """
        # find shared faces for each vertices
        mapper = [[] for v in xrange(len(self.vertices))]
        for c, face in enumerate(self.faces):
            for i in face:
                mapper[i].append(c)
        # now calculate vertex normal based on face normals
        vn = []
        fn = self.face_normals
        fa = self.face_areas
        for fi in mapper:
            x, y, z = 0, 0, 0
            for n, a in zip(tuple(fn[i] for i in fi), tuple(fa[i] for i in fi)):
                x += n.x * a
                y += n.y * a
                z += n.z * a
            _v = Vector3D(x, y, z)
            vn.append(_v.normalize())
        self._vertex_normals = tuple(vn)

    def _get_edge_type(self, edge_type):
        """Get all of the edges of a certain type in this mesh."""
        if self._edges is None:
            self.edges
        sel_edges = []
        for i, type in enumerate(self._edge_types):
            if type == edge_type:
                sel_edges.append(self._edges[i])
        return tuple(sel_edges)

    def _tri_face_centroid(self, face):
        """Compute the centroid of a triangular face."""
        return Mesh3D._tri_centroid(tuple(self._vertices[i] for i in face))

    def _quad_face_centroid(self, face):
        """Compute the centroid of a quadrilateral face."""
        return Mesh3D._quad_centroid(tuple(self._vertices[i] for i in face))

    def _mesh_transform(self, verts):
        """Transform mesh in a way that transfers properties and avoids extra checks."""
        _new_mesh = Mesh3D(verts, self.faces)
        self._transfer_properties(_new_mesh)
        return _new_mesh

    def _mesh_transform_move(self, verts):
        """Move mesh in a way that transfers properties and avoids extra checks."""
        _new_mesh = Mesh3D(verts, self.faces)
        self._transfer_properties(_new_mesh)
        _new_mesh._face_normals = self._face_normals
        _new_mesh._vertex_normals = self._vertex_normals
        return _new_mesh

    def _mesh_scale(self, verts, factor):
        """Scale mesh in a way that transfers properties and avoids extra checks."""
        _new_mesh = Mesh3D(verts, self.faces)
        self._transfer_properties_scale(_new_mesh, factor)
        _new_mesh._face_normals = self._face_normals
        _new_mesh._vertex_normals = self._vertex_normals
        return _new_mesh

    def _check_vertices_input(self, vertices):
        """Check the input vertices."""
        if not isinstance(vertices, tuple):
            vertices = tuple(vertices)
        for vert in vertices:
            assert isinstance(vert, Point3D), \
                'Expected Point3D for {} vertex. Got {}.'.format(
                    self.__class__.__name__, type(vert))
        return vertices

    @staticmethod
    def _calculate_normal_and_area_for_triangle(pts):
        """Calculate normal and area for three points.

        Returns:
            n = Normalized normal vector for the triangle.
            a = Area of the triangle.
        """
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        n = v1.cross(v2)
        a = n.magnitude / 2
        return n.normalize(), a

    @staticmethod
    def _calculate_normal_and_area_for_quad(pts):
        """Calculate normal and area for four points.

        This method uses an area-weighted average of the two triangle normals
        that compose the quad face.

        Returns:
            n = Normalized normal vector for the quad.
            a = Area of the quad.
        """
        # TODO: Get this method to work for concave quads.
        # This method is only reliable when quads are convex since we assume
        # either diagonal of the quad splits it into two triangles.
        # It seems Rhino never produces concave quads when it automatically meshes
        # but we will likely want to add support for this if meshes have other origins
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        n1 = v1.cross(v2)

        v3 = pts[3] - pts[2]
        v4 = pts[1] - pts[2]
        n2 = v3.cross(v4)

        a = (n1.magnitude + n2.magnitude) / 2
        n = Vector3D((n1.x + n2.x) / 2, (n1.y + n2.y) / 2, (n1.z + n2.z) / 2)
        return n.normalize(), a

    @staticmethod
    def _face_center(verts):
        """Get the center of a list of Point3D vertices."""
        _cent_x = sum([v.x for v in verts])
        _cent_y = sum([v.y for v in verts])
        _cent_z = sum([v.z for v in verts])
        v_count = len(verts)
        return Point3D(_cent_x / v_count, _cent_y / v_count, _cent_z / v_count)

    @staticmethod
    def _tri_centroid(verts):
        """Get the centroid of a list of 3 Point3D vertices."""
        _cent_x = sum([v.x for v in verts])
        _cent_y = sum([v.y for v in verts])
        _cent_z = sum([v.z for v in verts])
        return Point3D(_cent_x / 3, _cent_y / 3, _cent_z / 3)

    @staticmethod
    def _quad_centroid(verts):
        """Get the centroid of a list of 4 Point3D vertices."""
        # TODO: Get this method to recognize concave quads.
        # This method is only reliable when quads are convex since we assume
        # either diagonal of the quad splits it into two triangles.
        # It seems Rhino never produces concave quads when it automatically meshes
        _tri_verts = ((verts[0], verts[1], verts[2]), (verts[2], verts[3], verts[0]))
        _tri_c = [Mesh3D._tri_centroid(tri) for tri in _tri_verts]
        _tri_a = [Mesh3D._get_tri_area(tri) for tri in _tri_verts]
        _tot_a = sum(_tri_a)
        try:
            _cent_x = (_tri_c[0].x * _tri_a[0] + _tri_c[1].x * _tri_a[1]) / _tot_a
            _cent_y = (_tri_c[0].y * _tri_a[0] + _tri_c[1].y * _tri_a[1]) / _tot_a
            _cent_z = (_tri_c[0].z * _tri_a[0] + _tri_c[1].z * _tri_a[1]) / _tot_a
        except ZeroDivisionError:
            _cent_x = sum([v.x for v in verts]) / 4
            _cent_y = sum([v.y for v in verts]) / 4
            _cent_z = sum([v.z for v in verts]) / 4
        return Point3D(_cent_x, _cent_y, _cent_z)

    @staticmethod
    def _get_tri_area(pts):
        """Get the area of a triangle from three Point3D objects."""
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        n1 = v1.cross(v2)
        return n1.magnitude / 2

    @staticmethod
    def _remap_values(values, tmin, tmax):
        """Remap a set of values to offset distances within a domain."""
        omin = min(values)
        omax = max(values)
        odiff = omax - omin
        tdiff = tmax - tmin
        if odiff == 0:
            return [tmin] * len(values)
        else:
            return [(v - omin) * tdiff / odiff + tmin for v in values]

    def __copy__(self):
        _new_mesh = Mesh3D(self.vertices, self.faces)
        self._transfer_properties(_new_mesh)
        _new_mesh._face_centroids = self._face_centroids
        _new_mesh._face_normals = self._face_normals
        _new_mesh._vertex_normals = self._vertex_normals
        return _new_mesh

    def __key(self):
        """A tuple based on the object properties, useful for hashing."""
        return tuple(hash(pt) for pt in self._vertices) + \
            tuple(hash(face) for face in self._faces)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(other, Mesh3D) and self.__key() == other.__key()

    def __repr__(self):
        return 'Mesh3D ({} faces) ({} vertices)'.format(
            len(self.faces), len(self))
