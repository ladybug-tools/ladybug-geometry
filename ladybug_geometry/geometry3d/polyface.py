# coding=utf-8
"""Object with Multiple Planar Faces in 3D Space"""
from __future__ import division

from .pointvector import Vector3D, Point3D
from .ray import Ray3D
from .line import LineSegment3D
from .plane import Plane
from .face import Face3D
from ._2d import Base2DIn3D


class Polyface3D(Base2DIn3D):
    """Object with Multiple Planar Faces in 3D Space. Includes solid objects and polyhedra.

    Properties:
        vertices
        faces
        edges
        naked_edges
        internal_edges
        non_manifold_edges
        face_indices
        edge_indices
        edge_types
        min
        max
        center
        area
        is_solid
    """
    __slots__ = ('_vertices', '_faces', '_edges',
                 '_naked_edges', '_internal_edges', '_non_manifold_edges',
                 '_face_indices', '_edge_indices', '_edge_types'
                 '_min', '_max', '_center', '_area', '_is_solid')

    def __init__(self, vertices, face_indices, edge_information=None):
        """Initilize Polyface3D.

        Args:
            vertices: A list of Point3D objects representing the vertices of
                this PolyFace.
            face_indices: A list of tuples that contain integers corresponding
                to indices within the vertices list. Each tuple represents a
                face of the polyface.
            edge_information: Optional edge information, which will speed up the
                creation of the Polyface object if it is available. If None, this
                will be computed from the vertices and face_indices. Edge information
                should be formatted as a dictionary with two keys as follows:
                'edge_indices': A list of tuple objects that each contain two integers.
                    These integers correspond to indices within the vertices list and
                    each tuple represents a line sengment for an edge of the polyface.
                'edge_types': A list of integers for each edge that parallels
                    the edge_indices list. An integer of 0 denotes a naked edge, an
                    integer of 1 denotes an internal edge. Anything higher is a
                    non-manifold edge.
        """
        # assign input properties
        self._check_vertices_input(vertices)
        self._face_indices = face_indices if isinstance(face_indices, tuple) \
            else tuple(face_indices)

        # unpack or autocalculate edge information
        if edge_information is not None:
            edge_i = edge_information['edge_indices']
            edge_t = edge_information['edge_types']
        else:  # determine unique edges from the vertices and faces.
            edge_i = []
            edge_t = []
            for fi in face_indices:
                for i, vi in enumerate(fi):
                    try:  # this can get slow for large number of vertices.
                        ind = edge_i.index((fi[i - 1], vi))
                        edge_t[ind] += 1
                    except ValueError:  # make sure reversed edge isn't there
                        try:
                            ind = edge_i.index((vi, fi[i - 1]))
                            edge_t[ind] += 1
                        except ValueError:  # add a new edge
                            edge_i.append((fi[i - 1], vi))
                            edge_t.append(0)
        self._edge_indices = edge_i if isinstance(edge_i, tuple) else tuple(edge_i)
        self._edge_types = edge_t if isinstance(edge_t, tuple) else tuple(edge_t)

        # determine solidity of the polyface using the euler characteristic
        if len(self._vertices) - len(self._edge_indices) + len(self._face_indices) == 2:
            self._is_solid = True
        else:
            self._is_solid = False

        # assign default properties
        self._faces = None
        self._edges = None
        self._naked_edges = None
        self._internal_edges = None
        self._non_manifold_edges = None
        self._min = None
        self._max = None
        self._center = None
        self._area = None

    @classmethod
    def from_faces(cls, faces):
        """Initilize Polyface3D from a list of Face3D objects.

        Initializing a Polyface3D this way will preserve the properties of the
        underlying Face3D objects, including the order of Face3D objects on the faces
        property and the presence of holes in the Face3D objects. As such,
        it is the recommended way to create a polyface when Face3D objects are
        available.

        Args:
            faces: A list of Face3D objects representing the boundary of this Polyface.
        """
        # extract unique vertices from the faces
        vertices = []  # collection of vertices as point objects
        face_indices = []  # collection of face indices
        for f in faces:
            ind = []
            for v in f:
                try:  # this can get slow for large number of vertices.
                    ind.append(vertices.index(v))
                except ValueError:  # add new point
                    vertices.append(v)
                    ind.append(len(vertices) - 1)
            face_indices.append(tuple(ind))

        # get the polyface object and assign correct faces to it
        _polyface = cls(vertices, face_indices)
        if _polyface._is_solid:
            _polyface._faces = cls._correct_face_direction(faces)
        else:
            _polyface._faces = faces
        return _polyface

    @classmethod
    def from_box(cls, length, width, height, base_plane=None):
        """Initilize Polyface3D from parameters describing a box.

        Initializing a polyface this way has the added benefit of having its
        faces property quickly calculated.

        Args:
            length: A number for the length of the box (in the X direction).
            width: A number for the width of the box (in the Y direction).
            height: A number for the height of the box (in the Z direction).
            base_plane: A Plane object from which to generate the box.
                If None, default is the WorldXY plane.
        """
        assert isinstance(length, (float, int)), 'Box length must be a number.'
        assert isinstance(width, (float, int)), 'Box width must be a number.'
        assert isinstance(height, (float, int)), 'Box height must be a number.'
        if base_plane is not None:
            assert isinstance(base_plane, Plane), \
                'base_plane must be Plane. Got {}.'.format(type(base_plane))
        else:
            base_plane = Plane(Vector3D(0, 0, 1), Point3D())
        _o = base_plane.o
        _l_vec = base_plane.x * length
        _w_vec = base_plane.y * width
        _h_vec = base_plane.n * height
        _verts = (_o, _o + _w_vec, _o + _w_vec + _l_vec, _o + _l_vec,
                  _o + _h_vec, _o + _w_vec + _h_vec,
                  _o + _w_vec + _l_vec + _h_vec, _o + _l_vec + _h_vec)
        _face_indices = ((0, 1, 2, 3), (0, 4, 5, 1), (0, 3, 7, 4),
                         (2, 1, 5, 6), (6, 7, 3, 2), (7, 6, 5, 4))
        _edge_indices = ((3, 0), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5),
                         (5, 1), (3, 7), (7, 4), (6, 2), (5, 6), (6, 7))
        polyface = cls(_verts, _face_indices, {'edge_indices': _edge_indices,
                                               'edge_types': [1] * 12})
        verts = tuple(tuple(_verts[i] for i in face) for face in _face_indices)
        polyface._faces = tuple(Face3D.from_vertices(v) for v in verts)
        return polyface

    @property
    def vertices(self):
        """Tuple of all vertices in this polyface.

        Note that, in the case of a polyface with holes, some vertices will be repeated
        since this property effectively traces out a single boundary around the
        whole shape, winding inward to cut out the holes.
        """
        return self._vertices

    @property
    def faces(self):
        """Tuple of all Face3D objects making up this polyface."""
        if self._faces is None:
            verts = tuple(tuple(self.vertices[i] for i in face)
                          for face in self._face_indices)
            faces = tuple(Face3D.from_vertices(v) for v in verts)
            if self._is_solid:
                faces = Polyface3D._correct_face_direction(faces)
            self._faces = faces
        return self._faces

    @property
    def edges(self):
        """"Tuple of all edges in this polyface as LineSegment3D objects."""
        if self._edges is None:
            self._edges = tuple(LineSegment3D.from_end_points(
                self.vertices[seg[0]], self.vertices[seg[1]])
                                for seg in self._edge_indices)
        return self._edges

    @property
    def naked_edges(self):
        """"Tuple of all naked edges in this polyface as LineSegment3D objects.

        Naked edges belong to only one face in the polyface (they are not
        shared between faces).
        """
        if self._naked_edges is None:
            self._naked_edges = self._get_edge_type(0)
        return self._naked_edges

    @property
    def internal_edges(self):
        """"Tuple of all internal edges in this polyface as LineSegment3D objects.

        Internal edges are shared between two faces in the polyface.
        """
        if self._internal_edges is None:
            self._internal_edges = self._get_edge_type(1)
        return self._internal_edges

    @property
    def non_manifold_edges(self):
        """"Tuple of all non-manifold edges in this polyface as LineSegment3D objects.

        Non-manifold edges are shared between three or more faces and are therefore
        not allowed in solid polyfaces.
        """
        if self._non_manifold_edges is None:
            if self._edges is None:
                self.edges
            nm_edges = []
            for i, type in enumerate(self._edge_types):
                if type > 2:
                    nm_edges.append(self._edges[i])
            self._non_manifold_edges = tuple(nm_edges)
        return self._non_manifold_edges

    @property
    def face_indices(self):
        """Tuple of face tuples with integers corresponding to indices of vertices."""
        return self._face_indices

    @property
    def edge_indices(self):
        """Tuple of edge tuples with integers corresponding to indices of vertices."""
        return self._edge_indices

    @property
    def edge_types(self):
        """Tuple of integers for each edge that denotes the type of edge.

        0 denotes a naked edge, 1 denotes an internal edge, and anything higher is a
        non-manifold edge.
        """
        return self._edge_types

    @property
    def area(self):
        """The total surface area of the polyface."""
        if self._area is None:
            self._area = sum([face.area for face in self.faces])
        return self._area

    @property
    def is_solid(self):
        """A boolean to note whether the polyface is solid (True) or is open (False).

        Note that all solid polyface objects will have faces pointing outwards.
        """
        return self._is_solid

    def is_point_inside(self, point, test_vector=Vector3D(1, 0, 0)):
        """Test whether a Point3D lies inside or outside the polyface.

        Note that, if this polyface is not solid, the result will always be False.

        Args:
            point: A Point3D for which the inside/outside relationship will be tested.
            test_vector: Optional vector to set the direction in which intersections
                with the polyface faces will be evaluated to determine if the
                point is inside. Default is the X-unit vector.

        Returns:
            A boolean denoting whether the point lies inside (True) or outside (False).
        """
        if not self.is_solid:
            return False
        test_ray = Ray3D(point, test_vector)
        n_int = 0
        for _f in self.faces:
            if _f.intersect_line_ray(test_ray):
                n_int += 1
        if n_int % 2 == 0:
            return False
        return True

    def _get_edge_type(self, edge_type):
        """Get all of the edges of a certain type in this polyface."""
        if self._edges is None:
            self.edges
        sel_edges = []
        for i, type in enumerate(self._edge_types):
            if type == edge_type:
                sel_edges.append(self._edges[i])
        return tuple(sel_edges)

    @staticmethod
    def _correct_face_direction(faces):
        """Correct the direction that Face3D are pointing when the polyface is solid."""
        final_faces = []
        for i, face in enumerate(faces):
            # construct a ray with the face normal and a point on the face
            move_vec = (face.center - face[0]) * 0.000001
            point_on_face = face[0] + move_vec
            test_ray = Ray3D(point_on_face, face.normal)

            # if the ray intersects with an even number of other faces, it is correct
            n_int = 0
            for _f in faces[i + 1:]:
                if _f.intersect_line_ray(test_ray):
                    n_int += 1
            for _f in faces[:i]:
                if _f.intersect_line_ray(test_ray):
                    n_int += 1
            if n_int % 2 == 0:
                final_faces.append(face)
            else:
                final_faces.append(face.flip())
        return tuple(final_faces)
