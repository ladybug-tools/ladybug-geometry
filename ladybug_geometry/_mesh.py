# coding=utf-8
"""Base class for all Mesh objects."""
from __future__ import division

import sys
if (sys.version_info > (3, 0)):  # python 3
    xrange = range


class MeshBase(object):
    """Base class for all Mesh objects.

    Args:
        vertices: A list or tuple of Point objects for vertices.
        faces: A list of tuples with each tuple having either 3 or 4 integers.
            These integers correspond to indices within the list of vertices.
        colors: An optional list of colors that correspond to either the faces
            of the mesh or the vertices of the mesh. Default is None.

    Properties:
        * vertices
        * faces
        * colors
        * is_color_by_face
        * face_areas
        * area
        * face_centroids
        * vertex_connected_faces
    """
    __slots__ = ('_vertices', '_faces', '_colors', '_is_color_by_face',
                 '_area', '_face_areas', '_face_centroids', '_vertex_connected_faces')

    def __init__(self, vertices, faces, colors=None):
        """Initialize MeshBase."""
        self._vertices = vertices
        self._faces = self._check_faces_input(faces)
        self._is_color_by_face = False  # default if colors is None
        self.colors = colors
        self._area = None
        self._face_areas = None
        self._face_centroids = None
        self._vertex_connected_faces = None

    @property
    def vertices(self):
        """Tuple of all vertices in this geometry."""
        return self._vertices

    @property
    def faces(self):
        """Tuple of all faces in the mesh."""
        return self._faces

    @property
    def colors(self):
        """Get or set a list of colors for the mesh. Will be None if no colors assigned.
        """
        return self._colors

    @colors.setter
    def colors(self, col):
        if col is not None:
            assert isinstance(col, (list, tuple)), \
                'colors should be a list or tuple. Got {}'.format(type(col))
            if isinstance(col, list):
                col = tuple(col)
            if len(col) == len(self.faces):
                self._is_color_by_face = True
            elif len(col) == len(self.vertices):
                self._is_color_by_face = False
            elif len(col) == 0:
                col = None
            else:
                raise ValueError('Number of colors ({}) does not match the number of'
                                 ' mesh faces ({}) nor the number of vertices ({}).'
                                 .format(len(col), len(self.faces), len(self.vertices)))
        self._colors = col

    @property
    def is_color_by_face(self):
        """Boolean for whether colors are face-by-face (True) or vertex-by-vertex (False)

        When colors are assigned to the mesh, this property is linked to the colors
        and cannot be set. However, when no colors are assigned, it can be set
        because it affects how the mesh is translated to external interfaces.
        In such cases, this property notes whether the mesh will be translated
        in a format that can accept face-by-face colors.
        Default is False when no colors are assigned.
        """
        return self._is_color_by_face

    @property
    def area(self):
        """The area of the entire mesh."""
        if self._area is None:
            self._area = sum(self.face_areas)
        return self._area

    @property
    def face_centroids(self):
        """Tuple of face centroids that parallels the Faces property."""
        if self._face_centroids is None:
            _f_cent = []
            for face in self.faces:
                if len(face) == 3:
                    _f_cent.append(self._tri_face_centroid(face))
                else:
                    _f_cent.append(self._quad_face_centroid(face))
            self._face_centroids = tuple(_f_cent)
        return self._face_centroids

    @property
    def vertex_connected_faces(self):
        """Tuple with a tuple for each vertex that lists the indexes of connected faces.
        """
        if self._vertex_connected_faces is None:
            _vert_faces = [[] for i in xrange(len(self._vertices))]
            for i, face in enumerate(self._faces):
                for j in face:
                    _vert_faces[j].append(i)
            self._vertex_connected_faces = tuple(tuple(vert) for vert in _vert_faces)
        return self._vertex_connected_faces

    def move(self, moving_vec):
        """Get a mesh that has been moved along a vector.

        Args:
            moving_vec: A Vector with the direction and distance to move the mesh.
        """
        _verts = tuple(pt.move(moving_vec) for pt in self.vertices)
        return self._mesh_transform(_verts)

    def reflect(self, normal, origin):
        """Get a mesh reflected across a plane with the input normal vector and origin.

        Args:
            normal: A Vector representing the normal vector for the plane across
                which the mesh will be reflected. THIS VECTOR MUST BE NORMALIZED.
            origin: A Point representing the origin from which to reflect.
        """
        _verts = tuple(pt.reflect(normal, origin) for pt in self.vertices)
        return self._mesh_transform(_verts)

    def duplicate(self):
        """Get a copy of this object."""
        return self.__copy__()

    def _check_faces_input(self, faces):
        """Check input faces for correct formatting."""
        if not isinstance(faces, tuple):
            faces = tuple(faces)
        assert len(faces) > 0, 'Mesh must have at least one face.'
        for f in faces:
            assert isinstance(f, tuple), \
                'Expected tuple for Mesh face. Got {}.'.format(type(f))
            assert len(f) == 3 or len(f) == 4, \
                'Mesh face can only have 3 or 4 vertices. Got {}.'.format(len(f))
            for ind in f:
                try:
                    self._vertices[ind]
                except IndexError:
                    raise IndexError(
                        'mesh face index {} does not correspond to any vertex. There '
                        'are {} vertices in the mesh.'.format(ind, len(self._vertices)))
                except TypeError:
                    raise TypeError(
                        'Mesh face must use integers to reference vertices. '
                        'Got {}.'.format(type(ind)))
        return faces

    def _check_face_pattern(self, pattern):
        """Check input pattern for remove faces for compatibility with this mesh."""
        assert isinstance(pattern, (list, tuple)), 'pattern for remove_faces must' \
            ' be a list or tuple. Got {}.'.format(type(pattern))
        assert len(pattern) == len(self.faces), 'Length of pattern for remove_faces'\
            ' ({}) must match the number of faces in the mesh ({}).'.format(
                len(pattern), len(self.faces))

    def _remove_vertices(self, pattern, face_pattern=None):
        """Get new faces, colors, centroids, and areas when removing vertices."""
        assert isinstance(pattern, (list, tuple)), 'pattern for remove_vertices must' \
            ' be a list or tuple. Got {}.'.format(type(pattern))
        assert len(pattern) == len(self), 'Length of pattern for remove_vertices ({})' \
            ' must match the number of vertices in the mesh ({}).'.format(
                len(pattern), len(self))
        _new_verts = []
        _new_faces = []
        # make a dictionary that maps old vertex indices to new ones
        _vdict = {}
        _vcount = 0
        for i, _in_mesh in enumerate(pattern):
            if _in_mesh is True:
                _vdict[i] = _vcount
                _vcount += 1
                _new_verts.append(self._vertices[i])

        # get the new faces
        if face_pattern is None:
            face_pattern = []
            for _f in self.faces:
                try:
                    if len(_f) == 3:
                        _new_f = (_vdict[_f[0]], _vdict[_f[1]], _vdict[_f[2]])
                    else:
                        _new_f = (_vdict[_f[0]], _vdict[_f[1]], _vdict[_f[2]],
                                  _vdict[_f[3]])
                    _new_faces.append(_new_f)
                    face_pattern.append(True)
                except KeyError:
                    face_pattern.append(False)
        else:
            for i, _f in enumerate(self.faces):
                if face_pattern[i] is True:
                    if len(_f) == 3:
                        _new_f = (_vdict[_f[0]], _vdict[_f[1]], _vdict[_f[2]])
                    else:
                        _new_f = (_vdict[_f[0]], _vdict[_f[1]], _vdict[_f[2]],
                                  _vdict[_f[3]])
                    _new_faces.append(_new_f)

        # remove colors if they are assigned
        _new_colors = None
        if self._colors is not None:
            _new_col = []
            if self._is_color_by_face is True:
                for i, _p in enumerate(face_pattern):
                    if _p is True:
                        _new_col.append(self._colors[i])
            else:
                for i, _p in enumerate(pattern):
                    if _p is True:
                        _new_col.append(self._colors[i])
            _new_colors = tuple(_new_col)

        # transfer face centroids and areas if they exist so they don't get re-computed
        _new_f_cent, _new_f_area = self._transfer_face_centroids_areas(face_pattern)

        return tuple(_new_verts), tuple(_new_faces), _new_colors, \
            _new_f_cent, _new_f_area, face_pattern

    def _transfer_face_centroids_areas(self, face_pattern):
        """Get face centroids and areas when removing faces.

        This is so they can be transferred to the new mesh.
        """
        _new_f_cent = None
        if self._face_centroids is not None:
            _new_f_cent = tuple(
                self._face_centroids[i] for i, _p in enumerate(face_pattern) if _p)
        _new_f_area = None
        if self._face_areas is not None:
            if isinstance(self._face_areas, (float, int)):
                _new_f_area = self._face_areas
            else:
                _new_f_area = tuple(
                    self._face_areas[i] for i, _p in enumerate(face_pattern) if _p)
        return _new_f_cent, _new_f_area

    def _vertex_pattern_from_remove_faces(self, pattern):
        """Get a pattern of vertices to remove from a pattern of faces to remove."""
        self._check_face_pattern(pattern)
        vertex_pattern = [False for _v in self.vertices]
        for i, _in_mesh in enumerate(pattern):
            if _in_mesh is True:
                _face = self._faces[i]
                for j in _face:
                    vertex_pattern[j] = True
        return vertex_pattern

    def _remove_faces_only(self, pattern):
        """Get new faces, colors, centroids, and areas when removing faces only."""
        self._check_face_pattern(pattern)
        _new_faces = []
        for i, _in_mesh in enumerate(pattern):
            if _in_mesh is True:
                _new_faces.append(self._faces[i])

        # transfer the colors over to the new mesh if they are assigned face-by-face
        _new_colors = self._colors
        if self._colors is not None and self._is_color_by_face is True:
            _new_col = []
            for i, _p in enumerate(pattern):
                if _p is True:
                    _new_col.append(self._colors[i])
            _new_colors = _new_col

        # transfer face centroids and areas if they exist so they don't get re-computed
        _new_f_cent, _new_f_area = self._transfer_face_centroids_areas(pattern)

        return tuple(_new_faces), _new_colors, _new_f_cent, _new_f_area

    def _transfer_properties(self, new_mesh):
        """Transfer properties when making a copy of the mesh or doing transforms."""
        new_mesh._colors = self._colors
        new_mesh._is_color_by_face = self._is_color_by_face
        new_mesh._face_areas = self._face_areas
        new_mesh._area = self._area

    def _transfer_properties_scale(self, new_mesh, factor):
        """Transfer properties when performing a scale operation."""
        new_mesh._colors = self._colors
        new_mesh._is_color_by_face = self._is_color_by_face
        if self._face_areas is not None:
            new_mesh._face_areas = tuple(a * factor for a in self._face_areas)
        if self._area is not None:
            new_mesh._area = self._area * factor

    @staticmethod
    def _interpret_input_from_face_vertices(faces, purge):
        """Get faces and vertices from a list of faces as points."""
        vertices = []  # collection of vertices as point objects
        face_collector = []  # collection of face indices
        if purge:
            for f in faces:
                ind = []
                for v in f:
                    try:  # this can get very slow for large number of vertices.
                        ind.append(vertices.index(v))
                    except ValueError:  # add new point
                        vertices.append(v)
                        ind.append(len(vertices) - 1)
                face_collector.append(tuple(ind))
        else:
            ver_counter = 0
            for f in faces:
                ind = []
                for v in f:
                    vertices.append(v)
                    ind.append(ver_counter)
                    ver_counter += 1
                face_collector.append(tuple(ind))
        return vertices, face_collector

    def __len__(self):
        return len(self._vertices)

    def __getitem__(self, key):
        return self._vertices[key]

    def __iter__(self):
        return iter(self._vertices)

    def __copy__(self):
        return MeshBase(self._vertices, self._faces, self._colors)

    def __key(self):
        """A tuple based on the object properties, useful for hashing."""
        return tuple(hash(pt) for pt in self._vertices) + \
            tuple(hash(face) for face in self._faces)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(other, MeshBase) and self.__key() == other.__key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __repr__(self):
        """Base MEsh representation."""
        return 'Base Mesh Object'
