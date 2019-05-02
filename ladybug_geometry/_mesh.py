# coding=utf-8
"""Base class for all Mesh objects."""
from __future__ import division


class MeshBase(object):
    """Base class for all Mesh objects.

    Properties:
        vertices
        faces
        colors
        is_color_by_face
        face_areas
        area
        face_centroids
    """

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
            if len(col) == len(self.faces):
                self._is_color_by_face = True
            elif len(col) == len(self.vertices):
                self._is_color_by_face = False
            else:
                raise ValueError('Number of colors ({}) does not match the number of'
                                 ' mesh faces ({}) nor the number of vertices ({}).'
                                 .format(len(col), len(self.faces), len(self.vertices)))
            if isinstance(col, list):
                col = tuple(col)
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

    @is_color_by_face.setter
    def is_color_by_face(self, by_face):
        assert isinstance(by_face, bool), \
            'is_color_by_face must be a boolean. Got {}'.format(type(by_face))
        if self._colors is not None and self._is_color_by_face is not by_face:
            raise AttributeError('is_color_by_face cannot be set when colors are'
                                 ' already assinged to the mesh.')
        self._is_color_by_face = by_face

    @property
    def area(self):
        """The area of the entire mesh."""
        if self._area is None:
            self._area = sum(self.face_areas)
        return self._area

    @property
    def face_centroids(self):
        """A tuple of face centroids that parallels the Faces property."""
        if self._face_centroids is None:
            _f_cent = []
            for face in self.faces:
                if len(face) == 3:
                    _f_cent.append(self._tri_face_centroid(face))
                else:
                    _f_cent.append(self._quad_face_centroid(face))
            self._face_centroids = tuple(_f_cent)
        return self._face_centroids

    def _check_faces_input(self, faces):
        """Check input faces for correct formatting."""
        assert isinstance(faces, (list, tuple)), \
            'faces should be a list or tuple. Got {}'.format(type(faces))
        assert len(faces) > 0, 'Mesh must have at least one face.'
        for f in faces:
            assert isinstance(f, tuple), \
                'Expected tuple for Mesh face. Got {}.'.format(type(f))
            assert len(f) == 3 or len(f) == 4, \
                'Mesh face can only have 3 or 4 vertices. Got {}.'.format(len(f))
        assert isinstance(faces[0][0], int), 'Mesh face must use integers to ' \
            'reference vertices. Got {}.'.format(type(faces[0][0]))
        if isinstance(faces, list):
            faces = tuple(faces)
        self._faces = faces

    def _check_face_pattern(self, pattern):
        """Check any input face pattern for compatability with this mesh."""
        assert isinstance(pattern, (list, tuple)), 'pattern for remove_faces must' \
            ' be a list or tuple. Got {}.'.format(type(pattern))
        assert len(pattern) == len(self.faces), 'Length of pattern for remove_faces'\
            ' ({}) must match the number of faces in the mesh ({}).'.format(
                len(pattern), len(self.faces))

    def _remove_vertices(self, pattern):
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
        face_pattern = []
        for _f in self.faces:
            try:
                if len(_f) == 3:
                    _new_f = (_vdict[_f[0]], _vdict[_f[1]], _vdict[_f[2]])
                else:
                    _new_f = (_vdict[_f[0]], _vdict[_f[1]], _vdict[_f[2]], _vdict[_f[3]])
                    _new_faces.append(_new_f)
                    face_pattern.append(True)
            except KeyError:
                face_pattern.append(False)

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
            _f_cent = []
            for i, _p in enumerate(face_pattern):
                if _p is True:
                    _f_cent.append(self._face_centroids[i])
            _new_f_cent = tuple(_f_cent)
        _new_f_area = None
        if self._face_areas is not None:
            if isinstance(self._face_areas, (float, int)):
                _new_f_area = self._face_areas
            else:
                _f_area = []
                for i, _p in enumerate(face_pattern):
                    if _p is True:
                        _f_area.append(self._face_areas[i])
                _new_f_area = tuple(_f_area)
        return _new_f_cent, _new_f_area

    def _vertex_pattern_from_remove_faces(self, pattern):
        """Get a pattern of vertices to remove from a pattern of faces to remove."""
        self._check_face_pattern(pattern)
        vertex_pattern = [False for _v in self.vertices]
        for i, _in_mesh in enumerate(pattern):
            if _in_mesh is True:
                _face = self._faces[i]
                for _v in _face:
                    vertex_pattern[i] = True
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
            new_mesh._face_areas = tuple([a * factor for a in self._face_areas])
        if self._area is not None:
            new_mesh._area = tuple([a * factor for a in self._area])

    @staticmethod
    def _interpret_input_from_faces(faces, purge):
        """Get faces anv vertices from a list of faces as points."""
        vertices = []  # collection of vertices as point objects
        face_collector = []  # collection of face indices
        if purge:
            for f in faces:
                ind = []
                for v in f:
                    try:  # this can get very slow for large number of vertices.
                        index = vertices.index(v)
                        ind.append(index)
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
