# coding=utf-8
"""2D Mesh and 2D Mesh Face"""
from __future__ import division

from .pointvector import Point2D, Point2DImmutable
from .line import LineSegment2D
from .polygon import Polygon2D
from ._2d import Base2D

try:
    from itertools import izip as zip  # python 2
except ImportError:
    xrange = range  # python 3


class Mesh2D(Base2D):
    """2D Mesh object.

    Properties:
        vertices
        faces
        colors
        color_by_face
        face_areas
        area
        face_centroids
        centroid
        min
        max
        center
    """
    _check_required = True
    _colors = None
    _color_by_face = None
    _face_areas = None
    _area = None
    _face_centroids = None
    _centroid = None

    def __init__(self, vertices, faces, colors=None):
        """Initilize Mesh2D.

        Args:
            vertices: A list or tuple of Point2D objects for vertices.
            faces: A list of tuples with each tuple having either 3 or 4 integers.
                These integers correspond to indices within the list of vertices.
            colors: An optional list of colors that correspond to either the faces
                of the mesh or the vertices of the mesh. Default is None.
        """
        if self._check_required:
            self._check_vertices_input(vertices)
            self._check_faces_input(faces)
        else:
            self._vertices = vertices
            self._faces = faces
        self.colors = colors

    @classmethod
    def from_faces(cls, faces, purge=True):
        """Create a mesh from a list of faces each defined by Point2D objects.

        Args:
            faces: A list of faces with each face defined as a list of 3 or 4 Point2D.
            purge: A boolean to indicate if duplicate vertices should be shared between
                faces. Default is True to purge duplicate vertices, which can be slow
                for large lists of faces but results in a higher-quality mesh.
                Default is True.
        """
        vertices = []  # collection of vertices as Point2D
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
                face_collector.append(ind)
        else:
            ver_counter = 0
            for f in faces:
                ind = []
                for v in f:
                    vertices.append(v)
                    ind.append(ver_counter)
                    ver_counter += 1
                face_collector.append(ind)
        return cls(tuple(vertices), tuple(face_collector))

    @classmethod
    def from_polygon_triangulated(cls, polygon, purge=True):
        """Initialize a triangulated Mesh2D from a Polygon2D.

        Args:
            polygon: A Polygon2D.
            purge: A boolean to indicate if duplicate vertices should be shared between
                faces. This has no bearing on the triagnulation of convex polygons
                and only affects concave polygons with more than 4 sides.
                Default is True to purge duplicate vertices, which can be slow
                for large lists of faces but results in a higher-quality mesh.
                Default is True.
        """
        assert isinstance(polygon, Polygon2D), 'polygon must be a Polygon2D to use ' \
            'from_polygon_triangulated. Got {}.'.format(type(polygon))
        cls._check_required = False  # Turn off checks since we know the mesh is valid

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
            _new_mesh = cls.from_faces(_faces, purge)

        cls._check_required = True  # Turn the checks back on
        return _new_mesh

    @classmethod
    def from_polygon_grid(cls, polygon, x_dim, y_dim):
        """Initialize a gridded Mesh2D from a Polygon2D.

        Note that this gridded mesh will not completely fill the polygon unless it
        is perfectly rectangular. Essentially this method generates a grid over
        the domain of the polygon and then removes any points that do not lie within
        the polygon."""

    @classmethod
    def from_grid(cls, num_x, num_y, x_dim, y_dim):
        """Initialize a Mesh2D from parameters that define a grid."""

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
        """Tuple of all colors in the mesh or None if no colors assigned."""
        return self._colors

    @colors.setter
    def colors(self, col):
        if col is not None:
            assert isinstance(col, (list, tuple)), \
                'colors should be a list or tuple. Got {}'.format(type(col))
            if len(col) == len(self.faces):
                self._color_by_face = True
            elif len(col) == len(self.vertices):
                self._color_by_face = False
            else:
                raise ValueError('Number of colors ({}) does not match the number of'
                                 ' mesh faces ({}) nor the number of vertices ({}).'
                                 .format(len(col), len(self.faces), len(self.vertices)))
            self._colors = col

    @property
    def color_by_face(self):
        """Boolean for whether colors are face-by-face (True) or vertex-by-vertex (False)

        Will be None if no colors are assigned.
        """
        return self._color_by_face

    @property
    def face_areas(self):
        """A tuple of face areas that parallels the Mesh2D.Faces property."""
        if self._face_areas is None:
            _f_areas = []
            for face in self.faces:
                _f_areas.append(self._face_area(face))
            self._face_areas = tuple(_f_areas)
        return self._face_areas

    @property
    def area(self):
        """The area of the entire mesh."""
        if self._area is None:
            self._area = sum(self.face_areas)
        return self._area

    @property
    def face_centroids(self):
        """A tuple of face centroids that parallels the Mesh2D.Faces property."""
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
            self._centroid = Point2DImmutable(_weight_x / self.area,
                                              _weight_y / self.area)
        return self._centroid

    def triangulated(self):
        """Get a version of this Mesh2D where all quads have been triangulated."""
        _new_faces = []
        for face in enumerate(self.faces):
            if len(face) == 3:
                _new_faces.append(face)
            else:
                _triangles = Mesh2D._quad_to_triangles([self._vertices[i] for i in face])
                _new_faces.extend(_triangles)
        _new_faces = tuple(_new_faces)

        _new_colors = self.colors
        if self.color_by_face is True:
            _new_colors = []
            for i, face in enumerate(self.faces):
                if len(face) == 3:
                    _new_colors.append(self.colors[i])
                else:
                    _new_colors.extend([self.colors[i]] * 2)
            _new_colors = tuple(_new_colors)

        Mesh2D._check_required = False  # Turn off checks since we know the mesh is valid
        _new_mesh = Mesh2D(self.vertices, _new_faces, _new_colors)
        Mesh2D._check_required = True  # Turn the checks back on
        return _new_mesh

    def _check_vertices_input(self, vertices):
        assert isinstance(vertices, (list, tuple)), \
            'vertices should be a list or tuple. Got {}'.format(type(vertices))
        _verts_immutable = []
        for p in vertices:
            assert isinstance(p, (Point2D, Point2DImmutable)), \
                'Expected Point2D for Mesh2D vertex. Got {}.'.format(type(p))
            _verts_immutable.append(p.to_immutable())
        self._vertices = tuple(_verts_immutable)

    def _check_faces_input(self, faces):
        assert isinstance(faces, (list, tuple)), \
            'faces should be a list or tuple. Got {}'.format(type(faces))
        assert len(faces) > 0, 'Mesh2D must have at least one face.'
        for f in faces:
            assert isinstance(f, tuple), \
                'Expected tuple for Mesh2D face. Got {}.'.format(type(f))
            assert len(f) == 3 or len(f) == 4, \
                'Mesh2D face can only have 3 or 4 vertices. Got {}.'.format(len(f))
        assert isinstance(faces[0][0], int), 'Mesh2D face must use integers to ' \
            'reference vertices. Got {}.'.format(type(faces[0][0]))
        if isinstance(faces, list):
            faces = tuple(faces)
        self._faces = faces

    def _face_area(self, face):
        """Return the area of a face."""
        return Mesh2D._get_area([self._vertices[i] for i in face])

    def _tri_face_centroid(self, face):
        """Compute the centroid of a triangular face."""
        return Mesh2D._tri_centroid([self._vertices[i] for i in face])

    def _quad_face_centroid(self, face):
        """Compute the centroid of a quadrilateral face."""
        return Mesh2D._quad_centroid([self._vertices[i] for i in face])

    @staticmethod
    def _ear_clipping_triangulation(polygon):
        """Triangulate a polygon using the ear clipping method."""
        _faces = []
        _clipping_poly = polygon.duplicate()
        Polygon2D._check_required = False  # Turn off check since we know input is valid
        while len(_clipping_poly) > 3:
            _ear, _index = Mesh2D._find_ear(_clipping_poly)
            _faces.append(_ear)
            new_verts = list(_clipping_poly.vertices)
            del new_verts[_index]
            _clipping_poly = Polygon2D(new_verts)
        _faces.append(_clipping_poly.vertices)
        Polygon2D._check_required = True  # Turn the checks back on
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
                if len(polygon.intersect_line2(diagonal)) < 5:
                    ear = (polygon[i - 1], polygon[i], polygon[i + 1])  # found an ear!
                    break
        return ear, i

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
        Polygon2D._check_required = False  # Turn off check. We know input is valid
        quad_poly = Polygon2D(verts)
        Polygon2D._check_required = True  # Turn the checks back on
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
        return Point2DImmutable(_cent_x, _cent_y)

    @staticmethod
    def _tri_centroid(verts):
        """Get the centroid of a list of 3 Point2D vertices."""
        _cent_x = sum([v.x for v in verts])
        _cent_y = sum([v.y for v in verts])
        return Point2DImmutable(_cent_x / 3, _cent_y / 3)

    @staticmethod
    def _get_area(verts):
        """Return the area of a list of Point2D vertices."""
        _a = 0
        for i, pt in enumerate(verts):
            _a += verts[i - 1].x * pt.y - verts[i - 1].y * pt.x
        return abs(_a / 2)

    def __copy__(self):
        Mesh2D._check_required = False  # Turn off check since we know the mesh is valid
        _new_mesh = Mesh2D(self.vertices, self.faces, self.colors)
        Mesh2D._check_required = True  # Turn the checks back on
        return _new_mesh

    def __repr__(self):
        return 'Ladybug Mesh2D ({} faces) ({} vertices)'.format(
            len(self.faces), len(self))
