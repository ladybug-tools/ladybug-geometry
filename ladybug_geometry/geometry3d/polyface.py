# coding=utf-8
"""Object with Multiple Planar Faces in 3D Space"""
from __future__ import division

from .pointvector import Vector3D, Point3D
from .ray import Ray3D
from .line import LineSegment3D
from .plane import Plane
from .face import Face3D
from ._2d import Base2DIn3D

import math
try:
    from itertools import izip as zip  # python 2
except ImportError:
    xrange = range  # python 3


class Polyface3D(Base2DIn3D):
    """Object with Multiple Planar Faces in 3D Space. Includes solid objects and polyhedra.

    Args:
        vertices: A list of Point3D objects representing the vertices of
            this PolyFace.
        face_indices: A list of lists with one list for each face of the polyface.
            Each face list must contain at least one tuple of integers corresponding
            to indices within the vertices list. Additional tuples of integers may
            follow this one such that the first tuple denotes the boundary of the
            face while each subsequent tuple denotes a hole in the face.
        edge_information: Optional edge information, which will speed up the
            creation of the Polyface object if it is available but should be left
            as None if it is unknown. If None, edge_information will be computed
            from the vertices and face_indices inputs. Edge information
            should be formatted as a dictionary with two keys as follows:

            *   'edge_indices':
                An array objects that each contain two integers.
                These integers correspond to indices within the vertices list and
                each tuple represents a line segment for an edge of the polyface.
            *   'edge_types':
                An array of integers for each edge that parallels the edge_indices
                list. An integer of 0 denotes a naked edge, an integer of 1
                denotes an internal edge. Anything higher is a non-manifold edge.

    Properties:
        * vertices
        * faces
        * edges
        * naked_edges
        * internal_edges
        * non_manifold_edges
        * face_indices
        * edge_indices
        * edge_types
        * min
        * max
        * center
        * area
        * volume
        * is_solid
    """
    __slots__ = ('_faces', '_edges',
                 '_naked_edges', '_internal_edges', '_non_manifold_edges',
                 '_face_indices', '_edge_indices', '_edge_types',
                 '_area', '_volume', '_is_solid')

    def __init__(self, vertices, face_indices, edge_information=None):
        """Initialize Polyface3D."""
        # assign input properties
        Base2DIn3D.__init__(self, vertices)
        self._face_indices = tuple(tuple(tuple(loop) for loop in face)
                                   for face in face_indices)

        if edge_information is not None:  # unpack the input edge information
            edge_i = edge_information['edge_indices']
            edge_t = edge_information['edge_types']
        else:  # autocalculate the edge information from the vertices and faces
            edge_i = []
            edge_t = []
            for face in face_indices:
                for fi in face:
                    for i, vi in enumerate(fi):
                        try:  # this can get slow for large number of vertices
                            ind = edge_i.index((vi, fi[i - 1]))
                            edge_t[ind] += 1
                        except ValueError:  # make sure reversed edge isn't there
                            try:
                                ind = edge_i.index((fi[i - 1], vi))
                                edge_t[ind] += 1
                            except ValueError:  # add a new edge
                                if fi[i - 1] != vi:  # avoid cases of same start and end
                                    edge_i.append((fi[i - 1], vi))
                                    edge_t.append(0)
        self._edge_indices = edge_i if isinstance(edge_i, tuple) else tuple(edge_i)
        self._edge_types = edge_t if isinstance(edge_t, tuple) else tuple(edge_t)

        # determine solidity of the polyface by checking for internal edges
        self._is_solid = True
        for edge in self._edge_types:
            if edge != 1:
                self._is_solid = False
                break

        # assign default properties
        self._faces = None
        self._edges = None
        self._naked_edges = None
        self._internal_edges = None
        self._non_manifold_edges = None
        self._area = None
        self._volume = None

    @classmethod
    def from_dict(cls, data):
        """Create a Face3D from a dictionary.

        Args:
            data: A python dictionary in the following format

        .. code-block:: python

            {
                "type": "Polyface3D",
                "vertices": [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)],
                "face_indices": [[(0, 1, 2)], [(3, 0, 1)]],
                "edge_information": {"edge_indices":[(0, 1), (1, 2), (2, 0), (2, 3), (3, 0)],
                                    "edge_types":[0, 0, 1, 0, 0]}
            }
        """
        if 'edge_information' in data and data['edge_information'] is not None:
            edge_information = data['edge_information']
        else:
            edge_information = None

        return cls(tuple(Point3D.from_array(pt) for pt in data['vertices']),
                   data['face_indices'], edge_information)

    @classmethod
    def from_faces(cls, faces, tolerance):
        """Initialize Polyface3D from a list of Face3D objects.

        Note that the Polyface3D.faces property of the resulting polyface will
        have an order of faces that matches the order input to this classmethod.

        Args:
            faces: A list of Face3D objects representing the boundary of this Polyface.
            tolerance: The maximum difference between x, y, and z values at which
                the vertex of two adjacent faces is considered the same.
        """
        # extract unique vertices from the faces
        vertices = []  # collection of vertices as point objects
        face_indices = []  # collection of face indices
        for f in faces:
            ind = []
            loops = (f.boundary,) if not f.has_holes else (f.boundary,) + f.holes
            for j, loop in enumerate(loops):
                ind.append([])
                for v in loop:
                    found = False
                    for i, vert in enumerate(vertices):
                        if v.is_equivalent(vert, tolerance):
                            found = True
                            ind[j].append(i)
                            break
                    if not found:  # add new point
                        vertices.append(v)
                        ind[j].append(len(vertices) - 1)
            face_indices.append(tuple(ind))

        # get the polyface object and assign correct faces to it
        face_obj = cls(vertices, face_indices)
        if face_obj._is_solid:
            face_obj._faces = cls.get_outward_faces(faces, 0.01)
        else:
            face_obj._faces = tuple(faces)
        return face_obj

    @classmethod
    def from_box(cls, width, depth, height, base_plane=None):
        """Initialize Polyface3D from parameters describing a box.

        Initializing a polyface this way has the added benefit of having its
        faces property quickly calculated.

        Args:
            width: A number for the width of the box (in the X direction).
            depth: A number for the depth of the box (in the Y direction).
            height: A number for the height of the box (in the Z direction).
            base_plane: A Plane object from which to generate the box.
                If None, default is the WorldXY plane.
        """
        assert isinstance(width, (float, int)), 'Box width must be a number.'
        assert isinstance(depth, (float, int)), 'Box depth must be a number.'
        assert isinstance(height, (float, int)), 'Box height must be a number.'
        if base_plane is not None:
            assert isinstance(base_plane, Plane), \
                'base_plane must be Plane. Got {}.'.format(type(base_plane))
        else:
            base_plane = Plane(Vector3D(0, 0, 1), Point3D())
        _o = base_plane.o
        _w_vec = base_plane.x * width
        _d_vec = base_plane.y * depth
        _h_vec = base_plane.n * height
        _verts = (_o, _o + _d_vec, _o + _d_vec + _w_vec, _o + _w_vec,
                  _o + _h_vec, _o + _d_vec + _h_vec,
                  _o + _d_vec + _w_vec + _h_vec, _o + _w_vec + _h_vec)
        _face_indices = ([(0, 1, 2, 3)], [(2, 1, 5, 6)], [(6, 7, 3, 2)],
                         [(0, 3, 7, 4)], [(0, 4, 5, 1)], [(7, 6, 5, 4)])
        _edge_indices = ((3, 0), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5),
                         (5, 1), (3, 7), (7, 4), (6, 2), (5, 6), (6, 7))
        polyface = cls(_verts, _face_indices, {'edge_indices': _edge_indices,
                                               'edge_types': [1] * 12})
        verts = tuple(tuple(_verts[i] for i in face[0]) for face in _face_indices)
        bottom = Face3D(verts[0], base_plane.flip(), enforce_right_hand=False)
        middle = tuple(Face3D(v, enforce_right_hand=False) for v in verts[1:5])
        top = Face3D(verts[5], base_plane.move(_h_vec), enforce_right_hand=False)
        polyface._faces = (bottom,) + middle + (top,)
        polyface._volume = width * depth * height
        return polyface

    @classmethod
    def from_offset_face(cls, face, offset):
        """Initialize a solid Polyface3D from a Face3D offset along its normal.

        The resulting polyface will always be offset in the direction of
        the face normal.

        When a polyface is initialized this way, the first face of the
        Polysurface3D.faces will always be the input face used to create the
        object, the last face will be the offset version of the face, and all
        other faces will form the extrusion connecting the two.

        Args:
            face: A Face3D to serve as a base for the polyface.
            offset: A number for the distance to offset the face to make a solid.
        """
        assert isinstance(face, Face3D), \
            'face must be a Face3D. Got {}.'.format(type(face))
        assert isinstance(offset, (float, int)), \
            'height must be a number. Got {}.'.format(type(offset))
        # compute vertices, face indices, and edges of the extrusion
        extru_vec = face.normal * offset
        verts, face_ind_extru, edge_indices = \
            Polyface3D._verts_faces_edges_from_boundary(face.boundary, extru_vec)
        if face.has_holes:
            _st_i = len(verts)
            for i, hole in enumerate(face.hole_polygon2d):
                hole_verts = face._holes[i] if hole.is_clockwise else \
                    tuple(reversed(face._holes[i]))
                verts_2, face_ind_extru_2, edge_indices_2 = \
                    Polyface3D._verts_faces_edges_from_boundary(
                        hole_verts, extru_vec, _st_i)
                verts.extend(verts_2)
                face_ind_extru.extend(face_ind_extru_2)
                edge_indices.extend(edge_indices_2)
                _st_i += len(hole_verts * 2)
        face_ind_extru = [[fc] for fc in face_ind_extru]
        # compute the final faces (accounting for top and bottom)
        if not face.has_holes:
            len_faces = len(face.boundary)
            face_ind_bottom = [tuple(reversed(xrange(len_faces)))]
            face_ind_top = [tuple(
                reversed(xrange(len_faces * 2 - 1, len_faces - 1, -1)))]
        else:
            face_verts_bottom = [list(reversed(face.boundary))] + list(face.holes)
            face_verts_top = [[pt.move(extru_vec) for pt in reversed(loop)]
                              for loop in face_verts_bottom]
            face_ind_bottom = [tuple(verts.index(pt) for pt in loop)
                               for loop in face_verts_bottom]
            face_ind_top = [tuple(verts.index(pt) for pt in loop)
                            for loop in face_verts_top]
        faces_ind = [face_ind_bottom] + face_ind_extru + [face_ind_top]
        # create the polysurface and assign known properties.
        polyface = cls(verts, faces_ind, {'edge_indices': edge_indices,
                                          'edge_types': [1] * len(edge_indices)})
        polyface._volume = face.area * offset
        face_verts = tuple(
            tuple(tuple(verts[i] for i in loop) for loop in f) for f in faces_ind)
        if not face.has_holes:
            polyface._faces = tuple(Face3D(v[0], enforce_right_hand=False)
                                    for v in face_verts)
        else:
            mid_faces = [Face3D(v[0], enforce_right_hand=False)
                         for v in face_verts[1:-1]]
            bottom_face = face.flip()
            top_face = face.move(extru_vec)
            polyface._faces = tuple([bottom_face] + mid_faces + [top_face])
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
            faces = []
            for face in self._face_indices:
                boundary = tuple(self.vertices[i] for i in face[0])
                if len(face) == 1:
                    faces.append(Face3D(boundary))
                else:
                    holes = tuple(tuple(self.vertices[i] for i in f) for f in face[1:])
                    faces.append(Face3D(boundary=boundary, holes=holes))
            if self._is_solid:
                self._faces = Polyface3D.get_outward_faces(faces, 0.01)
            else:
                self._faces = tuple(faces)
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
    def edge_information(self):
        """Dictionary with keys: 'edge_indices', 'edge_types' and corresponding properties.
        """
        return {'edge_indices': self._edge_indices, 'edge_types': self._edge_types}

    @property
    def area(self):
        """The total surface area of the polyface."""
        if self._area is None:
            self._area = sum([face.area for face in self.faces])
        return self._area

    @property
    def volume(self):
        """The volume enclosed by the polyface.

        Note that, if this polyface is not solid (with all face normals pointing
        outward), the value of this property will not be valid.
        """
        if self._volume is None:
            # formula taken from https://en.wikipedia.org/wiki/Polyhedron#Volume
            _v = 0
            for i, face in enumerate(self.faces):
                _v += face[0].dot(face.normal) * face.area
            self._volume = _v / 3
        return self._volume

    @property
    def is_solid(self):
        """A boolean to note whether the polyface is solid (True) or is open (False).

        Note that all solid polyface objects will have faces pointing outwards.
        """
        return self._is_solid

    def merge_overlapping_edges(self, tolerance, angle_tolerance):
        """Get this object with overlapping naked edges merged into single internal edges.

        This can be used to determine if a polyface is truly solid.
        The default test of edge conditions that runs upon creation of a polyface does
        not check for cases where overlapping colinear edges could be considered
        a single internal edge such as the case below:

        .. code-block:: shell

                             |           1          |
                            A|______________________|C
                             |          B|          |
                             |           |          |
                             |     2     |     3    |

        If Face 1 only has edge AC and not two separate edges for AB and BC, the
        creation of the polyface will yield naked edges for AC, AB, and BC, meaning
        the shape would not be considered solid when it might actually be so. This
        merge_overlapping_edges method overcomes this by replacing the entire set
        of 3 naked edges above a single internal edge running from A to C.

        Args:
            tolerance: The minimum distance between a vertex and the boundary segments
                at which point the vertex is considered colinear.
            angle_tolerance: The max angle in radians that vertices are allowed to
                 differ from one another in order to consider them colinear.
        """
        # get naked edges
        naked_edges = list(self.naked_edges)
        if len(naked_edges) == 0:
            return self

        # establish lists that will be iteratively edited
        remove_i = []
        add_edges = []
        naked_edge_i = []
        naked_edge_ind = []
        for i, x in enumerate(self.edge_types):
            if x == 0:
                naked_edge_i.append(i)
                naked_edge_ind.append(self._edge_indices[i])

        while len(naked_edge_i) > 1:
            # get all of the edges that are colinear with the first edge
            coll_edges = list(naked_edge_ind[0])
            coll_i = [naked_edge_i[0]]
            kept_i = []
            for edge, ind, nei, i in zip(
                    naked_edges[1:], naked_edge_ind[1:], naked_edge_i[1:],
                    xrange(1, len(naked_edges))):
                try:
                    if edge.is_colinear(naked_edges[0], tolerance, angle_tolerance):
                        coll_edges.extend(ind)
                        coll_i.append(nei)
                    else:
                        kept_i.append(i)
                except ZeroDivisionError:  # duplicate vertices resulted in 0 length edge
                    coll_edges.extend(ind)
                    coll_i.append(nei)

            # determine if  colinear edges create a full double line along the edge
            if len(coll_edges) == 1:
                overlapping = False
            else:
                final_vi = []
                coll_edges.sort()
                overlapping = True
                for i in range(0, len(coll_edges), 2):
                    final_vi.append(coll_edges[i])
                    if not coll_edges[i] == coll_edges[i + 1]:
                        overlapping = False
                        break

            # if fully overlapping edges have been found, remake them into one
            if overlapping:
                remove_i.extend(coll_i)  # remove overlapping edges from the list
                verts = [self.vertices[j] for j in final_vi]
                dir_vec = verts[0] - verts[1]
                if dir_vec.x != 0:
                    vert_coor = [v.x for v in verts]
                elif dir_vec.y != 0:
                    vert_coor = [v.y for v in verts]
                else:
                    vert_coor = [v.z for v in verts]
                vert_coor, final_vi = zip(*sorted(zip(vert_coor, final_vi)))
                add_edges.append((final_vi[0], final_vi[-1]))

            # delete the colinear vertices that have been accounted for
            naked_edges = [naked_edges[i] for i in kept_i]
            naked_edge_ind = [naked_edge_ind[i] for i in kept_i]
            naked_edge_i = [naked_edge_i[i] for i in kept_i]

        # create the new edge information and the new polyface
        new_edge_indices = list(self.edge_indices)
        new_edge_types = list(self.edge_types)
        add_i = []
        for i in range(len(new_edge_indices)):
            if i not in remove_i:
                add_i.append(i)
        new_edge_indices = [new_edge_indices[i] for i in add_i]
        new_edge_types = [new_edge_types[i] for i in add_i]
        for new_edge in add_edges:
            new_edge_indices.append(new_edge)
            new_edge_types.append(1)
        _new_polyface = Polyface3D(self._vertices, self._face_indices,
                                   {'edge_indices': new_edge_indices,
                                    'edge_types': new_edge_types})
        return _new_polyface

    def move(self, moving_vec):
        """Get a polyface that has been moved along a vector.

        Args:
            moving_vec: A Vector3D with the direction and distance to move the polyface.
        """
        _verts = tuple(pt.move(moving_vec) for pt in self.vertices)
        _new_pface = Polyface3D(_verts, self.face_indices, self.edge_information)
        if self._faces is not None:
            _new_pface._faces = tuple(face.move(moving_vec) for face in self._faces)
        _new_pface._volume = self._volume
        return _new_pface

    def rotate(self, axis, angle, origin):
        """Rotate a polyface by a certain angle around an axis and origin.

        Right hand rule applies:
        If axis has a positive orientation, rotation will be clockwise.
        If axis has a negative orientation, rotation will be counterclockwise.

        Args:
            axis: A Vector3D axis representing the axis of rotation.
            angle: An angle for rotation in radians.
            origin: A Point3D for the origin around which the object will be rotated.
        """
        _verts = tuple(pt.rotate(axis, angle, origin) for pt in self.vertices)
        _new_pface = Polyface3D(_verts, self.face_indices, self.edge_information)
        if self._faces is not None:
            _new_pface._faces = tuple(face.rotate(axis, angle, origin)
                                      for face in self._faces)
        _new_pface._volume = self._volume
        return _new_pface

    def rotate_xy(self, angle, origin):
        """Get a polyface rotated counterclockwise in the world XY plane by an angle.

        Args:
            angle: An angle in radians.
            origin: A Point3D for the origin around which the object will be rotated.
        """
        _verts = tuple(pt.rotate_xy(angle, origin) for pt in self.vertices)
        _new_pface = Polyface3D(_verts, self.face_indices, self.edge_information)
        if self._faces is not None:
            _new_pface._faces = tuple(face.rotate_xy(angle, origin)
                                      for face in self._faces)
        _new_pface._volume = self._volume
        return _new_pface

    def reflect(self, normal, origin):
        """Get a polyface reflected across a plane with the input normal vector and origin.

        Args:
            normal: A Vector3D representing the normal vector for the plane across
                which the polyface will be reflected. THIS VECTOR MUST BE NORMALIZED.
            origin: A Point3D representing the origin from which to reflect.
        """
        _verts = tuple(pt.reflect(normal, origin) for pt in self.vertices)
        _new_pface = Polyface3D(_verts, self.face_indices, self.edge_information)
        if self._faces is not None:
            _new_pface._faces = tuple(face.reflect(normal, origin)
                                      for face in self._faces)
        _new_pface._volume = self._volume
        return _new_pface

    def scale(self, factor, origin=None):
        """Scale a polyface by a factor from an origin point.

        Args:
            factor: A number representing how much the polyface should be scaled.
            origin: A Point3D representing the origin from which to scale.
                If None, it will be scaled from the World origin (0, 0, 0).
        """
        if origin is None:
            _verts = tuple(Point3D(pt.x * factor, pt.y * factor, pt.z * factor)
                           for pt in self._vertices)
        else:
            _verts = tuple(pt.scale(factor, origin) for pt in self.vertices)
        _new_pface = Polyface3D(_verts, self.face_indices, self.edge_information)
        if self._faces is not None:
            _new_pface._faces = tuple(face.scale(factor, origin)
                                      for face in self._faces)
        _new_pface._volume = self._volume * factor ** 3 \
            if self._volume is not None else None
        return _new_pface

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

    def does_intersect_line_ray_exist(self, line_ray):
        """Boolean denoting whether an intersection exists between the input Line3D or Ray3D.

        Args:
            line_ray: A Line3D or Ray3D object for which intersection will be evaluated.

        Returns:
            True if an intersection exists. False if it does not exist.
        """
        for face in self.faces:
            _int = face.intersect_line_ray(line_ray)
            if _int is not None:
                return True
        return False

    def intersect_line_ray(self, line_ray):
        """Get the intersections between this polyface and the input Line3D or Ray3D.

        Args:
            line_ray: A Line3D or Ray3D object for which intersection will be computed.

        Returns:
            A list of Point3D for the intersection. Will be an empty list if no
            intersection exists.
        """
        _inters = []
        for face in self.faces:
            _int = face.intersect_line_ray(line_ray)
            if _int is not None:
                _inters.append(_int)
        return _inters

    def intersect_plane(self, plane):
        """Get the intersection between this polyface and the input plane.

        Args:
            plane: A Plane object for which intersection will be computed.

        Returns:
            List of LineSegment3D objects for the intersection.
            Will be an empty list if no intersection exists.
        """
        _inters = []
        for face in self.faces:
            _int = face.intersect_plane(plane)
            if _int is not None:
                _inters.extend(_int)
        return _inters

    @staticmethod
    def overlapping_bounding_boxes(polyface1, polyface2, tolerance):
        """Check if the bounding boxes of two polyfaces overlap within a tolerance.

        This is particularly useful as a check before performing computationally
        intenseprocesses between two polyfaces like intersection or checking for
        adjacency. Checking the overlap of the bounding boxes is extremely quick
        given this method's use of the Separating Axis Theorem.

        Args:
            polyface1: The first polyface to check.
            polyface2: The second polyface to check.
            tolerance: Distance within which two points are considered to be co-located.
        """
        # Bounding box check using the Separating Axis Theorem
        polyf1_width = polyface1.max.x - polyface1.min.x
        polyf2_width = polyface2.max.x - polyface2.min.x
        dist_btwn_x = abs(polyface1.center.x - polyface2.center.x)
        x_gap_btwn_box = dist_btwn_x - (0.5 * polyf1_width) - (0.5 * polyf2_width)

        polyf1_depth = polyface1.max.y - polyface1.min.y
        polyf2_depth = polyface2.max.y - polyface2.min.y
        dist_btwn_y = abs(polyface1.center.y - polyface2.center.y)
        y_gap_btwn_box = dist_btwn_y - (0.5 * polyf1_depth) - (0.5 * polyf2_depth)

        polyf1_height = polyface1.max.z - polyface1.min.z
        polyf2_height = polyface2.max.z - polyface2.min.z
        dist_btwn_z = abs(polyface1.center.z - polyface2.center.z)
        z_gap_btwn_box = dist_btwn_z - (0.5 * polyf1_height) - (0.5 * polyf2_height)

        if x_gap_btwn_box > tolerance or y_gap_btwn_box > tolerance or \
                z_gap_btwn_box > tolerance:
            return False  # no overlap
        return True  # overlap exists

    @staticmethod
    def get_outward_faces(faces, tolerance):
        """Turn a list of faces forming a solid into one where they all point outward.

        Note that, if the input faces do not form a closed solid, there may be some
        output faces that are not pointing outward.  However, if the gaps in the
        combined solid are within the input tolerance, this should not be an issue.

        Also, note that this method runs automatically for any solid polyface
        (meaning every solid polyface automatically has outward-facing faces). So there
        is no need to rerun this method for faces from a solid polyface.

        Args:
            faces: A list of Face3D objects that together form a solid.
            tolerance: Optional tolerance for the permissable size of gap between
                faces at which point the faces are considered to have a single edge.

        Returns:
            outward_faces -- A list of the input Face3D objects that all point outwards
            (provided the input faces form a solid).
        """
        outward_faces = []
        for i, face in enumerate(faces):
            # construct a ray with the face normal and a point on the face
            point_on_face = face._point_on_face(tolerance)
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
                outward_faces.append(face)
            else:
                outward_faces.append(face.flip())
        return outward_faces

    def to_dict(self, include_edge_information=True):
        """Get Polyface3D as a dictionary.

        Args:
            include_edge_information: Set to True to include the edge_information
                in the dictionary, which will allow for fast initialization when
                it is de-serialized. Default True.
        """
        base = {'type': 'Polyface3D',
                'vertices': [v.to_array() for v in self.vertices],
                'face_indices': self.face_indices}
        if include_edge_information:
            base['edge_information'] = self.edge_information
        return base

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
    def _verts_faces_edges_from_boundary(cclock_verts, extru_vec, st_i=0):
        """Get vertices and face indices for a given Face3D loop (boundary or hole)."""
        verts = list(cclock_verts) + [pt.move(extru_vec) for pt in cclock_verts]
        len_faces = len(cclock_verts)
        faces_ind = []
        for i in xrange(st_i, st_i + len_faces - 1):
            faces_ind.append((i, i + 1, i + len_faces + 1, i + len_faces))
        faces_ind.append((st_i + len_faces - 1, st_i,
                          st_i + len_faces, st_i + len_faces * 2 - 1))
        edge_i1 = [(st_i + i, st_i + i + 1) for i in xrange(len_faces - 1)]
        edge_i2 = [(st_i + i, st_i + i + len_faces) for i in xrange(len_faces)]
        edge_i3 = [(st_i + len_faces + i, st_i + len_faces + i + 1)
                   for i in xrange(len_faces - 1)]
        edge_indices = edge_i1 + [(st_i + len_faces - 1, st_i)] + edge_i2 + edge_i3 + \
            [(st_i + len_faces * 2 - 1, st_i + len_faces)]
        return verts, faces_ind, edge_indices

    def __copy__(self):
        _new_poly = Polyface3D(self.vertices, self.face_indices, self.edge_information)
        _new_poly._faces = self._faces
        return _new_poly

    def __key(self):
        """A tuple based on the object properties, useful for hashing."""
        return tuple(hash(pt) for pt in self._vertices) + \
            tuple(hash(face) for face in self._face_indices)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(other, Polyface3D) and self.__key() == other.__key()

    def __repr__(self):
        return 'Polyface3D ({} faces) ({} vertices)'.format(
            len(self.faces), len(self))
