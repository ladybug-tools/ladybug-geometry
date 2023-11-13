# coding=utf-8
"""A class that supports the import and export of OBJ data to/from ladybug_geometry.
"""
import os

try:
    from itertools import izip as zip  # python 2
    writemode = 'wb'
except ImportError:
    writemode = 'w'  # python 3

from ladybug_geometry.geometry2d.pointvector import Point2D
from ladybug_geometry.geometry3d.pointvector import Vector3D, Point3D


class OBJ(object):
    """A class that supports the import and export of OBJ data to/from ladybug_geometry.

    Note that ladybug_geometry Mesh3D can be easily created from this OBJ by
    taking the vertices and normals.

    Args:
        vertices: A list or tuple of Point3D objects for vertices.
        faces: A list of tuples with each tuple having either 3 or 4 integers.
            These integers correspond to indices within the list of vertices.
        vertex_texture_map: An optional list or tuple of Point2D that align with the
            vertices input. All coordinate values of the Point2D should be between
            0 and 1 and are intended to map to the XY system of images to be mapped
            onto the OBJ mesh. If None, the OBJ file is written without
            textures. (Default: None).
        vertex_normals: An optional list or tuple of Vector3D that align with the
            vertices input and describe the normal vector to be used at each vertex.
            If None, the OBJ file is written without normals. (Default: None).
        vertex_colors: An optional list of colors that align with the vertices input.
            Note that these are written into the OBJ alongside the vertex
            coordinates separately from the texture map. Not all programs support
            importing OBJs with this color information but Rhino does. (Default: None).
        material_structure: A list of tuples where each tuple contains two elements.
            The first is the identifier of a material that is used in the OBJ and
            the second is the index of the face where the application of the new
            material begins. If None, everything will be assumed to have the
            same diffuse material. (Default: None).

    Properties:
        * vertices
        * faces
        * vertex_texture_map
        * vertex_normals
        * vertex_colors
        * material_structure
    """

    __slots__ = (
        '_vertices', '_faces', '_vertex_texture_map', '_vertex_normals',
        '_vertex_colors', '_material_structure'
    )

    def __init__(
            self, vertices, faces, vertex_texture_map=None, vertex_normals=None,
            vertex_colors=None, material_structure=None
        ):
        self._vertices = self._check_vertices_input(vertices)
        self._faces = self._check_faces_input(faces)
        self.vertex_texture_map = vertex_texture_map
        self.vertex_normals = vertex_normals
        self.vertex_colors = vertex_colors
        self.material_structure = material_structure

    @classmethod
    def from_file(cls, file_path):
        """Create an OBJ object from a .obj file.

        Args:
            file_path: Path to an OBJ file as a text string. Note that, if the file
                includes texture mapping coordinates or vertex normals, the number
                of texture coordinates and normals must align with the number of
                vertices to be importable. Nearly all OBJ files follow this standard.
                If any of the OBJ mesh faces contain more than 4 vertices, only
                the first 4 vertices will be counted.
        """
        vertices, faces, vertex_texture_map, vertex_normals, vertex_colors = \
            [], [], [], [], []
        mat_struct = []
        with open(file_path, 'r') as fp:
            for line in fp:
                if line.startswith('#'):
                    continue
                wds = line.split()
                if len(wds) > 0:
                    first_word = wds[0]
                    if first_word == 'v':  # start of a new vertex
                        vert = Point3D(float(wds[1]), float(wds[2]), float(wds[3]))
                        vertices.append(vert)
                        if len(wds) > 4:
                            vertex_colors.append(tuple(wds[4:]))
                    elif first_word == 'f':  # start of a new face
                        face = []
                        for fv in wds[1:]:
                            face.append(int(fv.split('/')[0]) - 1)
                        if len(face) > 4:  # truncate for compatibility with Mesh3D
                            face = face[:4]
                        faces.append(tuple(face))
                    elif first_word == 'vn':  # start of a new vertex normal
                        norm = Vector3D(float(wds[1]), float(wds[2]), float(wds[3]))
                        vertex_normals.append(norm)
                    elif first_word == 'vt':  # start of a new texture coordinate
                        texture = Point2D(float(wds[1]), float(wds[2]))
                        vertex_texture_map.append(texture)
                    elif first_word == 'usemtl':  # start of a new material application
                        mat_struct.append((wds[1], len(faces)))
        return cls(vertices, faces, vertex_texture_map, vertex_normals,
                   vertex_colors, mat_struct)

    @classmethod
    def from_mesh3d(cls, mesh, include_colors=True, include_normals=False):
        """Create an OBJ object from a ladybug_geometry Mesh3D.
        
        If colors are specified on the Mesh3D, they will be correctly transferred
        to the resulting OBJ object as long as include_colors is True.

        Args:
            mesh: A ladybug_geometry Mesh3D object to be converted to an OBJ object.
            include_colors: Boolean to note whether the Mesh3D colors should be
                transferred to the OBJ object. (Default: True).
            include_normals: Boolean to note whether the vertex normals should be
                included in the resulting OBJ object. (Default: False).
        """
        if include_colors and mesh.is_color_by_face:
            # we need to duplicate vertices to preserve colors
            vertices, faces, colors = [], [], []
            v_ct = 0
            for face_verts, col in zip(mesh.face_vertices, mesh.colors):
                vertices.extend(face_verts)
                if len(face_verts) == 4:
                    faces.append((v_ct, v_ct + 1, v_ct + 2, v_ct + 3))
                    colors.extend([col] * 4)
                    v_ct += 4
                else:
                    faces.append((v_ct, v_ct + 1, v_ct + 2))
                    colors.extend([col] * 3)
                    v_ct += 3
            if include_normals:
                msh_norms = mesh.vertex_normals
                vert_normals = []
                for face in mesh.faces:
                    for fi in face:
                        vert_normals.append(msh_norms[fi])
                return cls(vertices, faces, vertex_normals=msh_norms,
                           vertex_colors=colors)
            return cls(vertices, faces, vertex_colors=colors)
        vertex_colors = mesh.colors if include_colors else None
        if include_normals:
            return cls(mesh.vertices, mesh.faces, vertex_normals=mesh.vertex_normals,
                       vertex_colors=vertex_colors)
        return cls(mesh.vertices, mesh.faces, vertex_colors=vertex_colors)

    @classmethod
    def from_mesh3ds(cls, meshes, material_ids=None, include_normals=False):
        """Create an OBJ object from a list of ladybug_geometry Mesh3D.

        Mesh3D colors are ignored when using this method with the assumption that
        materials are used to specify how the meshes should be rendered.

        Args:
            meshes: A list of ladybug_geometry Mesh3D objects to be converted
                into an OBJ object.
            material_ids: An optional list of strings that aligns with the input
                meshes and denote materials assigned to each mesh. This list of
                material IDs will be automatically converted into an efficient
                material_structure for the OBJ object where materials used for
                multiple meshes only include one reference to the material. If
                None, the OBJ will have no material structure. (Default: None).
            include_normals: Boolean to note whether the vertex normals should be
                included in the resulting OBJ object. (Default: False).
        """
        # sort the meshes by material ID to ensure efficient material structure
        if material_ids is not None:
            assert len(material_ids) == len(meshes), 'Length of OBJ material_ids ({}) ' \
                'does not match the length of meshes ({}).'.format(
                    len(material_ids), len(meshes))
            meshes = [x for _, x in sorted(zip(material_ids, meshes))]
            material_ids = sorted(material_ids)

        # gather all vertices, faces, and (optionally) normals together
        vertices, faces, normals, mat_struct = [], [], [], []
        v_count = 0
        if material_ids is not None:
            last_mat = None
            for mesh, mat_id in zip(meshes, material_ids):
                if mat_id != last_mat:
                    mat_struct.append((mat_id, len(faces)))
                    last_mat = mat_id
                vertices.extend(mesh.vertices)
                if include_normals:
                    normals.extend(mesh.vertex_normals)
                if v_count == 0:
                    faces.extend(mesh.faces)
                else:
                    for f in mesh.faces:
                        faces.append(tuple(fi + v_count for fi in f))
                v_count += len(mesh.vertices)
        else:
            for mesh in meshes:
                vertices.extend(mesh.vertices)
                if include_normals:
                    normals.extend(mesh.vertex_normals)
                if v_count == 0:
                    faces.extend(mesh.faces)
                else:
                    for f in mesh.faces:
                        faces.append(tuple(fi + v_count for fi in f))
                v_count += len(mesh.vertices)

        return cls(
            vertices, faces, vertex_normals=normals, material_structure=mat_struct)

    @property
    def vertices(self):
        """Tuple of Point3D for all vertices in the OBJ."""
        return self._vertices

    @property
    def faces(self):
        """Tuple of tuples for all faces in the OBJ."""
        return self._faces

    @property
    def vertex_texture_map(self):
        """Get or set a tuple of Point2D for texture image coordinates for each vertex.

        Will be None if no texture map is assigned.
        """
        return self._vertex_texture_map

    @vertex_texture_map.setter
    def vertex_texture_map(self, value):
        if value is not None:
            assert isinstance(value, (list, tuple)), 'vertex_texture_map should be ' \
                'a list or tuple. Got {}'.format(type(value))
            if isinstance(value, list):
                value = tuple(value)
            if len(value) == 0:
                value = None
            elif len(value) != len(self.vertices):
                raise ValueError(
                    'Number of items in vertex_texture_map ({}) does not match number'
                    'of OBJ vertices ({}).'.format(len(value), len(self.vertices)))
            else:
                for vert in value:
                    assert isinstance(vert, Point2D), 'Expected Point2D for OBJ ' \
                        'vertex texture. Got {}.'.format(type(vert))
        self._vertex_texture_map = value

    @property
    def vertex_normals(self):
        """Get or set a tuple of Vector3D for vertex normals.

        Will be None if no vertex normals are assigned.
        """
        return self._vertex_normals

    @vertex_normals.setter
    def vertex_normals(self, value):
        if value is not None:
            assert isinstance(value, (list, tuple)), \
                'vertex_normals should be a list or tuple. Got {}'.format(type(value))
            if isinstance(value, list):
                value = tuple(value)
            if len(value) == 0:
                value = None
            elif len(value) != len(self.vertices):
                raise ValueError(
                    'Number of OBJ vertex_normals ({}) does not match the number of'
                    ' OBJ vertices ({}).'.format(len(value), len(self.vertices)))
            else:
                for norm in value:
                    assert isinstance(norm, Vector3D), 'Expected Vector3D for OBJ ' \
                        'vertex normal. Got {}.'.format(type(norm))
        self._vertex_normals = value

    @property
    def vertex_colors(self):
        """Get or set a list of colors for the OBJ. Will be None if no colors assigned.
        """
        return self._vertex_colors

    @vertex_colors.setter
    def vertex_colors(self, value):
        if value is not None:
            assert isinstance(value, (list, tuple)), \
                'vertex_normals should be a list or tuple. Got {}'.format(type(value))
            if isinstance(value, list):
                value = tuple(value)
            if len(value) == 0:
                value = None
            elif len(value) != len(self.vertices):
                raise ValueError(
                    'Number of OBJ vertex_normals ({}) does not match the number of'
                    ' OBJ vertices ({}).'.format(len(value), len(self.vertices)))
        self._vertex_colors = value

    @property
    def material_structure(self):
        """Get or set a tuple of tuples that specify the material structure of the obj.

        Each sub-tuple contains two elements. The first is the identifier of a
        material that is used in the OBJ and the second is the index of the face
        where the application of the new material begins. If None, everything
        will be assumed to have the same diffuse material.
        """
        return self._material_structure

    @material_structure.setter
    def material_structure(self, value):
        if value is not None:
            assert isinstance(value, (list, tuple)), \
                'vertex_normals should be a list or tuple. Got {}'.format(type(value))
            if len(value) == 0:
                value = None
            else:
                for mt in value:
                    assert isinstance(mt, tuple), 'Expected tuple for OBJ material ' \
                        'structure. Got {}.'.format(type(mt))
                    assert len(mt) == 2, 'OBJ material structure must have 2 items. ' \
                        'Got {}.'.format(len(mt))
                    assert isinstance(mt[0], str), 'Expected String for OBJ material ' \
                        'identifier. Got {}.'.format(type(mt[0]))
                    try:
                        self._faces[mt[1]]
                    except IndexError:
                        raise IndexError(
                            'OBJ material index {} does not correspond to any face. '
                            'There are {} faces in the mesh.'.format(
                                mt[1], len(self._faces)))
                    except TypeError:
                        raise TypeError(
                            'OBJ material must use integers to reference faces. '
                            'Got {}.'.format(type(mt[1])))
                value = sorted(value, key=lambda x: x[1])
                value = tuple(value)
        self._material_structure = value

    def to_file(self, folder, name, triangulate_quads=False, include_mtl=False):
        """Write the OBJ object to an ASCII text file.

        Args:
            folder: A text string for the directory where the OBJ will be written.
            name: A text string for the name of the OBJ file. Note that, if an image
                texture is meant to be assigned to this OBJ, the image should have
                the same name as the one input here except with the .mtl extension
                instead of the .obj extension.
            triangulate_quads: Boolean to note whether quad faces should be
                triangulated upon export to OBJ. This may be needed for certain
                software platforms that require the mesh to be composed entirely
                of triangles (eg. Radiance). (Default: False).
            include_mtl: Boolean to note whether an .mtl file should be automatically
                generated from the material structure written next to the .obj
                file in the output folder. All materials in the mtl file will
                be diffuse white, with the assumption that these will be
                customized later. (Default: False).
        """
        # set up a name and folder
        file_name = name if name.lower().endswith('.obj') else '{}.obj'.format(name)
        obj_file = os.path.join(folder, file_name)
        mtl_file = '{}.mtl'.format(name) if not name.lower().endswith('.obj') else \
            '{}.mtl'.format(name[:-4])

        # write everything into the OBJ file
        with open(obj_file, writemode) as outfile:
            # add a comment at the top to note where the OBJ is written from
            outfile.write('# OBJ file written by ladybug geometry\n\n')

            # add material file name if include_mtl is true
            if self._material_structure is not None or include_mtl:
                if include_mtl:
                    outfile.write('mtllib ' + mtl_file + '\n')
                if self._material_structure is None:
                    outfile.write('usemtl diffuse_0\n')

            # loop through the vertices and add them to the file
            if self.vertex_colors is None:
                for v in self.vertices:
                    outfile.write('v {} {} {}\n'.format(v.x, v.y, v.z))
            else:  # write the vertex colors alongside the vertices
                if len(self.vertex_colors[0]) > 3:
                    for v, c in zip(self.vertices, self.vertex_colors):
                        outfile.write(
                            'v {} {} {} {} {} {}\n'.format(
                                v.x, v.y, v.z, c[0], c[1], c[2])
                        )
                else:  # might be a grayscale weight
                    for v, c in zip(self.vertices, self.vertex_colors):
                        outfile.write(
                            'v {} {} {} {}\n'.format(v.x, v.y, v.z, ' '.join(c))
                        )

            # loop through the texture vertices, if present, and add them to the file
            if self.vertex_texture_map is not None:
                for vt in self.vertex_texture_map:
                    outfile.write('vt {} {}\n'.format(vt.x, vt.y))

            # loop through the normals, if present, and add them to the file
            if self.vertex_normals is not None:
                for vn in self.vertex_normals:
                    outfile.write('vn {} {} {}\n'.format(vn.x, vn.y, vn.z))

            # triangulate the faces if requested
            formatted_faces, formatted_mats = self.faces, self.material_structure
            if triangulate_quads:
                formatted_faces = []
                if formatted_mats is None or len(formatted_mats) == 1:
                    for f in self.faces:
                        if len(f) > 3:
                            formatted_faces.append((f[0], f[1], f[2]))
                            formatted_faces.append((f[2], f[3], f[0]))
                        else:
                            formatted_faces.append(f)
                else:
                    mat_ind = [mat[1] for mat in formatted_mats]
                    for i, f in enumerate(self.faces):
                        if len(f) > 3:
                            formatted_faces.append((f[0], f[1], f[2]))
                            formatted_faces.append((f[2], f[3], f[0]))
                            for j, m in enumerate(formatted_mats):
                                if m[1] > i:
                                    mat_ind[j] = mat_ind[j] + 1
                        else:
                            formatted_faces.append(f)
                    formatted_mats = \
                        [(mn[0], mi) for mn, mi in zip(formatted_mats, mat_ind)]

            # loop through the faces and get all lines of text for them
            face_txt = []
            if self.vertex_texture_map is None and self.vertex_normals is None:
                for f in formatted_faces:
                    face_txt.append('f ' + ' '.join(str(fi + 1) for fi in f) + '\n')
            else:
                if self.vertex_texture_map is not None and \
                        self.vertex_normals is not None:
                    f_map = '{0}/{0}/{0}'
                elif self.vertex_texture_map is None and \
                        self.vertex_normals is not None:
                    f_map = '{0}//{0}'
                else:
                    f_map = '{0}/{0}'
                for f in formatted_faces:
                    face_txt.append(
                        'f ' + ' '.join(f_map.format(fi + 1) for fi in f) + '\n'
                    )

            # write the faces into the file with the material structure
            if formatted_mats is not None:  # insert the materials
                for mat in reversed(formatted_mats):
                    face_txt.insert(mat[1], 'usemtl {}\n'.format(mat[0]))
            for f_lin in face_txt:
                outfile.write(f_lin)

        # write the MTL file if requested
        if include_mtl:
            mat_struct = [('diffuse_0', 0)] if self._material_structure is None else \
                self._material_structure
            mtl_fp = os.path.join(folder, mtl_file)
            with open(mtl_fp, writemode) as mtl_f:
                mtl_f.write('# Ladybug Geometry\n')
                for mat in reversed(mat_struct):
                    mtl_str =  \
                        'newmtl {}\n' \
                        'Ka 0.0000 0.0000 0.0000\n' \
                        'Kd 1.0000 1.0000 1.0000\n' \
                        'Ks 0.0000 0.0000 0.0000\n' \
                        'Tf 0.0000 0.0000 0.0000\n' \
                        'd 1.0000\n' \
                        'Ns 0.0000\n'.format(mat[0])
                    mtl_f.write(mtl_str)

        return obj_file

    def _check_vertices_input(self, vertices):
        """Check the input vertices."""
        if not isinstance(vertices, tuple):
            vertices = tuple(vertices)
        for vert in vertices:
            assert isinstance(vert, Point3D), \
                'Expected Point3D for OBJ vertex. Got {}.'.format(type(vert))
        return vertices

    def _check_faces_input(self, faces):
        """Check input faces for correct formatting."""
        if not isinstance(faces, tuple):
            faces = tuple(faces)
        assert len(faces) > 0, 'OBJ mesh must have at least one face.'
        for f in faces:
            assert isinstance(f, tuple), \
                'Expected tuple for Mesh face. Got {}.'.format(type(f))
            assert len(f) >= 3, \
                'OBJ mesh face must have 3 or more vertices. Got {}.'.format(len(f))
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

    def __len__(self):
        return len(self._vertices)

    def __getitem__(self, key):
        return self._vertices[key]

    def __iter__(self):
        return iter(self._vertices)

    def __repr__(self):
        return 'OBJ ({} vertices) ({} faces)'.format(
            len(self._vertices), len(self._faces))
