# coding=utf-8
"""A class that supports the import and export of STL data to/from ladybug_geometry.

The methods in the object below are inspired from pySTL module.

[1] Daniel Balzerson. 2013. pySTL - Python code for working with .STL
(sterolithography) files. https://github.com/proverbialsunrise/pySTL
"""
import os
import struct
import re

try:
    from itertools import izip as zip  # python 2
    writemode = 'wb'
except ImportError:
    writemode = 'w'  # python 3

from ladybug_geometry.geometry3d.pointvector import Vector3D, Point3D


class STL(object):
    """A class that supports the import and export of STL data to/from ladybug_geometry.

    Args:
        face_vertices: A list of tuples where each tuple is a triangular face
            of three Point3Ds.
        face_normals: A list of Vector3Ds for the normals of the faces in the STL.
        name: Text string for the name of the solid object in the STL
            file. (Default: polyhedron).

    Properties:
        * name
        * face_vertices
        * face_normals
    """

    __slots__ = ('_name', '_face_vertices', '_face_normals')

    def __init__(self, face_vertices, face_normals, name='polyhedron'):
        self.name = name
        self._face_normals = None  # bypass check on first time
        self.face_vertices = face_vertices
        self.face_normals = face_normals

    @classmethod
    def from_file(cls, file_path):
        """Create an STL object from a .stl file.

        Args:
            file_path: Path to an STL file as a text string. The STL file can be
                in either ASCII or binary format.
        """
        face_vertices, face_normals, name = cls._load_stl(file_path)
        return cls(face_vertices, face_normals, name)

    @classmethod
    def from_mesh3d(cls, mesh, name='polyhedron'):
        """Create an STL object from a ladybug_geometry Mesh3D object.

        All quad faces will be automatically triangulated by using this method.

        Args:
            mesh: A ladybug_geometry Mesh3D object to be converted to an OBJ object.
            name: Text string for the name of the solid object in the STL
                file. (Default: polyhedron).
        """
        face_vertices, face_normals = [], []
        for f, fn in zip(mesh.faces, mesh.face_normals):
            if len(f) == 3:
                face_vertices.append(tuple(mesh._vertices[i] for i in f))
                face_normals.append(fn)
            else:  # it's a quad mesh to be triangulated
                vts1 = (mesh._vertices[f[0]], mesh._vertices[f[1]], mesh._vertices[f[2]])
                vts2 = (mesh._vertices[f[2]], mesh._vertices[f[3]], mesh._vertices[f[0]])
                face_vertices.append(vts1)
                face_vertices.append(vts2)
                face_normals.append(fn)
                face_normals.append(fn)
        return cls(face_vertices, face_normals, name)

    @property
    def name(self):
        """Get the name of the solid object in the STL file."""
        return self._name

    @name.setter
    def name(self, value):
        input_name = 'STL object name'
        try:
            non_ascii = tuple(i for i in value if ord(i) >= 128)
        except TypeError:
            raise TypeError('Input {} must be a text string. Got {}: {}.'.format(
                input_name, type(value), value))
        assert non_ascii == (), 'Illegal characters {} found in {}'.format(
            non_ascii, input_name)
        illegal_match = re.search(r'[,;!\n\t]', value)
        assert illegal_match is None, 'Illegal character "{}" found in {}'.format(
            illegal_match.group(0), input_name)
        assert len(value) > 0, 'Input {} "{}" contains no characters.'.format(
            input_name, value)
        assert len(value) <= 80, 'Input {} "{}" must be less than 80 characters.'.format(
            input_name, value)
        self._name = value

    @property
    def face_vertices(self):
        """Get a list of tuples where each tuple is a triangular face of three Point3Ds.
        """
        return self._face_vertices

    @face_vertices.setter
    def face_vertices(self, val):
        assert isinstance(val, (list, tuple)), \
            'face_vertices should be a list or tuple. Got {}'.format(type(val))
        if isinstance(val, list):
            val = tuple(val)
        for f in val:
            assert len(f) == 3, 'All face_vertices of an STL must be triangles. ' \
                'Got face of length {}.'.format(len(f))
        self._face_vertices = val
        self._check_faces_match()

    @property
    def face_normals(self):
        """Get a list of Vector3Ds for the normals of the faces in the STL."""
        return self._face_normals

    @face_normals.setter
    def face_normals(self, val):
        assert isinstance(val, (list, tuple)), \
            'face_normals should be a list or tuple. Got {}'.format(type(val))
        if isinstance(val, list):
            val = tuple(val)
        self._face_normals = val
        self._check_faces_match()

    def to_file(self, folder, name=None):
        """Write the STL object to an ASCII STL file.

        Args:
            folder: A text string for the directory where the STL will be written.
            name: A text string for the name of the STL file. If None, the name
                of the STL object will be used. (Default: None).
        """
        # set up a name and folder
        if name is None:
            name = self.name
        file_name = name if name.lower().endswith('.stl') else '{}.stl'.format(name)
        stl_file = os.path.join(folder, file_name)
        # loop through the faces and normals to write them to the file
        with open(stl_file, writemode) as fp:
            fp.write('solid {:s}\n'.format(self.name))
            for facet, nm in zip(self.face_vertices, self.face_normals):
                fp.write(
                    '  facet normal {0:.6E} {1:.6E} {2:.6E}\n'.format(nm.x, nm.y, nm.z)
                )
                fp.write('    outer loop\n')
                for pt in facet:
                    fp.write(
                        '      vertex {0:.6E} {1:.6E} {2:.6E}\n'.format(pt.x, pt.y, pt.z)
                    )
                fp.write('    endloop\n')
                fp.write('  endfacet\n')
            fp.write('endsolid {:s}\n'.format(self.name))
        return stl_file

    def _check_faces_match(self):
        if self._face_normals is not None:
            assert len(self._face_vertices) == len(self._face_normals), \
                'Number of STL face_vertices ({}) does not match the number ' \
                'of _face_normals ({}).'.format(
                    len(self._face_vertices), len(self._face_normals))

    @staticmethod
    def _load_stl(file_path):
        """Load data from an STL file.

        Args:
            file_path: Path to an STL file as a text string. The STL file can be
                in either ASCII or binary format.

        Returns:
            A tuple with three elements.

            -   face_vertices:
                A list of tuples where each tuple is a triangular face of three Point3Ds.

            -   face_normals:
                A list of Vector3Ds for the normals of the faces in the STL.

            -   name:
                Text string for the name of the STL object
        """
        # check the first bytes of the file to determine whether it's ASCII or binary
        assert os.path.isfile(file_path), 'Failed to find %s' % file_path
        with open(file_path, 'rb') as fp:
            header = fp.read(80)  # 80 characters should have the full name
            first_word = header[0:5]
        # load the STL data depending on whether it is ASCII or binary
        if first_word.decode('utf-8') == 'solid':
            return STL._load_text_stl(file_path)
        else:
            return STL._load_binary_stl(file_path)

    @staticmethod
    def _load_text_stl(file_path):
        """Read text stl file and extract triangular faces."""
        _face_vertices, _face_normals, _name = [], [], 'polyhedron'
        with open(file_path, 'r') as fp:
            for line in fp:
                words = line.split()
                if len(words) > 0:
                    first_word = words[0]
                    if first_word == 'facet':  # start of a new face
                        vertices = []
                        norm = Vector3D(
                            float(words[2]), float(words[3]), float(words[4])
                        )
                        _face_normals.append(norm)
                    elif first_word == 'vertex':  # vertex of a face
                        vertices.append(
                            Point3D(float(words[1]), float(words[2]), float(words[3]))
                        )
                    elif first_word == 'endloop':  # end of a face
                        _face_vertices.append(tuple(vertices))
                    elif first_word == 'solid':  # very start of the file
                        try:
                            _name = words[1]
                        except IndexError:  # no name assigned; leave the default one
                            pass
        return _face_vertices, _face_normals, _name

    @staticmethod
    def _load_binary_stl(file_path):
        """Read binary stl file and extract triangular faces."""
        _face_vertices, _face_normals, _name = [], [], 'polyhedron'
        with open(file_path, 'rb') as fp:
            # interpret the 80-character header as the name of the object
            _name = fp.read(80).decode('utf-8').strip()
            # ignore the total face count in the first 4 characters
            struct.unpack('I', fp.read(4))[0]

            # loop through the file contents and load all vertices and vectors
            count = 0
            while True:
                try:
                    # read the face normal
                    p = fp.read(12)
                    if len(p) == 12:
                        norm = Vector3D(
                            struct.unpack('f', p[0:4])[0],
                            struct.unpack('f', p[4:8])[0],
                            struct.unpack('f', p[8:12])[0]
                        )
                    else:
                        break
                    # read the first vertex
                    p = fp.read(12)
                    if len(p) == 12:
                        p1 = Point3D(
                            struct.unpack('f', p[0:4])[0],
                            struct.unpack('f', p[4:8])[0],
                            struct.unpack('f', p[8:12])[0]
                        )
                    else:
                        break
                    # read the second vertex
                    p = fp.read(12)
                    if len(p) == 12:
                        p2 = Point3D(
                            struct.unpack('f', p[0:4])[0],
                            struct.unpack('f', p[4:8])[0],
                            struct.unpack('f', p[8:12])[0]
                        )
                    else:
                        break
                    # read the third vertex
                    p = fp.read(12)
                    if len(p) == 12:
                        p3 = Point3D(
                            struct.unpack('f', p[0:4])[0],
                            struct.unpack('f', p[4:8])[0],
                            struct.unpack('f', p[8:12])[0]
                        )
                    else:
                        break
                    # add the triangle to the face vertices
                    _face_vertices.append((p1, p2, p3))
                    _face_normals.append(norm)
                    count += 1
                    fp.read(2)

                    if len(p) == 0:
                        break  # no more points to read
                except EOFError:
                    break  # we have reached the end of the file
        return _face_vertices, _face_normals, _name

    def __len__(self):
        return len(self._face_vertices)

    def __getitem__(self, key):
        return self._face_vertices[key]

    def __iter__(self):
        return iter(self._face_vertices)

    def __repr__(self):
        return 'STL ({} faces)'.format(len(self._face_vertices))
