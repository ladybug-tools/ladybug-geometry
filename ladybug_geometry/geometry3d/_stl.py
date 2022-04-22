"""Support functions to turn stl into a Ladybug Mesh3D object.

The methods in the object below are inspired from pySTL module here:
https://github.com/proverbialsunrise/pySTL/blob/6afb180e14ee13fe313028cc9d9efa9104f7b1fe/pySTL.py
"""

import os
import struct
import warnings
from .pointvector import Point3D

try:
    from itertools import izip as zip  # python 2
except ImportError:
    xrange = range  # python 3


class STL:
    """A class to represent an STL object.

    Args:
        file_path: Path to the STL file.
    """

    def __init__(self, file_path):
        self._file_path = file_path
        self._face_vertices = []
        self._load_stl()

    @property
    def face_vertices(self):
        """Get a list of faces where each face is a list of vertices for the face."""
        return self._face_vertices

    def _load_stl(self):
        """Load the ASCII STL file and extract triangular faces."""

        assert os.path.isfile(self._file_path), 'Failed to find %s' % self._file_path

        with open(self._file_path, 'rb') as fp:
            h = fp.read(80)
            first_word = h[0:5]

        if first_word.decode('utf-8') == 'solid':
            self._load_text_stl()
            if len(self._face_vertices) < 1:
                warnings.warn('Failed to read ascii STL file. Trying binary read.')
                self._load_binary_stl()

        else:
            self._load_binary_stl()

        if len(self._face_vertices) < 1:
            raise ValueError(
                "No triangle found in the file. This may not be a valid .STL file.")

    def _load_text_stl(self):
        """read text stl file and extract triangular faces."""
        with open(self._file_path, 'r') as fp:
            for line in fp:
                words = line.split()
                if len(words) > 0:
                    if words[0] == 'solid':
                        try:
                            self.name = words[1]
                        except IndexError:
                            self.name = "polyhedron"

                    if words[0] == 'facet':
                        vertices = []

                    elif words[0] == 'vertex':
                        vertices.append(
                            Point3D(float(words[1]), float(words[2]), float(words[3])))

                    elif words[0] == 'endloop':
                        if len(vertices) == 3:
                            self._face_vertices.append(vertices)

    def _load_binary_stl(self):
        """read binary stl file and extract triangular faces."""

        with open(self._file_path, 'rb') as fp:

            h = fp.read(80)
            l = struct.unpack('I', fp.read(4))[0]
            count = 0
            while True:
                try:
                    p = fp.read(12)
                    if len(p) == 12:
                        n = struct.unpack('f', p[0:4])[0], struct.unpack(
                            'f', p[4:8])[0], struct.unpack('f', p[8:12])[0]

                    p = fp.read(12)
                    if len(p) == 12:
                        p1 = Point3D(struct.unpack('f', p[0:4])[0], struct.unpack(
                            'f', p[4:8])[0], struct.unpack('f', p[8:12])[0])

                    p = fp.read(12)
                    if len(p) == 12:
                        p2 = Point3D(struct.unpack('f', p[0:4])[0], struct.unpack(
                            'f', p[4:8])[0], struct.unpack('f', p[8:12])[0])

                    p = fp.read(12)
                    if len(p) == 12:
                        p3 = Point3D(struct.unpack('f', p[0:4])[0], struct.unpack(
                            'f', p[4:8])[0], struct.unpack('f', p[8:12])[0])

                    self._face_vertices.append(([p1, p2, p3]))
                    count += 1
                    fp.read(2)

                    if len(p) == 0:
                        break
                except EOFError:
                    break
