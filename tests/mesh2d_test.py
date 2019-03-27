# coding=utf-8

from ladybug_geometry.geometry2d.pointvector import Point2D
from ladybug_geometry.geometry2d.mesh import Mesh2D

import unittest
import pytest



class Mesh2DTestCase(unittest.TestCase):
    """Test for Mesh2D"""

    def test_polygon2_init(self):
        """Test the initalization of Mesh2D objects and basic properties."""
        pts = [Point2D(0, 0), Point2D(0, 2), Point2D(2, 2), Point2D(2, 0)]
        mesh = Mesh2D(pts, [(0, 1, 2, 3)])
        str(mesh)  # test the string representation of the object

        assert len(mesh.vertices) == 4
        assert len(mesh.faces) == 1
        assert mesh[0] == Point2D(0, 0)
        assert mesh[1] == Point2D(0, 2)
        assert mesh[2] == Point2D(2, 2)
        assert mesh[3] == Point2D(2, 0)
        assert mesh.area == 4


if __name__ == "__main__":
    unittest.main()
