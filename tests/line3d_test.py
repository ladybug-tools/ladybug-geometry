# coding=utf-8

from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
from ladybug_geometry.geometry3d.line import LineSegment3D, LineSegment3DImmutable

import unittest
import pytest
import math


class LineSegment3DTestCase(unittest.TestCase):
    """Test for LineSegment3D"""

    def test_linesegment3d_init(self):
        """Test the initalization of LineSegment3D objects and basic properties."""
        pt = Point3D(2, 0, 2)
        vec = Vector3D(0, 2, 0)
        seg = LineSegment3D(pt, vec)
        str(seg)  # test the string representation of the line segment

        assert seg.p == Point3D(2, 0, 2)
        assert seg.v == Vector3D(0, 2, 0)
        assert seg.p1 == Point3D(2, 0, 2)
        assert seg.p2 == Point3D(2, 2, 2)
        assert seg.midpoint == Point3D(2, 1, 2)
        assert seg.point_at(0.25) == Point3D(2, 0.5, 2)
        assert seg.point_at_length(1) == Point3D(2, 1, 2)
        assert seg.length == 2

        flip_seg = seg.flipped()
        assert flip_seg.p == Point3D(2, 2, 2)
        assert flip_seg.v == Vector3D(0, -2, 0)

        assert seg.p == Point3D(2, 0, 2)
        seg.flip()
        assert seg.p == Point3D(2, 2, 2)
        assert seg.v == Vector3D(0, -2, 0)

    def test_init_from_endpoints(self):
        """Test the initalization of LineSegment3D from end points."""
        pt_1 = Point3D(2, 0, 2)
        pt_2 = Point3D(2, 2, 2)
        seg = LineSegment3D.from_end_points(pt_1, pt_2)

        assert seg.p == Point3D(2, 0, 2)
        assert seg.v == Vector3D(0, 2, 0)
        assert seg.p1 == Point3D(2, 0, 2)
        assert seg.p2 == Point3D(2, 2, 2)
        assert seg.length == 2

    def test_init_from_sdl(self):
        """Test the initalization of LineSegment3D from start, direction, length."""
        pt = Point3D(2, 0, 2)
        vec = Vector3D(0, 1, 0)
        seg = LineSegment3D.from_sdl(pt, vec, 2)

        assert seg.p == Point3D(2, 0, 2)
        assert seg.v == Vector3D(0, 2, 0)
        assert seg.p1 == Point3D(2, 0, 2)
        assert seg.p2 == Point3D(2, 2, 2)
        assert seg.length == 2

    def test_linesegment3d_mutability(self):
        """Test the mutability and immutability of LineSegment3D objects."""
        pt = Point3D(2, 0, 0)
        vec = Vector3D(0, 2, 0)
        seg = LineSegment3D(pt, vec)

        assert isinstance(seg, LineSegment3D)
        assert seg.is_mutable is True
        seg.p = Point3D(0, 0, 0)
        assert seg.p == Point3D(0, 0, 0)
        seg.p.x = 1
        assert seg.p == Point3D(1, 0, 0)

        seg_imm = seg.to_immutable()
        assert isinstance(seg_imm, LineSegment3DImmutable)
        assert seg_imm.is_mutable is False
        with pytest.raises(AttributeError):
            seg_imm.p.x = 3
        with pytest.raises(AttributeError):
            seg_imm.v.x = 3
        with pytest.raises(AttributeError):
            seg_imm.p = Point3D(0, 0, 0)
        with pytest.raises(AttributeError):
            seg_imm.v = Vector3D(2, 2, 0)
        seg_move = seg_imm.move(Vector3D(-1, 0, 0))  # ensure operations that yield new objects are ok
        assert seg_move.p == Point3D(0, 0, 0)

        seg = LineSegment3DImmutable(pt, vec)
        assert isinstance(seg, LineSegment3DImmutable)
        assert seg.is_mutable is False
        with pytest.raises(AttributeError):
            seg_imm.p.x = 3
        assert seg.p == Point3D(2, 0, 0)
        seg_copy = seg.duplicate()
        assert seg.p == seg_copy.p
        assert seg.v == seg_copy.v

        seg_mut = seg.to_mutable()
        assert isinstance(seg_mut, LineSegment3D)
        assert seg_mut.is_mutable is True
        seg_mut.p.x = 1
        assert seg_mut.p.x == 1

    def test_move(self):
        """Test the LineSegment3D move method."""
        pt = Point3D(2, 0, 2)
        vec = Vector3D(0, 2, 0)
        seg = LineSegment3D(pt, vec)

        vec_1 = Vector3D(2, 2, 2)
        new_seg = seg.move(vec_1)
        assert new_seg.p == Point3D(4, 2, 4)
        assert new_seg.v == vec
        assert new_seg.p1 == Point3D(4, 2, 4)
        assert new_seg.p2 == Point3D(4, 4, 4)

    def test_scale(self):
        """Test the LineSegment3D scale method."""
        pt = Point3D(2, 2, 2)
        vec = Vector3D(0, 2, 0)
        seg = LineSegment3D(pt, vec)

        origin_1 = Point3D(0, 2, 2)
        origin_2 = Point3D(1, 1, 2)
        new_seg = seg.scale(2, origin_1)
        assert new_seg.p == Point3D(4, 2, 2)
        assert new_seg.v == Point3D(0, 4, 0)
        assert new_seg.length == 4

        new_seg = seg.scale(2, origin_2)
        assert new_seg.p == Point3D(3, 3, 2)
        assert new_seg.v == Point3D(0, 4, 0)

    def test_scale_world_origin(self):
        """Test the LineSegment3D scale_world_origin method."""
        pt = Point3D(2, 2, 2)
        vec = Vector3D(0, 2, 0)
        seg = LineSegment3D(pt, vec)

        new_seg = seg.scale_world_origin(2)
        assert new_seg.p == Point3D(4, 4, 4)
        assert new_seg.v == Point3D(0, 4)
        assert new_seg.length == 4

    def test_rotate(self):
        """Test the LineSegment3D rotate method."""
        pt = Point3D(2, 2, 2)
        vec = Vector3D(0, 2, 0)
        seg = LineSegment3D(pt, vec)
        origin_1 = Point3D(0, 0, 0)
        axis_1 = Vector3D(1, 0, 0)

        test_1 = seg.rotate(axis_1, math.pi, origin_1)
        assert test_1.p.x == pytest.approx(2, rel=1e-3)
        assert test_1.p.y == pytest.approx(-2, rel=1e-3)
        assert test_1.p.z == pytest.approx(-2, rel=1e-3)
        assert test_1.v.x == pytest.approx(0, rel=1e-3)
        assert test_1.v.y == pytest.approx(-2, rel=1e-3)
        assert test_1.v.z == pytest.approx(0, rel=1e-3)

        test_2 = seg.rotate(axis_1, math.pi/2, origin_1)
        assert test_2.p.x == pytest.approx(2, rel=1e-3)
        assert test_2.p.y == pytest.approx(-2, rel=1e-3)
        assert test_2.p.z == pytest.approx(2, rel=1e-3)
        assert test_2.v.x == pytest.approx(0, rel=1e-3)
        assert test_2.v.y == pytest.approx(0, rel=1e-3)
        assert test_2.v.z == pytest.approx(2, rel=1e-3)

    def test_rotate_xy(self):
        """Test the LineSegment3D rotate_xy method."""
        pt = Point3D(2, 2, 2)
        vec = Vector3D(0, 2, 0)
        seg = LineSegment3D(pt, vec)
        origin_1 = Point3D(0, 2, 2)

        test_1 = seg.rotate_xy(math.pi, origin_1)
        assert test_1.p.x == pytest.approx(-2, rel=1e-3)
        assert test_1.p.y == pytest.approx(2, rel=1e-3)
        assert test_1.v.x == pytest.approx(0, rel=1e-3)
        assert test_1.v.y == pytest.approx(-2, rel=1e-3)

        test_2 = seg.rotate_xy(math.pi/2, origin_1)
        assert test_2.p.x == pytest.approx(0, rel=1e-3)
        assert test_2.p.y == pytest.approx(4, rel=1e-3)
        assert test_2.v.x == pytest.approx(-2, rel=1e-3)
        assert test_2.v.y == pytest.approx(0, rel=1e-3)

    def test_reflect(self):
        """Test the LineSegment3D reflect method."""
        pt = Point3D(2, 2, 2)
        vec = Vector3D(0, 2, 0)
        seg = LineSegment3D(pt, vec)

        origin_1 = Point3D(0, 1, 2)
        origin_2 = Point3D(1, 1, 2)
        normal_1 = Vector3D(0, 1, 0)
        normal_2 = Vector3D(-1, 1, 0).normalized()

        assert seg.reflect(normal_1, origin_1).p == Point3D(2, 0, 2)
        assert seg.reflect(normal_1, origin_1).v == Vector3D(0, -2, 0)
        assert seg.reflect(normal_1, origin_2).p == Point3D(2, 0, 2)
        assert seg.reflect(normal_1, origin_2).v == Vector3D(0, -2, 0)

        test_1 = seg.reflect(normal_2, origin_2)
        assert test_1.p == Point3D(2, 2, 2)
        assert test_1.v.x == pytest.approx(2, rel=1e-3)
        assert test_1.v.y == pytest.approx(0, rel=1e-3)
        assert test_1.v.z == pytest.approx(0, rel=1e-3)

        test_2 = seg.reflect(normal_2, origin_1)
        assert test_2.p.x == pytest.approx(1, rel=1e-3)
        assert test_2.p.y == pytest.approx(3, rel=1e-3)
        assert test_2.p.z == pytest.approx(2, rel=1e-3)
        assert test_2.v.x == pytest.approx(2, rel=1e-3)
        assert test_2.v.y == pytest.approx(0, rel=1e-3)
        assert test_2.v.z == pytest.approx(0, rel=1e-3)

    def test_subdivide(self):
        """Test the LineSegment3D subdivide methods."""
        pt = Point3D(2, 2, 2)
        vec = Vector3D(0, 2, 0)
        seg = LineSegment3D(pt, vec)

        divisions = seg.subdivide(0.5)
        assert len(divisions) == 5
        assert divisions[0] == pt
        assert divisions[1] == Point3D(2, 2.5, 2)
        assert divisions[2] == Point3D(2, 3, 2)
        assert divisions[3] == Point3D(2, 3.5, 2)
        assert divisions[4] == Point3D(2, 4, 2)

        divisions = seg.subdivide([1, 0.5, 0.25])
        assert len(divisions) == 5
        assert divisions[0] == pt
        assert divisions[1] == Point3D(2, 3, 2)
        assert divisions[2] == Point3D(2, 3.5, 2)
        assert divisions[3] == Point3D(2, 3.75, 2)
        assert divisions[4] == Point3D(2, 4, 2)

        divisions = seg.subdivide_evenly(4)
        assert len(divisions) == 5
        assert divisions[0] == pt
        assert divisions[1] == Point3D(2, 2.5, 2)
        assert divisions[2] == Point3D(2, 3, 2)
        assert divisions[3] == Point3D(2, 3.5, 2)
        assert divisions[4] == Point3D(2, 4, 2)

    def test_closest_point(self):
        """Test the LineSegment3D closest_point method."""
        pt = Point3D(2, 2, 2)
        vec = Vector3D(0, 2, 0)
        seg = LineSegment3D(pt, vec)

        near_pt = Point3D(3, 3, 0)
        assert seg.closest_point(near_pt) == Point3D(2, 3, 2)
        near_pt = Point3D(2, 0, 0)
        assert seg.closest_point(near_pt) == Point3D(2, 2, 2)
        near_pt = Point3D(2, 5, 0)
        assert seg.closest_point(near_pt) == Point3D(2, 4, 2)

    def test_distance_to_point(self):
        """Test the LineSegment3D distance_to_point method."""
        pt = Point3D(2, 2, 2)
        vec = Vector3D(0, 2, 0)
        seg = LineSegment3D(pt, vec)

        near_pt = Point3D(3, 3, 2)
        assert seg.distance_to_point(near_pt) == 1
        near_pt = Point3D(2, 0, 2)
        assert seg.distance_to_point(near_pt) == 2
        near_pt = Point3D(2, 5, 2)
        assert seg.distance_to_point(near_pt) == 1


if __name__ == "__main__":
    unittest.main()
