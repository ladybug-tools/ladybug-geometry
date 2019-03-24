# coding=utf-8

from ladybug_geometry.geometry2d.pointvector import Point2D, Vector2D
from ladybug_geometry.geometry2d.line import LineSegment2D, LineSegment2DImmutable

import unittest
import pytest
import math


class LineSegment2DTestCase(unittest.TestCase):
    """Test for LineSegment2D"""

    def test_linesegment2_init(self):
        """Test the initalization of LineSegement2 objects and basic properties."""
        pt = Point2D(2, 0)
        vec = Vector2D(0, 2)
        seg = LineSegment2D(pt, vec)
        str(seg)  # test the string representation of the line segment

        assert seg.p == Point2D(2, 0)
        assert seg.v == Vector2D(0, 2)
        assert seg.p1 == Point2D(2, 0)
        assert seg.p2 == Point2D(2, 2)
        assert seg.length == 2
        assert seg.length_squared == 4

        flip_seg = seg.flipped()
        assert flip_seg.p == Point2D(2, 2)
        assert flip_seg.v == Vector2D(0, -2)

        assert seg.p == Point2D(2, 0)
        seg.flip()
        assert seg.p == Point2D(2, 2)
        assert seg.v == Vector2D(0, -2)

    def test_init_from_endpoints(self):
        """Test the initalization of LineSegement2 from end points."""
        pt_1 = Point2D(2, 0)
        pt_2 = Point2D(2, 2)
        seg = LineSegment2D.from_end_points(pt_1, pt_2)

        assert seg.p == Point2D(2, 0)
        assert seg.v == Vector2D(0, 2)
        assert seg.p1 == Point2D(2, 0)
        assert seg.p2 == Point2D(2, 2)
        assert seg.length == 2
        assert seg.length_squared == 4

    def test_init_from_sdl(self):
        """Test the initalization of LineSegement2 from start, direction, length."""
        pt = Point2D(2, 0)
        vec = Vector2D(0, 1)
        seg = LineSegment2D.from_sdl(pt, vec, 2)

        assert seg.p == Point2D(2, 0)
        assert seg.v == Vector2D(0, 2)
        assert seg.p1 == Point2D(2, 0)
        assert seg.p2 == Point2D(2, 2)
        assert seg.length == 2
        assert seg.length_squared == 4

    def test_linesegment2_mutability(self):
        """Test the mutability and immutability of LineSegement2 objects."""
        pt = Point2D(2, 0)
        vec = Vector2D(0, 2)
        seg = LineSegment2D(pt, vec)

        assert isinstance(seg, LineSegment2D)
        assert seg.is_mutable is True
        seg.p = Point2D(0, 0)
        assert seg.p == Point2D(0, 0)
        seg.p.x = 1
        assert seg.p == Point2D(1, 0)

        seg_imm = seg.to_immutable()
        assert isinstance(seg_imm, LineSegment2DImmutable)
        assert seg_imm.is_mutable is False
        with pytest.raises(AttributeError):
            seg_imm.p.x = 3
        with pytest.raises(AttributeError):
            seg_imm.v.x = 3
        with pytest.raises(AttributeError):
            seg_imm.p = Point2D(0, 0)
        with pytest.raises(AttributeError):
            seg_imm.v = Vector2D(2, 2)
        seg_move = seg_imm.move(Vector2D(-1, 0))  # ensure operations that yield new objects are ok
        assert seg_move.p == Point2D(0, 0)

        seg = LineSegment2DImmutable(pt, vec)
        assert isinstance(seg, LineSegment2DImmutable)
        assert seg.is_mutable is False
        with pytest.raises(AttributeError):
            seg_imm.p.x = 3
        assert seg.p == Point2D(2, 0)
        seg_copy = seg.duplicate()
        assert seg.p == seg_copy.p
        assert seg.v == seg_copy.v

        seg_mut = seg.to_mutable()
        assert isinstance(seg_mut, LineSegment2D)
        assert seg_mut.is_mutable is True
        seg_mut.p.x = 1
        assert seg_mut.p.x == 1

    def test_move(self):
        """Test the LineSegement2 move method."""
        pt = Point2D(2, 0)
        vec = Vector2D(0, 2)
        seg = LineSegment2D(pt, vec)

        vec_1 = Vector2D(2, 2)
        new_seg = seg.move(vec_1)
        assert new_seg.p == Point2D(4, 2)
        assert new_seg.v == vec
        assert new_seg.p1 == Point2D(4, 2)
        assert new_seg.p2 == Point2D(4, 4)

    def test_scale(self):
        """Test the LineSegement2 scale method."""
        pt = Point2D(2, 2)
        vec = Vector2D(0, 2)
        seg = LineSegment2D(pt, vec)

        origin_1 = Point2D(0, 2)
        origin_2 = Point2D(1, 1)
        new_seg = seg.scale(2, origin_1)
        assert new_seg.p == Point2D(4, 2)
        assert new_seg.v == Point2D(0, 4)
        assert new_seg.length == 4

        new_seg = seg.scale(2, origin_2)
        assert new_seg.p == Point2D(3, 3)
        assert new_seg.v == Point2D(0, 4)

    def test_rotate(self):
        """Test the LineSegement2 rotate method."""
        pt = Point2D(2, 2)
        vec = Vector2D(0, 2)
        seg = LineSegment2D(pt, vec)
        origin_1 = Point2D(0, 2)

        test_1 = seg.rotate(math.pi, origin_1)
        assert test_1.p.x == pytest.approx(-2, rel=1e-3)
        assert test_1.p.y == pytest.approx(2, rel=1e-3)
        assert test_1.v.x == pytest.approx(0, rel=1e-3)
        assert test_1.v.y == pytest.approx(-2, rel=1e-3)

        test_2 = seg.rotate(math.pi/2, origin_1)
        assert test_2.p.x == pytest.approx(0, rel=1e-3)
        assert test_2.p.y == pytest.approx(4, rel=1e-3)
        assert test_2.v.x == pytest.approx(-2, rel=1e-3)
        assert test_2.v.y == pytest.approx(0, rel=1e-3)

    def test_reflect(self):
        """Test the Point2D reflect method."""
        pt = Point2D(2, 2)
        vec = Vector2D(0, 2)
        seg = LineSegment2D(pt, vec)

        origin_1 = Point2D(0, 1)
        origin_2 = Point2D(1, 1)
        normal_1 = Vector2D(0, 1)
        normal_2 = Vector2D(-1, 1).normalized()

        assert seg.reflect(normal_1, origin_1).p == Point2D(2, 0)
        assert seg.reflect(normal_1, origin_1).v == Vector2D(0, -2)
        assert seg.reflect(normal_1, origin_2).p == Point2D(2, 0)
        assert seg.reflect(normal_1, origin_2).v == Vector2D(0, -2)

        test_1 = seg.reflect(normal_2, origin_2)
        assert test_1.p == Point2D(2, 2)
        assert test_1.v.x == pytest.approx(2, rel=1e-3)
        assert test_1.v.y == pytest.approx(0, rel=1e-3)

        test_2 = seg.reflect(normal_2, origin_1)
        assert test_2.p.x == pytest.approx(1, rel=1e-3)
        assert test_2.p.y == pytest.approx(3, rel=1e-3)
        assert test_1.v.x == pytest.approx(2, rel=1e-3)
        assert test_1.v.y == pytest.approx(0, rel=1e-3)


if __name__ == "__main__":
    unittest.main()
