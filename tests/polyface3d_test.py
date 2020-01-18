# coding=utf-8
import pytest

from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
from ladybug_geometry.geometry3d.plane import Plane
from ladybug_geometry.geometry3d.ray import Ray3D
from ladybug_geometry.geometry3d.line import LineSegment3D
from ladybug_geometry.geometry3d.face import Face3D
from ladybug_geometry.geometry3d.polyface import Polyface3D

import math


def test_polyface3d_init_solid():
    """Test the initalization of Poyface3D and basic properties of solid objects."""
    pts = [Point3D(0, 0, 0), Point3D(0, 2, 0), Point3D(2, 2, 0), Point3D(2, 0, 0),
           Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2)]
    face_indices = [[(0, 1, 2, 3)], [(0, 4, 5, 1)], [(0, 3, 7, 4)],
                    [(2, 1, 5, 6)], [(2, 3, 7, 6)], [(4, 5, 6, 7)]]
    polyface = Polyface3D(pts, face_indices)

    assert len(polyface.vertices) == 8
    assert len(polyface.face_indices) == 6
    assert len(polyface.faces) == 6
    assert len(polyface.edge_indices) == 12
    assert len(polyface.edges) == 12
    assert len(polyface.naked_edges) == 0
    assert len(polyface.non_manifold_edges) == 0
    assert len(polyface.internal_edges) == 12
    assert polyface.area == 24
    assert polyface.volume == 8
    assert polyface.is_solid

    for face in polyface.faces:
        assert face.area == 4
        assert face.is_clockwise is False
    assert polyface.faces[0].normal == Vector3D(0, 0, -1)
    assert polyface.faces[1].normal == Vector3D(-1, 0, 0)
    assert polyface.faces[2].normal == Vector3D(0, -1, 0)
    assert polyface.faces[3].normal == Vector3D(0, 1, 0)
    assert polyface.faces[4].normal == Vector3D(1, 0, 0)
    assert polyface.faces[5].normal == Vector3D(0, 0, 1)


def test_equality():
    """Test the equality of Polyface3D objects."""
    pts = [Point3D(0, 0, 0), Point3D(0, 2, 0), Point3D(2, 2, 0), Point3D(2, 0, 0),
           Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2)]
    face_indices = [[(0, 1, 2, 3)], [(0, 4, 5, 1)], [(0, 3, 7, 4)],
                    [(2, 1, 5, 6)], [(2, 3, 7, 6)], [(4, 5, 6, 7)]]
    face_indices_2 = [[(0, 1, 2, 3)], [(0, 4, 5, 1)], [(0, 3, 7, 4)],
                      [(2, 1, 5, 6)], [(2, 3, 7, 6)]]
    polyface = Polyface3D(pts, face_indices)
    polyface_dup = polyface.duplicate()
    polyface_alt = Polyface3D(pts, face_indices_2)

    assert polyface is polyface
    assert polyface is not polyface_dup
    assert polyface == polyface_dup
    assert hash(polyface) == hash(polyface_dup)
    assert polyface != polyface_alt
    assert hash(polyface) != hash(polyface_alt)


def test_polyface3d_init_open():
    """Test the initalization of Poyface3D and basic properties of open objects."""
    pts = [Point3D(0, 0, 0), Point3D(0, 2, 0), Point3D(2, 2, 0), Point3D(2, 0, 0),
           Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2)]
    face_indices = [[(0, 1, 2, 3)], [(0, 4, 5, 1)], [(0, 3, 7, 4)],
                    [(2, 1, 5, 6)], [(2, 3, 7, 6)]]
    polyface = Polyface3D(pts, face_indices)

    assert len(polyface.vertices) == 8
    assert len(polyface.face_indices) == 5
    assert len(polyface.faces) == 5
    assert len(polyface.edge_indices) == 12
    assert len(polyface.edges) == 12
    assert len(polyface.naked_edges) == 4
    assert len(polyface.non_manifold_edges) == 0
    assert len(polyface.internal_edges) == 8
    assert polyface.area == 20
    assert polyface.volume == 0
    assert polyface.is_solid is False

    for face in polyface.faces:
        assert face.area == 4
        assert face.is_clockwise is False


def test_polyface3d_init_from_faces_solid():
    """Test the initalization of Poyface3D from_faces with a solid."""
    pts_1 = [Point3D(0, 0, 0), Point3D(0, 2, 0), Point3D(2, 2, 0), Point3D(2, 0, 0)]
    pts_2 = [Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(0, 2, 0)]
    pts_3 = [Point3D(0, 0, 0), Point3D(2, 0, 0), Point3D(2, 0, 2), Point3D(0, 0, 2)]
    pts_4 = [Point3D(2, 2, 0), Point3D(0, 2, 0), Point3D(0, 2, 2), Point3D(2, 2, 2)]
    pts_5 = [Point3D(2, 2, 0), Point3D(2, 0, 0), Point3D(2, 0, 2), Point3D(2, 2, 2)]
    pts_6 = [Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2)]
    face_1 = Face3D(pts_1)
    face_2 = Face3D(pts_2)
    face_3 = Face3D(pts_3)
    face_4 = Face3D(pts_4)
    face_5 = Face3D(pts_5)
    face_6 = Face3D(pts_6)
    polyface = Polyface3D.from_faces(
        [face_1, face_2, face_3, face_4, face_5, face_6])

    assert len(polyface.vertices) == 8
    assert len(polyface.face_indices) == 6
    assert len(polyface.faces) == 6
    assert len(polyface.edge_indices) == 12
    assert len(polyface.edges) == 12
    assert len(polyface.naked_edges) == 0
    assert len(polyface.non_manifold_edges) == 0
    assert len(polyface.internal_edges) == 12
    assert polyface.area == 24
    assert polyface.volume == 8
    assert polyface.is_solid

    for face in polyface.faces:
        assert face.area == 4
        assert face.is_clockwise is False
    assert polyface.faces[0].normal == Vector3D(0, 0, -1)
    assert polyface.faces[1].normal == Vector3D(-1, 0, 0)
    assert polyface.faces[2].normal == Vector3D(0, -1, 0)
    assert polyface.faces[3].normal == Vector3D(0, 1, 0)
    assert polyface.faces[4].normal == Vector3D(1, 0, 0)
    assert polyface.faces[5].normal == Vector3D(0, 0, 1)


def test_polyface3d_init_from_faces_open():
    """Test the initalization of Poyface3D from_faces with an open object."""
    pts_1 = [Point3D(0, 0, 0), Point3D(0, 2, 0), Point3D(2, 2, 0), Point3D(2, 0, 0)]
    pts_2 = [Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(0, 2, 0)]
    pts_3 = [Point3D(0, 0, 0), Point3D(2, 0, 0), Point3D(2, 0, 2), Point3D(0, 0, 2)]
    pts_4 = [Point3D(2, 2, 0), Point3D(0, 2, 0), Point3D(0, 2, 2), Point3D(2, 2, 2)]
    pts_5 = [Point3D(2, 2, 0), Point3D(2, 0, 0), Point3D(2, 0, 2), Point3D(2, 2, 2)]
    face_1 = Face3D(pts_1)
    face_2 = Face3D(pts_2)
    face_3 = Face3D(pts_3)
    face_4 = Face3D(pts_4)
    face_5 = Face3D(pts_5)
    polyface = Polyface3D.from_faces([face_1, face_2, face_3, face_4, face_5])

    assert len(polyface.vertices) == 8
    assert len(polyface.face_indices) == 5
    assert len(polyface.faces) == 5
    assert len(polyface.edge_indices) == 12
    assert len(polyface.edges) == 12
    assert len(polyface.naked_edges) == 4
    assert len(polyface.non_manifold_edges) == 0
    assert len(polyface.internal_edges) == 8
    assert polyface.area == 20
    assert polyface.volume == 0
    assert polyface.is_solid is False

    for face in polyface.faces:
        assert face.area == 4
        assert face.is_clockwise is False


def test_polyface3d_init_from_faces_coplanar():
    """Test the initalization of Poyface3D from_faces with two coplanar faces."""
    # this is an important case that must be solved
    # can be done by iterating through naked edges and finding colinear ones
    pts_1 = [Point3D(0, 0, 0), Point3D(0, 2, 0), Point3D(2, 2, 0), Point3D(2, 0, 0)]
    pts_2 = [Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(0, 2, 0)]
    pts_3 = [Point3D(0, 0, 0), Point3D(2, 0, 0), Point3D(2, 0, 2), Point3D(0, 0, 2)]
    pts_4 = [Point3D(2, 2, 0), Point3D(0, 2, 0), Point3D(0, 2, 2), Point3D(2, 2, 2)]
    pts_5 = [Point3D(2, 2, 0), Point3D(2, 0, 0), Point3D(2, 0, 2), Point3D(2, 2, 2)]
    pts_6 = [Point3D(0, 0, 2), Point3D(0, 1, 2), Point3D(2, 1, 2), Point3D(2, 0, 2)]
    pts_7 = [Point3D(0, 1, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 1, 2)]
    face_1 = Face3D(pts_1)
    face_2 = Face3D(pts_2)
    face_3 = Face3D(pts_3)
    face_4 = Face3D(pts_4)
    face_5 = Face3D(pts_5)
    face_6 = Face3D(pts_6)
    face_7 = Face3D(pts_7)
    polyface = Polyface3D.from_faces(
        [face_1, face_2, face_3, face_4, face_5, face_6, face_7])
    polyface_2 = Polyface3D.from_faces(
        [face_1, face_2, face_3, face_4, face_5, face_7])

    assert not polyface.is_solid
    assert len(polyface.naked_edges) != 0

    new_polyface = polyface.merge_overlapping_edges(0.0001, 0.0001)
    assert new_polyface.is_solid
    assert len(new_polyface.naked_edges) == 0
    assert len(new_polyface.internal_edges) == 13

    new_polyface_2 = polyface_2.merge_overlapping_edges(0.0001, 0.0001)
    assert not new_polyface_2.is_solid
    assert len(new_polyface_2.naked_edges) != 0


def test_polyface3d_init_from_faces_coplanar_3face():
    """Test the initalization of Poyface3D from_faces with three coplanar faces."""
    # this is an important case that must be solved
    # can be done by iterating through naked edges and finding colinear ones
    pts_1 = [Point3D(0, 0, 0), Point3D(0, 2, 0), Point3D(2, 2, 0), Point3D(2, 0, 0)]
    pts_2 = [Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(0, 2, 0)]
    pts_3 = [Point3D(0, 0, 0), Point3D(2, 0, 0), Point3D(2, 0, 2), Point3D(0, 0, 2)]
    pts_4 = [Point3D(2, 2, 0), Point3D(0, 2, 0), Point3D(0, 2, 2), Point3D(2, 2, 2)]
    pts_5 = [Point3D(2, 2, 0), Point3D(2, 0, 0), Point3D(2, 0, 2), Point3D(2, 2, 2)]
    pts_6 = [Point3D(0, 0, 2), Point3D(0, 0.5, 2), Point3D(2, 0.5, 2), Point3D(2, 0, 2)]
    pts_7 = [Point3D(0, 0.5, 2), Point3D(0, 1, 2), Point3D(2, 1, 2), Point3D(2, 0.5, 2)]
    pts_8 = [Point3D(0, 1, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 1, 2)]
    face_1 = Face3D(pts_1)
    face_2 = Face3D(pts_2)
    face_3 = Face3D(pts_3)
    face_4 = Face3D(pts_4)
    face_5 = Face3D(pts_5)
    face_6 = Face3D(pts_6)
    face_7 = Face3D(pts_7)
    face_8 = Face3D(pts_8)
    polyface = Polyface3D.from_faces(
        [face_1, face_2, face_3, face_4, face_5, face_6, face_7, face_8])
    polyface_2 = Polyface3D.from_faces(
        [face_1, face_2, face_3, face_4, face_5, face_6, face_8])

    assert not polyface.is_solid
    assert len(polyface.naked_edges) != 0

    new_polyface = polyface.merge_overlapping_edges(0.0001, 0.0001)
    assert new_polyface.is_solid
    assert len(new_polyface.naked_edges) == 0
    assert len(new_polyface.internal_edges) == 14

    new_polyface_2 = polyface_2.merge_overlapping_edges(0.0001, 0.0001)
    assert not new_polyface_2.is_solid
    assert len(new_polyface_2.naked_edges) != 0


def test_polyface3d_init_from_faces_tolerance():
    """Test the initalization of Poyface3D from_faces with a tolerance."""
    pts_1 = [Point3D(0, 0, 0), Point3D(0, 2, 0), Point3D(2, 2, 0), Point3D(2, 0, 0)]
    pts_2 = [Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(0, 2, 0)]
    pts_3 = [Point3D(0, 0, 0), Point3D(2, 0, 0), Point3D(2, 0, 2), Point3D(0, 0, 2)]
    pts_4 = [Point3D(2, 2, 0), Point3D(0, 2, 0), Point3D(0, 2, 2), Point3D(2, 2, 2)]
    pts_5 = [Point3D(2, 2, 0), Point3D(2, 0, 0), Point3D(2, 0, 2), Point3D(2, 2, 2)]
    pts_6 = [Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2)]
    pts_7 = [Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2.0001)]
    face_1 = Face3D(pts_1)
    face_2 = Face3D(pts_2)
    face_3 = Face3D(pts_3)
    face_4 = Face3D(pts_4)
    face_5 = Face3D(pts_5)
    face_6 = Face3D(pts_6)
    face_7 = Face3D(pts_7)
    polyface_1 = Polyface3D.from_faces(
        [face_1, face_2, face_3, face_4, face_5, face_6], 0.001)
    polyface_2 = Polyface3D.from_faces(
        [face_1, face_2, face_3, face_4, face_5, face_7],  0.001)
    polyface_3 = Polyface3D.from_faces(
        [face_1, face_2, face_3, face_4, face_5, face_7],  0.000001)

    assert polyface_1.is_solid
    assert polyface_2.is_solid
    assert not polyface_3.is_solid


def test_polyface3d_init_from_box():
    """Test the initalization of Poyface3D from_box."""
    polyface = Polyface3D.from_box(2, 4, 2)

    assert len(polyface.vertices) == 8
    assert len(polyface.face_indices) == 6
    assert len(polyface.faces) == 6
    assert len(polyface.edge_indices) == 12
    assert len(polyface.edges) == 12
    assert len(polyface.naked_edges) == 0
    assert len(polyface.non_manifold_edges) == 0
    assert len(polyface.internal_edges) == 12
    assert polyface.area == 40
    assert polyface.volume == 16
    assert polyface.is_solid


def test_polyface3d_init_from_offset_face():
    """Test the initalization of Poyface3D from_offset_face."""
    face = Face3D.from_rectangle(2, 2)
    polyface = Polyface3D.from_offset_face(face, 2)

    assert len(polyface.vertices) == 8
    assert len(polyface.face_indices) == 6
    assert len(polyface.faces) == 6
    assert len(polyface.edge_indices) == 12
    assert len(polyface.edges) == 12
    assert len(polyface.naked_edges) == 0
    assert len(polyface.non_manifold_edges) == 0
    assert len(polyface.internal_edges) == 12
    assert polyface.area == 24
    assert polyface.volume == 8
    assert polyface.is_solid

    for f in polyface.faces:
        assert f.is_clockwise is False
    for e in polyface.edges:
        assert e.length == 2

    assert polyface.faces[0].normal.z == pytest.approx(-1, rel=1e-3)
    assert polyface.faces[-1].normal.z == pytest.approx(1, rel=1e-3)
    for face in polyface.faces:
        assert not face.is_clockwise


def test_polyface3d_init_from_offset_face_hexagon():
    """Test the initalization of Poyface3D from_offset_face."""
    face = Face3D.from_regular_polygon(6, 2)
    polyface = Polyface3D.from_offset_face(face, 2)

    assert len(polyface.vertices) == 12
    assert len(polyface.face_indices) == 8
    assert len(polyface.faces) == 8
    assert len(polyface.edge_indices) == 18
    assert len(polyface.edges) == 18
    assert len(polyface.naked_edges) == 0
    assert len(polyface.non_manifold_edges) == 0
    assert len(polyface.internal_edges) == 18
    assert polyface.area == pytest.approx(44.784609, rel=1e-3)
    assert polyface.volume == pytest.approx(20.78460, rel=1e-3)
    assert polyface.is_solid

    assert polyface.faces[0].normal.z == pytest.approx(-1, rel=1e-3)
    assert polyface.faces[-1].normal.z == pytest.approx(1, rel=1e-3)
    for face in polyface.faces:
        assert not face.is_clockwise


def test_polyface3d_init_from_offset_face_hole():
    """Test the initalization of Poyface3D from_offset_face for a face witha hole."""
    bound_pts = [Point3D(0, 0), Point3D(3, 0), Point3D(3, 3), Point3D(0, 3)]
    hole_pts = [Point3D(1, 1), Point3D(2, 1), Point3D(2, 2), Point3D(1, 2)]
    face = Face3D(bound_pts, None, [hole_pts])

    polyface = Polyface3D.from_offset_face(face, 1)

    assert len(polyface.vertices) == 16

    assert len(polyface.face_indices) == 10
    assert len(polyface.faces) == 10
    assert len(polyface.edge_indices) == 24
    assert len(polyface.edges) == 24
    assert len(polyface.naked_edges) == 0
    assert len(polyface.non_manifold_edges) == 0
    assert len(polyface.internal_edges) == 24
    assert polyface.area == pytest.approx(32, rel=1e-3)
    assert polyface.volume == pytest.approx(8, rel=1e-3)
    assert polyface.is_solid

    assert polyface.faces[0].normal.z == pytest.approx(-1, rel=1e-3)
    assert polyface.faces[-1].normal.z == pytest.approx(1, rel=1e-3)
    assert polyface.faces[0].has_holes
    assert polyface.faces[-1].has_holes
    for face in polyface.faces:
        assert not face.is_clockwise


def test_polyface3d_to_from_dict():
    """Test the to/from dict of Polyface3D objects."""
    polyface = Polyface3D.from_box(2, 4, 2)

    polyface_dict = polyface.to_dict()
    new_polyface = Polyface3D.from_dict(polyface_dict)
    assert isinstance(new_polyface, Polyface3D)
    assert new_polyface.to_dict() == polyface_dict

    assert len(new_polyface.vertices) == 8
    assert len(new_polyface.face_indices) == 6
    assert len(new_polyface.faces) == 6
    assert len(new_polyface.edge_indices) == 12
    assert len(new_polyface.edges) == 12
    assert len(new_polyface.naked_edges) == 0
    assert len(new_polyface.non_manifold_edges) == 0
    assert len(new_polyface.internal_edges) == 12
    assert new_polyface.area == 40
    assert new_polyface.volume == 16
    assert new_polyface.is_solid


def test_polyface3d_to_from_dict_hole():
    """Test the to/from dict of Polyface3D objects with a hole."""
    bound_pts = [Point3D(0, 0), Point3D(3, 0), Point3D(3, 3), Point3D(0, 3)]
    hole_pts = [Point3D(1, 1), Point3D(2, 1), Point3D(2, 2), Point3D(1, 2)]
    face = Face3D(bound_pts, None, [hole_pts])
    polyface = Polyface3D.from_offset_face(face, 1)

    polyface_dict = polyface.to_dict()
    new_polyface = Polyface3D.from_dict(polyface_dict)

    assert len(new_polyface.vertices) == 16
    assert len(new_polyface.face_indices) == 10
    assert len(new_polyface.faces) == 10
    assert len(new_polyface.edge_indices) == 24
    assert len(new_polyface.edges) == 24
    assert len(new_polyface.naked_edges) == 0
    assert len(new_polyface.non_manifold_edges) == 0
    assert len(new_polyface.internal_edges) == 24
    assert new_polyface.area == pytest.approx(32, rel=1e-3)
    assert new_polyface.volume == pytest.approx(8, rel=1e-3)
    assert new_polyface.is_solid

    assert new_polyface.faces[0].normal.z == pytest.approx(-1, rel=1e-3)
    assert new_polyface.faces[-1].normal.z == pytest.approx(1, rel=1e-3)
    assert new_polyface.faces[0].has_holes
    assert new_polyface.faces[-1].has_holes
    for face in polyface.faces:
        assert not face.is_clockwise


def test_polyface3d_to_from_dict_with_overlap():
    """Test the to/from dict of Polyface3D objects with overlapping edges."""
    pts_1 = [Point3D(0, 0, 0), Point3D(0, 2, 0), Point3D(2, 2, 0), Point3D(2, 0, 0)]
    pts_2 = [Point3D(0, 0, 0), Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(0, 2, 0)]
    pts_3 = [Point3D(0, 0, 0), Point3D(2, 0, 0), Point3D(2, 0, 2), Point3D(0, 0, 2)]
    pts_4 = [Point3D(2, 2, 0), Point3D(0, 2, 0), Point3D(0, 2, 2), Point3D(2, 2, 2)]
    pts_5 = [Point3D(2, 2, 0), Point3D(2, 0, 0), Point3D(2, 0, 2), Point3D(2, 2, 2)]
    pts_6 = [Point3D(0, 0, 2), Point3D(0, 1, 2), Point3D(2, 1, 2), Point3D(2, 0, 2)]
    pts_7 = [Point3D(0, 1, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 1, 2)]
    face_1 = Face3D(pts_1)
    face_2 = Face3D(pts_2)
    face_3 = Face3D(pts_3)
    face_4 = Face3D(pts_4)
    face_5 = Face3D(pts_5)
    face_6 = Face3D(pts_6)
    face_7 = Face3D(pts_7)
    polyface = Polyface3D.from_faces(
        [face_1, face_2, face_3, face_4, face_5, face_6, face_7])
    new_polyface = polyface.merge_overlapping_edges(0.0001, 0.0001)
    assert new_polyface.is_solid
    assert len(new_polyface.naked_edges) == 0
    assert len(new_polyface.internal_edges) == 13

    polyface_dict = new_polyface.to_dict()
    dict_polyface = Polyface3D.from_dict(polyface_dict)
    assert isinstance(dict_polyface, Polyface3D)
    assert dict_polyface.to_dict() == polyface_dict
    assert dict_polyface.is_solid
    assert len(dict_polyface.naked_edges) == 0
    assert len(dict_polyface.internal_edges) == 13


def test_is_solid_with_hole():
    """Test the is_solid property for a polyface with a hole.

    This ensures that the is_solid property still works where the Euler
    characteristic fails.
    """
    pts_1 = [Point3D(0, 0, 2), Point3D(0, 0, 0), Point3D(4, 0, 0), Point3D(4, 0, 2)]
    pts_2 = [Point3D(4, 0, 2), Point3D(4, 0, 0), Point3D(4, 4, 0), Point3D(4, 4, 2)]
    pts_3 = [Point3D(4, 4, 2), Point3D(4, 4, 0), Point3D(0, 4, 0), Point3D(0, 4, 2)]
    pts_4 = [Point3D(0, 4, 2), Point3D(0, 4, 0), Point3D(0, 0, 0), Point3D(0, 0, 2)]
    pts_5 = [Point3D(0, 0, 0), Point3D(0, 4, 0), Point3D(1, 3, 1), Point3D(1, 1, 1)]
    pts_6 = [Point3D(4, 0, 0), Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(3, 1, 1)]
    pts_7 = [Point3D(4, 4, 0), Point3D(4, 0, 0), Point3D(3, 1, 1), Point3D(3, 3, 1)]
    pts_8 = [Point3D(0, 4, 0), Point3D(4, 4, 0), Point3D(3, 3, 1), Point3D(1, 3, 1)]
    pts_9 = [Point3D(1, 1, 1), Point3D(1, 3, 1), Point3D(0, 4, 2), Point3D(0, 0, 2)]
    pts_10 = [Point3D(3, 1, 1), Point3D(1, 1, 1), Point3D(0, 0, 2), Point3D(4, 0, 2)]
    pts_11 = [Point3D(3, 3, 1), Point3D(3, 1, 1), Point3D(4, 0, 2), Point3D(4, 4, 2)]
    pts_12 = [Point3D(1, 3, 1), Point3D(3, 3, 1), Point3D(4, 4, 2), Point3D(0, 4, 2)]

    face_1 = Face3D(pts_1)
    face_2 = Face3D(pts_2)
    face_3 = Face3D(pts_3)
    face_4 = Face3D(pts_4)
    face_5 = Face3D(pts_5)
    face_6 = Face3D(pts_6)
    face_7 = Face3D(pts_7)
    face_8 = Face3D(pts_8)
    face_9 = Face3D(pts_9)
    face_10 = Face3D(pts_10)
    face_11 = Face3D(pts_11)
    face_12 = Face3D(pts_12)
    polyface = Polyface3D.from_faces([face_1, face_2, face_3, face_4, face_5,
                                      face_6, face_7, face_8, face_9, face_10,
                                      face_11, face_12])
    assert len(polyface.faces) + len(polyface.vertices) - len(polyface.edges) != 2
    assert polyface.area == pytest.approx(65.941125, rel=1e-3)
    assert polyface.volume == pytest.approx(13.333333, rel=1e-3)
    assert polyface.is_solid


def test_min_max_center():
    """Test the Face3D min, max and center."""
    polyface_1 = Polyface3D.from_box(2, 4, 2)
    polyface_2 = Polyface3D.from_box(math.sqrt(2), math.sqrt(2), 2, Plane(
        Vector3D(0, 0, 1), Point3D(1, 0, 0), Vector3D(1, 1, 0)))

    assert polyface_1.min == Point3D(0, 0, 0)
    assert polyface_1.max == Point3D(2, 4, 2)
    assert polyface_1.center == Point3D(1, 2, 1)
    assert polyface_1.volume == pytest.approx(16, rel=1e-3)

    assert polyface_2.min == Point3D(0, 0, 0)
    assert polyface_2.max == Point3D(2, 2, 2)
    assert polyface_2.center == Point3D(1, 1, 1)
    assert polyface_2.volume == pytest.approx(4, rel=1e-3)


def test_floor_mesh_grid():
    """Test the generation of a mesh grid from the floor of a box."""
    polyface = Polyface3D.from_box(5, 10, 3)
    floor_grid = polyface.faces[0].get_mesh_grid(1, 1, 1, True)
    assert len(floor_grid.faces) == 50

    angle = -1 * math.radians(45)
    x_axis = Vector3D(1, 0, 0).rotate_xy(angle)
    base_plane = Plane(Vector3D(0, 0, 1), Point3D(0, 0, 0), x_axis)
    polyface = Polyface3D.from_box(5, 10, 3, base_plane)
    floor_grid = polyface.faces[0].get_mesh_grid(1, 1, 1, True)
    assert len(floor_grid.faces) == 50


def test_duplicate():
    """Test the duplicate method of Face3D."""
    polyface = Polyface3D.from_box(2, 4, 2)
    new_polyface = polyface.duplicate()

    for i, pt in enumerate(polyface):
        assert pt == new_polyface[i]
    for i, fi in enumerate(polyface.face_indices):
        assert fi == new_polyface.face_indices[i]

    assert polyface.area == new_polyface.area
    assert polyface.is_solid == new_polyface.is_solid


def test_move():
    """Test the Polyface3D move method."""
    polyface = Polyface3D.from_box(2, 2, 2)

    vec_1 = Vector3D(2, 2, 2)
    new_polyface = polyface.move(vec_1)
    assert new_polyface[0] == Point3D(2, 2, 2)
    assert new_polyface[1] == Point3D(2, 4, 2)
    assert new_polyface[2] == Point3D(4, 4, 2)
    assert new_polyface[3] == Point3D(4, 2, 2)

    assert polyface.area == new_polyface.area
    assert polyface.volume == new_polyface.volume


def test_scale():
    """Test the Polyface3D scale method."""
    polyface_1 = Polyface3D.from_box(2, 2, 2, Plane(o=Point3D(0, 0, 2)))
    polyface_2 = Polyface3D.from_box(1, 1, 1, Plane(o=Point3D(1, 1, 0)))
    origin_1 = Point3D(2, 0)
    origin_2 = Point3D(1, 1)

    new_polyface_1 = polyface_1.scale(2, origin_1)
    assert new_polyface_1[0] == Point3D(-2, 0, 4)
    assert new_polyface_1[1] == Point3D(-2, 4, 4)
    assert new_polyface_1[2] == Point3D(2, 4, 4)
    assert new_polyface_1[3] == Point3D(2, 0, 4)
    assert new_polyface_1.area == polyface_1.area * 2 ** 2
    assert new_polyface_1.volume == polyface_1.volume * 2 ** 3

    new_polyface_2 = polyface_2.scale(2, origin_2)
    assert new_polyface_2[0] == Point3D(1, 1)
    assert new_polyface_2[1] == Point3D(1, 3)
    assert new_polyface_2[2] == Point3D(3, 3)
    assert new_polyface_2[3] == Point3D(3, 1)
    assert new_polyface_2.area == polyface_2.area * 2 ** 2
    assert new_polyface_2.volume == polyface_2.volume * 2 ** 3


def test_scale_world_origin():
    """Test the Polyface3D scale method with None origin."""
    polyface = Polyface3D.from_box(1, 1, 1, Plane(o=Point3D(1, 1, 2)))

    new_polyface = polyface.scale(2)
    assert new_polyface[0] == Point3D(2, 2, 4)
    assert new_polyface[1] == Point3D(2, 4, 4)
    assert new_polyface[2] == Point3D(4, 4, 4)
    assert new_polyface[3] == Point3D(4, 2, 4)
    assert new_polyface.area == polyface.area * 2 ** 2
    assert new_polyface.volume == polyface.volume * 2 ** 3


def test_rotate():
    """Test the Polyface3D rotate method."""
    polyface = Polyface3D.from_box(2, 2, 2, Plane(o=Point3D(0, 0, 2)))
    origin = Point3D(0, 0, 0)
    axis = Vector3D(1, 0, 0)

    test_1 = polyface.rotate(axis, math.pi, origin)
    assert test_1[0].x == pytest.approx(0, rel=1e-3)
    assert test_1[0].y == pytest.approx(0, rel=1e-3)
    assert test_1[0].z == pytest.approx(-2, rel=1e-3)
    assert test_1[2].x == pytest.approx(2, rel=1e-3)
    assert test_1[2].y == pytest.approx(-2, rel=1e-3)
    assert test_1[2].z == pytest.approx(-2, rel=1e-3)
    assert polyface.area == test_1.area
    assert polyface.volume == test_1.volume
    assert len(polyface.vertices) == len(test_1.vertices)

    test_2 = polyface.rotate(axis, math.pi/2, origin)
    assert test_2[0].x == pytest.approx(0, rel=1e-3)
    assert test_2[0].y == pytest.approx(-2, rel=1e-3)
    assert test_2[0].z == pytest.approx(0, rel=1e-3)
    assert test_2[2].x == pytest.approx(2, rel=1e-3)
    assert test_2[2].y == pytest.approx(-2, rel=1e-3)
    assert test_2[2].z == pytest.approx(2, rel=1e-3)
    assert polyface.area == test_2.area
    assert polyface.volume == test_1.volume
    assert len(polyface.vertices) == len(test_2.vertices)


def test_rotate_xy():
    """Test the Polyface3D rotate_xy method."""
    polyface = Polyface3D.from_box(1, 1, 1, Plane(o=Point3D(1, 1, 2)))
    origin_1 = Point3D(1, 1, 0)

    test_1 = polyface.rotate_xy(math.pi, origin_1)
    assert test_1[0].x == pytest.approx(1, rel=1e-3)
    assert test_1[0].y == pytest.approx(1, rel=1e-3)
    assert test_1[0].z == pytest.approx(2, rel=1e-3)
    assert test_1[2].x == pytest.approx(0, rel=1e-3)
    assert test_1[2].y == pytest.approx(0, rel=1e-3)
    assert test_1[2].z == pytest.approx(2, rel=1e-3)

    test_2 = polyface.rotate_xy(math.pi/2, origin_1)
    assert test_2[0].x == pytest.approx(1, rel=1e-3)
    assert test_2[0].y == pytest.approx(1, rel=1e-3)
    assert test_1[0].z == pytest.approx(2, rel=1e-3)
    assert test_2[2].x == pytest.approx(0, rel=1e-3)
    assert test_2[2].y == pytest.approx(2, rel=1e-3)
    assert test_1[2].z == pytest.approx(2, rel=1e-3)


def test_reflect():
    """Test the Polyface3D reflect method."""
    polyface = Polyface3D.from_box(1, 1, 1, Plane(o=Point3D(1, 1, 2)))

    origin_1 = Point3D(1, 0, 2)
    normal_1 = Vector3D(1, 0, 0)
    normal_2 = Vector3D(-1, -1, 0).normalize()

    test_1 = polyface.reflect(normal_1, origin_1)
    assert test_1[0].x == pytest.approx(1, rel=1e-3)
    assert test_1[0].y == pytest.approx(1, rel=1e-3)
    assert test_1[0].z == pytest.approx(2, rel=1e-3)
    assert test_1[2].x == pytest.approx(0, rel=1e-3)
    assert test_1[2].y == pytest.approx(2, rel=1e-3)
    assert test_1[2].z == pytest.approx(2, rel=1e-3)

    test_1 = polyface.reflect(normal_2, Point3D(0, 0, 2))
    assert test_1[0].x == pytest.approx(-1, rel=1e-3)
    assert test_1[0].y == pytest.approx(-1, rel=1e-3)
    assert test_1[0].z == pytest.approx(2, rel=1e-3)
    assert test_1[2].x == pytest.approx(-2, rel=1e-3)
    assert test_1[2].y == pytest.approx(-2, rel=1e-3)
    assert test_1[2].z == pytest.approx(2, rel=1e-3)

    test_2 = polyface.reflect(normal_2, origin_1)
    assert test_2[0].x == pytest.approx(0, rel=1e-3)
    assert test_2[0].y == pytest.approx(0, rel=1e-3)
    assert test_2[0].z == pytest.approx(2, rel=1e-3)
    assert test_2[2].x == pytest.approx(-1, rel=1e-3)
    assert test_2[2].y == pytest.approx(-1, rel=1e-3)
    assert test_2[2].z == pytest.approx(2, rel=1e-3)


def test_intersect_line_ray():
    """Test the Polyface3D intersect_line_ray method."""
    pts = (Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(2, 1, 2), Point3D(1, 1, 2),
           Point3D(1, 2, 2), Point3D(0, 2, 2))
    plane = Plane(Vector3D(0, 0, 1), Point3D(0, 0, 2))
    face = Face3D(pts, plane)
    polyface = Polyface3D.from_offset_face(face, 1)
    ray_1 = Ray3D(Point3D(0.5, 0.5, 0), Vector3D(0, 0, 1))
    ray_2 = Ray3D(Point3D(0.5, 0.5, 0), Vector3D(0, 0, -1))
    ray_3 = Ray3D(Point3D(1.5, 1.5, 0), Vector3D(0, 0, 1))
    ray_4 = Ray3D(Point3D(-1, -1, 0), Vector3D(0, 0, 1))
    line_1 = LineSegment3D(Point3D(0.5, 0.5, 0), Vector3D(0, 0, 3))
    line_2 = LineSegment3D(Point3D(0.5, 0.5, 0), Vector3D(0, 0, 1))

    assert polyface.does_intersect_line_ray_exist(ray_1) is True
    assert polyface.does_intersect_line_ray_exist(ray_2) is False
    assert polyface.does_intersect_line_ray_exist(ray_3) is False
    assert polyface.does_intersect_line_ray_exist(ray_4) is False
    assert polyface.does_intersect_line_ray_exist(line_1) is True
    assert polyface.does_intersect_line_ray_exist(line_2) is False

    assert polyface.intersect_line_ray(ray_1) == [Point3D(0.5, 0.5, 2),
                                                  Point3D(0.5, 0.5, 3)]
    assert polyface.intersect_line_ray(ray_2) == []
    assert polyface.intersect_line_ray(ray_3) == []
    assert polyface.intersect_line_ray(ray_4) == []
    assert polyface.intersect_line_ray(line_1) == [Point3D(0.5, 0.5, 2),
                                                   Point3D(0.5, 0.5, 3)]
    assert polyface.intersect_line_ray(line_2) == []


def test_intersect_plane():
    """Test the Polyface3D intersect_plane method."""
    pts = (Point3D(0, 0, 2), Point3D(2, 0, 2), Point3D(2, 1, 2), Point3D(1, 1, 2),
           Point3D(1, 2, 2), Point3D(0, 2, 2))
    plane = Plane(Vector3D(0, 0, 1), Point3D(0, 0, 2))
    face = Face3D(pts, plane)
    polyface = Polyface3D.from_offset_face(face, 1)

    plane_1 = Plane(Vector3D(0, 1, 0), Point3D(0.5, 0.5, 0))
    plane_2 = Plane(Vector3D(1, 0, 0), Point3D(0.5, 0.5, 0))
    plane_3 = Plane(Vector3D(0, 1, 0), Point3D(0.5, 1.5, 0))
    plane_4 = Plane(Vector3D(0, 1, 0), Point3D(0, 3, 0))
    plane_5 = Plane(Vector3D(1, 1, 0), Point3D(0, 2.5, 0))

    assert len(polyface.intersect_plane(plane_1)) == 4
    assert len(polyface.intersect_plane(plane_2)) == 4
    assert len(polyface.intersect_plane(plane_3)) == 4
    assert len(polyface.intersect_plane(plane_4)) == 0
    assert len(polyface.intersect_plane(plane_5)) == 8


def test_is_point_inside():
    """Test the is_point_inside method."""
    pts = [Point3D(0, 0, 0), Point3D(0, 2, 0), Point3D(2, 2, 0), Point3D(2, 0, 0),
           Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2)]
    face_indices = [[(0, 1, 2, 3)], [(0, 4, 5, 1)], [(0, 3, 7, 4)],
                    [(2, 1, 5, 6)], [(2, 3, 7, 6)], [(4, 5, 6, 7)]]
    polyface = Polyface3D(pts, face_indices)

    assert polyface.is_point_inside(Point3D(1, 1, 1)) is True
    assert polyface.is_point_inside(Point3D(4, 1, 1)) is False
    assert polyface.is_point_inside(Point3D(-1, 1, 1)) is False
    assert polyface.is_point_inside(Point3D(4, 4, 4)) is False


def test_overlapping_bounding_boxes():
    """Test the Polyface3D overlapping_bounding_boxes method."""
    polyface_1 = Polyface3D.from_box(1, 1, 1, Plane(o=Point3D(1, 1, 2)))
    polyface_2 = Polyface3D.from_box(1, 1, 1, Plane(o=Point3D(1, 1, 1)))
    polyface_3 = Polyface3D.from_box(1, 1, 1, Plane(o=Point3D(2, 1, 2)))
    polyface_4 = Polyface3D.from_box(1, 1, 1, Plane(o=Point3D(1, 2, 2)))
    polyface_5 = Polyface3D.from_box(1, 1, 1, Plane(o=Point3D(0, 0, 0)))

    assert Polyface3D.overlapping_bounding_boxes(polyface_1, polyface_2, 0.01)
    assert Polyface3D.overlapping_bounding_boxes(polyface_1, polyface_3, 0.01)
    assert Polyface3D.overlapping_bounding_boxes(polyface_1, polyface_4, 0.01)
    assert not Polyface3D.overlapping_bounding_boxes(polyface_1, polyface_5, 0.01)
