# coding=utf-8
import pytest

from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
from ladybug_geometry.geometry3d.mesh import Mesh3D
from ladybug_geometry.geometry3d.plane import Plane
from ladybug_geometry.geometry2d.mesh import Mesh2D
from ladybug_geometry.geometry2d.pointvector import Point2D

import math


def test_mesh3d_init():
    """Test the initalization of Mesh3D objects and basic properties."""
    pts = (Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2))
    mesh = Mesh3D(pts, [(0, 1, 2, 3)])
    str(mesh)  # test the string representation of the object

    assert len(mesh.vertices) == 4
    assert len(mesh.faces) == 1
    assert mesh[0] == Point3D(0, 0, 2)
    assert mesh[1] == Point3D(0, 2, 2)
    assert mesh[2] == Point3D(2, 2, 2)
    assert mesh[3] == Point3D(2, 0, 2)
    assert mesh.area == 4

    assert mesh.min == Point3D(0, 0, 2)
    assert mesh.max == Point3D(2, 2, 2)
    assert mesh.center == Point3D(1, 1, 2)

    assert len(mesh.face_areas) == 1
    assert mesh.face_areas[0] == 4
    assert len(mesh.face_centroids) == 1
    assert mesh.face_centroids[0] == Point3D(1, 1, 2)
    assert mesh._is_color_by_face is False
    assert mesh.colors is None
    assert len(mesh.vertex_connected_faces) == 4
    for vf in mesh.vertex_connected_faces:
        assert len(vf) == 1


def test_equality():
    """Test the equality of Mesh3D objects."""
    pts = (Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2))
    pts_2 = (Point3D(0.1, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2))
    mesh = Mesh3D(pts, [(0, 1, 2, 3)])
    mesh_dup = mesh.duplicate()
    mesh_alt = Mesh3D(pts_2, [(0, 1, 2, 3)])

    assert mesh is mesh
    assert mesh is not mesh_dup
    assert mesh == mesh_dup
    assert hash(mesh) == hash(mesh_dup)
    assert mesh != mesh_alt
    assert hash(mesh) != hash(mesh_alt)


def test_mesh3d_to_from_dict():
    """Test the to/from dict of Mesh3D objects."""
    pts = (Point3D(0, 0), Point3D(0, 2), Point3D(2, 2), Point3D(2, 0))
    mesh = Mesh3D(pts, [(0, 1, 2, 3)])
    mesh_dict = mesh.to_dict()
    new_mesh = Mesh3D.from_dict(mesh_dict)
    assert isinstance(new_mesh, Mesh3D)
    assert new_mesh.to_dict() == mesh_dict


def test_face_normals():
    """Test the Mesh3D face_normals property."""
    pts = (Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2))
    mesh = Mesh3D(pts, [(0, 1, 2, 3)])

    assert len(mesh.face_normals) == 1
    assert mesh.face_normals[0] == Vector3D(0, 0, -1)
    assert len(mesh.vertex_normals) == 4
    for vert_norm in mesh.vertex_normals:
        assert vert_norm == Vector3D(0, 0, -1)


def test_mesh3d_incorrect():
    """Test the initalization of Mesh3D objects with incorrect values."""
    pts = (Point3D(0, 0), Point3D(0, 2), Point3D(2, 2), Point3D(2, 0), Point3D(4, 0))
    with pytest.raises(AssertionError):
        Mesh3D(pts, [(0, 1, 2, 3, 5)])  # too many vertices in a face
    with pytest.raises(AssertionError):
        Mesh3D(pts, [])  # we need at least one face
    with pytest.raises(AssertionError):
        Mesh3D(pts, (0, 1, 2, 3))  # incorrect input type for face
    with pytest.raises(IndexError):
        Mesh3D(pts, [(0, 1, 2, 6)])  # incorrect index used by face
    with pytest.raises(TypeError):
        Mesh3D(pts, [(0.0, 1, 2, 6)])  # incorrect use of floats for face index


def test_mesh3d_init_two_faces():
    """Test the initalization of Mesh3D objects with two faces."""
    pts = (Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2),
           Point3D(2, 0, 2), Point3D(4, 0, 2))
    mesh = Mesh3D(pts, [(0, 1, 2, 3), (2, 3, 4)])

    assert len(mesh.vertices) == 5
    assert len(mesh.faces) == 2
    assert mesh[0] == Point3D(0, 0, 2)
    assert mesh[1] == Point3D(0, 2, 2)
    assert mesh[2] == Point3D(2, 2, 2)
    assert mesh[3] == Point3D(2, 0, 2)
    assert mesh[4] == Point3D(4, 0, 2)
    assert mesh.area == 6

    assert mesh.min == Point3D(0, 0, 2)
    assert mesh.max == Point3D(4, 2, 2)
    assert mesh.center == Point3D(2, 1, 2)

    assert len(mesh.face_areas) == 2
    assert mesh.face_areas[0] == 4
    assert mesh.face_areas[1] == 2
    assert len(mesh.face_centroids) == 2
    assert mesh.face_centroids[0] == Point3D(1, 1, 2)
    assert mesh.face_centroids[1].x == pytest.approx(2.67, rel=1e-2)
    assert mesh.face_centroids[1].y == pytest.approx(0.67, rel=1e-2)
    assert mesh.face_centroids[1].z == pytest.approx(2, rel=1e-2)
    assert mesh._is_color_by_face is False
    assert mesh.colors is None


def test_mesh3d_init_from_face_vertices():
    """Test the initalization of Mesh3D from_face_vertices."""
    face_1 = (Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2))
    face_2 = (Point3D(2, 2, 2), Point3D(2, 0, 2), Point3D(4, 0, 2))
    mesh_1 = Mesh3D.from_face_vertices([face_1, face_2])
    mesh_2 = Mesh3D.from_face_vertices([face_1, face_2], False)

    assert len(mesh_1.vertices) == 5
    assert len(mesh_2.vertices) == 7
    assert len(mesh_1.faces) == len(mesh_2.faces) == 2
    assert mesh_1.area == mesh_2.area == 6

    assert mesh_1.min == mesh_2.min == Point3D(0, 0, 2)
    assert mesh_1.max == mesh_2.max == Point3D(4, 2, 2)
    assert mesh_1.center == mesh_2.center == Point3D(2, 1, 2)

    assert len(mesh_1.face_areas) == len(mesh_2.face_areas) == 2
    assert mesh_1.face_areas[0] == mesh_2.face_areas[0] == 4
    assert mesh_1.face_areas[1] == mesh_2.face_areas[1] == 2
    assert len(mesh_1.face_centroids) == len(mesh_2.face_centroids) == 2
    assert mesh_1.face_centroids[0] == mesh_2.face_centroids[0] == Point3D(1, 1, 2)
    assert mesh_1.face_centroids[1].x == mesh_2.face_centroids[1].x == pytest.approx(2.67, rel=1e-2)
    assert mesh_1.face_centroids[1].y == mesh_2.face_centroids[1].y == pytest.approx(0.67, rel=1e-2)
    assert mesh_1.face_centroids[1].z == mesh_2.face_centroids[1].z == pytest.approx(2, rel=1e-2)

    assert mesh_1._is_color_by_face is mesh_1._is_color_by_face is False
    assert mesh_1.colors is mesh_1.colors is None


def test_mesh3d_from_mesh2d():
    """Test the initalization of Mesh3D objects from_mesh2d."""
    pts = (Point2D(0, 0), Point2D(0, 2), Point2D(2, 2), Point2D(2, 0))
    mesh_2d = Mesh2D(pts, [(0, 1, 2, 3)])
    plane = Plane(Vector3D(1, 0, 0), Point3D(0, 0, 0))

    mesh = Mesh3D.from_mesh2d(mesh_2d, plane)
    assert len(mesh.vertices) == 4
    assert len(mesh.faces) == 1
    assert mesh[0] == Point3D(0, 0, 0)
    assert mesh[1] == Point3D(0, 0, -2)
    assert mesh[2] == Point3D(0, -2, -2)
    assert mesh[3] == Point3D(0, -2, 0)
    assert mesh.area == 4
    assert mesh.min == Point3D(0, -2, -2)
    assert mesh.max == Point3D(0, 0, 0)
    assert mesh.center == Point3D(0, -1, -1)


def test_remove_vertices():
    """Test the Mesh3D remove_vertices method."""
    mesh_2d = Mesh2D.from_grid(Point2D(1, 1), 8, 2, 0.25, 1)
    mesh = Mesh3D.from_mesh2d(mesh_2d)
    assert len(mesh.vertices) == 27
    assert len(mesh.faces) == 16
    assert mesh.area == 4

    pattern_1 = []
    for i in range(9):
        pattern_1.extend([True, True, False])
    mesh_1, vert_pattern = mesh.remove_vertices(pattern_1)
    assert len(mesh_1.vertices) == 18
    assert len(mesh_1.faces) == 8
    assert mesh_1.area == 2
    for face in mesh_1.faces:
        for i in face:
            mesh_1[i]  # make sure all face indices reference current vertices


def test_remove_faces():
    """Test the Mesh3D remove_faces method."""
    mesh_2d = Mesh2D.from_grid(Point2D(1, 1), 8, 2, 0.25, 1)
    mesh = Mesh3D.from_mesh2d(mesh_2d)
    assert len(mesh.vertices) == 27
    assert len(mesh.faces) == 16
    assert mesh.area == 4

    pattern_1 = []
    for i in range(4):
        pattern_1.extend([True, False, False, False])
    mesh_1, vert_pattern = mesh.remove_faces(pattern_1)
    assert len(mesh_1.vertices) == 16
    assert len(mesh_1.faces) == 4
    assert mesh_1.area == 1
    for face in mesh_1.faces:
        for i in face:
            mesh_1[i]  # make sure all face indices reference current vertices

    pattern_2 = []
    for i in range(8):
        pattern_2.extend([True, False])
    mesh_2, vert_pattern = mesh.remove_faces(pattern_2)
    assert len(mesh_2.vertices) == 18
    assert len(mesh_2.faces) == 8
    assert mesh_2.area == 2
    for face in mesh_2.faces:
        for i in face:
            mesh_2[i]  # make sure all face indices reference current vertices


def test_remove_faces_only():
    """Test the Mesh3D remove_faces method."""
    mesh_2d = Mesh2D.from_grid(Point2D(1, 1), 8, 2, 0.25, 1)
    mesh = Mesh3D.from_mesh2d(mesh_2d)
    assert len(mesh.vertices) == 27
    assert len(mesh.faces) == 16
    assert mesh.area == 4

    pattern_1 = []
    for i in range(4):
        pattern_1.extend([True, False, False, False])
    mesh_1 = mesh.remove_faces_only(pattern_1)
    assert len(mesh_1.vertices) == 27
    assert len(mesh_1.faces) == 4
    assert mesh_1.area == 1

    pattern_2 = []
    for i in range(8):
        pattern_2.extend([True, False])
    mesh_2 = mesh.remove_faces_only(pattern_2)
    assert len(mesh_2.vertices) == 27
    assert len(mesh_2.faces) == 8
    assert mesh_2.area == 2


def test_move():
    """Test the Mesh3D move method."""
    pts = (Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2),
           Point3D(2, 0, 2), Point3D(4, 0, 2))
    mesh = Mesh3D(pts, [(0, 1, 2, 3), (2, 3, 4)])

    vec_1 = Vector3D(2, 2, -1)
    new_mesh = mesh.move(vec_1)
    assert new_mesh[0] == Point3D(2, 2, 1)
    assert new_mesh[1] == Point3D(2, 4, 1)
    assert new_mesh[2] == Point3D(4, 4, 1)
    assert new_mesh[3] == Point3D(4, 2, 1)
    assert new_mesh[4] == Point3D(6, 2, 1)

    assert mesh.area == new_mesh.area
    assert len(mesh.vertices) == len(new_mesh.vertices)
    assert len(mesh.faces) == len(new_mesh.faces)


def test_scale():
    """Test the Mesh3D scale method."""
    pts = (Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2),
           Point3D(2, 0, 2), Point3D(4, 0, 2))
    mesh = Mesh3D(pts, [(0, 1, 2, 3), (2, 3, 4)])
    origin_1 = Point3D(2, 0, 2)

    new_mesh_1 = mesh.scale(2, origin_1)
    assert new_mesh_1[0] == Point3D(-2, 0, 2)
    assert new_mesh_1[1] == Point3D(-2, 4, 2)
    assert new_mesh_1[2] == Point3D(2, 4, 2)
    assert new_mesh_1[3] == Point3D(2, 0, 2)
    assert new_mesh_1[4] == Point3D(6, 0, 2)
    assert new_mesh_1.area == 24
    assert len(mesh.vertices) == len(new_mesh_1.vertices)
    assert len(mesh.faces) == len(new_mesh_1.faces)


def test_scale_world_origin():
    """Test the Mesh2D scale method with None origin."""
    pts = (Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2),
           Point3D(2, 0, 2), Point3D(4, 0, 2))
    mesh = Mesh3D(pts, [(0, 1, 2, 3), (2, 3, 4)])

    new_mesh_1 = mesh.scale(2)
    assert new_mesh_1[0] == Point3D(0, 0, 4)
    assert new_mesh_1[1] == Point3D(0, 4, 4)
    assert new_mesh_1[2] == Point3D(4, 4, 4)
    assert new_mesh_1[3] == Point3D(4, 0, 4)
    assert new_mesh_1[4] == Point3D(8, 0, 4)
    assert new_mesh_1.area == 24
    assert len(mesh.vertices) == len(new_mesh_1.vertices)
    assert len(mesh.faces) == len(new_mesh_1.faces)


def test_rotate():
    """Test the Mesh3D rotate method."""
    pts = (Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2),
           Point3D(2, 0, 2), Point3D(4, 0, 2))
    mesh = Mesh3D(pts, [(0, 1, 2, 3), (2, 3, 4)])
    origin = Point3D(0, 0, 0)
    axis = Vector3D(1, 0, 0)

    test_1 = mesh.rotate(axis, math.pi, origin)
    assert test_1[0].x == pytest.approx(0, rel=1e-3)
    assert test_1[0].y == pytest.approx(0, rel=1e-3)
    assert test_1[0].z == pytest.approx(-2, rel=1e-3)
    assert test_1[2].x == pytest.approx(2, rel=1e-3)
    assert test_1[2].y == pytest.approx(-2, rel=1e-3)
    assert test_1[2].z == pytest.approx(-2, rel=1e-3)
    assert mesh.area == test_1.area
    assert len(mesh.vertices) == len(test_1.vertices)
    assert len(mesh.faces) == len(test_1.faces)

    test_2 = mesh.rotate(axis, math.pi/2, origin)
    assert test_2[0].x == pytest.approx(0, rel=1e-3)
    assert test_2[0].y == pytest.approx(-2, rel=1e-3)
    assert test_2[0].z == pytest.approx(0, rel=1e-3)
    assert test_2[2].x == pytest.approx(2, rel=1e-3)
    assert test_2[2].y == pytest.approx(-2, rel=1e-3)
    assert test_2[2].z == pytest.approx(2, rel=1e-3)
    assert mesh.area == test_2.area
    assert len(mesh.vertices) == len(test_2.vertices)
    assert len(mesh.faces) == len(test_2.faces)


def test_rotate_xy():
    """Test the Mesh3D rotate_xy method."""
    pts = (Point3D(1, 1, 2), Point3D(2, 1, 2), Point3D(2, 2, 2), Point3D(1, 2, 2))
    mesh = Mesh3D(pts, [(0, 1, 2, 3)])
    origin_1 = Point3D(1, 1, 0)

    test_1 = mesh.rotate_xy(math.pi, origin_1)
    assert test_1[0].x == pytest.approx(1, rel=1e-3)
    assert test_1[0].y == pytest.approx(1, rel=1e-3)
    assert test_1[0].z == pytest.approx(2, rel=1e-3)
    assert test_1[2].x == pytest.approx(0, rel=1e-3)
    assert test_1[2].y == pytest.approx(0, rel=1e-3)
    assert test_1[2].z == pytest.approx(2, rel=1e-3)

    test_2 = mesh.rotate_xy(math.pi/2, origin_1)
    assert test_2[0].x == pytest.approx(1, rel=1e-3)
    assert test_2[0].y == pytest.approx(1, rel=1e-3)
    assert test_1[0].z == pytest.approx(2, rel=1e-3)
    assert test_2[2].x == pytest.approx(0, rel=1e-3)
    assert test_2[2].y == pytest.approx(2, rel=1e-3)
    assert test_1[2].z == pytest.approx(2, rel=1e-3)


def test_reflect():
    """Test the Mesh3D reflect method."""
    pts = (Point3D(1, 1, 2), Point3D(2, 1, 2), Point3D(2, 2, 2), Point3D(1, 2, 2))
    mesh = Mesh3D(pts, [(0, 1, 2, 3)])
    origin_1 = Point3D(1, 0, 2)
    normal_1 = Vector3D(1, 0, 0)
    normal_2 = Vector3D(-1, -1, 0).normalize()

    test_1 = mesh.reflect(normal_1, origin_1)
    assert test_1[0].x == pytest.approx(1, rel=1e-3)
    assert test_1[0].y == pytest.approx(1, rel=1e-3)
    assert test_1[0].z == pytest.approx(2, rel=1e-3)
    assert test_1[2].x == pytest.approx(0, rel=1e-3)
    assert test_1[2].y == pytest.approx(2, rel=1e-3)
    assert test_1[2].z == pytest.approx(2, rel=1e-3)

    test_1 = mesh.reflect(normal_2, Point3D(0, 0, 0))
    assert test_1[0].x == pytest.approx(-1, rel=1e-3)
    assert test_1[0].y == pytest.approx(-1, rel=1e-3)
    assert test_1[0].z == pytest.approx(2, rel=1e-3)
    assert test_1[2].x == pytest.approx(-2, rel=1e-3)
    assert test_1[2].y == pytest.approx(-2, rel=1e-3)
    assert test_1[2].z == pytest.approx(2, rel=1e-3)

    test_2 = mesh.reflect(normal_2, origin_1)
    assert test_2[0].x == pytest.approx(0, rel=1e-3)
    assert test_2[0].y == pytest.approx(0, rel=1e-3)
    assert test_2[0].z == pytest.approx(2, rel=1e-3)
    assert test_2[2].x == pytest.approx(-1, rel=1e-3)
    assert test_2[2].y == pytest.approx(-1, rel=1e-3)
    assert test_2[2].z == pytest.approx(2, rel=1e-3)


def test_offset_mesh():
    """Test the offset_mesh method."""
    pts = (Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2))
    pts_rev = tuple(reversed(pts))
    mesh = Mesh3D(pts, [(0, 1, 2, 3)])
    mesh_rev = Mesh3D(pts_rev, [(0, 1, 2, 3)])

    new_mesh = mesh.offset_mesh(2)
    for v in new_mesh.vertices:
        assert v.z == 0

    new_mesh_rev = mesh_rev.offset_mesh(2)
    for v in new_mesh_rev.vertices:
        assert v.z == 4


def test_height_field_mesh():
    """Test the height_field_mesh method."""
    pts = (Point3D(0, 0, 0), Point3D(2, 0, 0), Point3D(2, 2, 0), Point3D(0, 2, 0))
    mesh = Mesh3D(pts, [(0, 1, 2, 3)])
    values = [-1, 0, 1, 2]

    new_mesh = mesh.height_field_mesh(values, (0, 3))
    assert new_mesh[0].z == 0
    assert new_mesh[1].z == 1
    assert new_mesh[2].z == 2
    assert new_mesh[3].z == 3


def test_height_field_mesh_faces():
    """Test the height_field_mesh method with values for faces."""
    pts = (Point3D(0, 0, 0), Point3D(2, 0, 0), Point3D(2, 2, 0), Point3D(0, 2, 0),
           Point3D(4, 0, 0))
    mesh = Mesh3D(pts, [(0, 1, 2, 3), (2, 3, 4)])
    values = [-1, 1]

    new_mesh = mesh.height_field_mesh(values, (1, 2))
    assert new_mesh[0].z == 1
    assert new_mesh[1].z == 1
    assert new_mesh[2].z == 1.5
    assert new_mesh[3].z == 1.5
    assert new_mesh[4].z == 2
