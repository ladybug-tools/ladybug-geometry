from ladybug_geometry.geometry2d import LineSegment2D
from ladybug_geometry.geometry3d import Vector3D, Point3D, LineSegment3D, \
    Plane, Face3D, Polyface3D
from ladybug_geometry.projection import project_geometry, project_geometry_2d


def test_projection():
    """Test the projection methods with arrays of 3D objects."""
    proj_plane = Plane(n=Vector3D(1, 1, 1))

    plane1 = Plane(o=Point3D(-5, 0, 0))
    plane2 = Plane(o=Point3D(0, -4, 4))
    plane3 = Plane(o=Point3D(2, 2, -4))
    polyface1 = Polyface3D.from_box(2, 4, 2, plane1)
    polyface2 = Polyface3D.from_box(2, 4, 2, plane2)
    polyface3 = Polyface3D.from_box(2, 4, 2, plane3)

    poly_faces = [polyface1, polyface2, polyface3]
    geometries = project_geometry(proj_plane, poly_faces)
    assert len(geometries) == 3
    for geo in geometries:
        assert isinstance(geo, Polyface3D)
        assert all(proj_plane.distance_to_point(pt) < 0.001 for pt in geo.vertices)

    faces = [f for pf in poly_faces for f in pf.faces]
    geometries = project_geometry(proj_plane, faces)
    assert len(geometries) == 18
    for geo in geometries:
        assert isinstance(geo, Face3D)
        assert all(proj_plane.distance_to_point(pt) < 0.001 for pt in geo.vertices)

    lines = [edge for pf in poly_faces for edge in pf.edges]
    geometries = project_geometry(proj_plane, lines)
    assert len(geometries) == 36
    for geo in geometries:
        assert isinstance(geo, LineSegment3D)
        assert all(proj_plane.distance_to_point(pt) < 0.001 for pt in geo.vertices)


def test_projection_2d():
    """Test the projection_2d methods with arrays of 3D objects."""
    proj_plane = Plane(n=Vector3D(1, 1, 1))

    plane1 = Plane(o=Point3D(-5, 0, 0))
    plane2 = Plane(o=Point3D(0, -4, 4))
    plane3 = Plane(o=Point3D(2, 2, -4))
    polyface1 = Polyface3D.from_box(2, 4, 2, plane1)
    polyface2 = Polyface3D.from_box(2, 4, 2, plane2)
    polyface3 = Polyface3D.from_box(2, 4, 2, plane3)

    poly_faces = [polyface1, polyface2, polyface3]
    geometries = project_geometry_2d(proj_plane, poly_faces)
    assert len(geometries) == 3
    for geo in geometries:
        assert isinstance(geo, Polyface3D)

    faces = [f for pf in poly_faces for f in pf.faces]
    geometries = project_geometry_2d(proj_plane, faces)
    assert len(geometries) == 18
    for geo in geometries:
        assert isinstance(geo, Face3D)

    lines = [edge for pf in poly_faces for edge in pf.edges]
    geometries = project_geometry_2d(proj_plane, lines)
    assert len(geometries) == 36
    for geo in geometries:
        assert isinstance(geo, LineSegment2D)
