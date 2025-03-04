# coding=utf-8
"""Utility functions for performing plane projections in 3D space.

This module can be used to create axonometric views of geometry among other
purposes.
"""
from __future__ import division

from .geometry2d import Vector2D, Point2D, Ray2D, LineSegment2D, \
    Polyline2D, Arc2D, Polygon2D, Mesh2D
from .geometry3d import Vector3D, Point3D, Ray3D, Plane, LineSegment3D, \
    Polyline3D, Arc3D, Face3D, Mesh3D, Polyface3D, Sphere, Cone, Cylinder


def project_geometry(plane, geometries):
    """"Project multiple geometries into a plane to get them in the world 3D system.

    Args:
        plane: The Plane into which the geometries will be projected.
        geometries: An array of any ladybug_geometry objects which will be projected
            into the plane. Note that 2D geometry objects will be converted into
            3D (and are assumed to be in the World XY plane) in order to be
            projected into the Plane. Also, Arcs will be converted to Polylines
            in order to be represented correctly in the plane.

    Returns:
        The input geometries projected into the Plane. All coordinate values will
        be in the World 3D system.
    """
    projected_geos = []
    for geo in geometries:
        if isinstance(geo, (Point3D, Point2D)):
            geo = Point3D.from_point2d(geo) if isinstance(geo, Point2D) else geo
            projected_geos.append(plane.project_point(geo))
        elif isinstance(geo, (LineSegment3D, LineSegment2D)):
            geo = LineSegment3D.from_line_segment2d(geo) \
                if isinstance(geo, LineSegment2D) else geo
            st, end = plane.project_points(geo.endpoints)
            projected_geos.append(LineSegment3D.from_end_points(st, end))
        elif isinstance(geo, (Polyline3D, Polyline2D)):
            geo = Polyline3D.from_polyline2d(geo) if isinstance(geo, Polyline2D) else geo
            vertices = plane.project_points(geo.vertices)
            projected_geos.append(Polyline3D(vertices, interpolated=geo.interpolated))
        elif isinstance(geo, (Arc3D, Arc2D)):
            geo = Arc3D.from_arc2d(geo) if isinstance(geo, Arc2D) else geo
            p_line = geo.to_polyline(30, interpolated=True)
            vertices = plane.project_points(p_line.vertices)
            projected_geos.append(Polyline3D(vertices, interpolated=True))
        elif isinstance(geo, Polygon2D):
            vertices = plane.project_points(geo.vertices)
            projected_geos.append(Face3D(vertices))
        elif isinstance(geo, Face3D):
            boundary = plane.project_points(geo.boundary)
            holes = None
            if geo.has_holes:
                holes = [plane.project_points(h) for h in geo.holes]
            projected_geos.append(Face3D(boundary, geo.plane, holes))
        elif isinstance(geo, (Mesh3D, Mesh2D)):
            geo = Mesh3D.from_mesh2d(geo) if isinstance(geo, Mesh2D) else geo
            vertices = plane.project_points(geo.vertices)
            projected_geos.append(Mesh3D(vertices, geo.faces, geo.colors))
        elif isinstance(geo, Polyface3D):
            vertices = plane.project_points(geo.vertices)
            proj_p_face = Polyface3D(vertices, geo.face_indices, geo.edge_information)
            projected_geos.append(proj_p_face)
        elif isinstance(geo, (Ray3D, Ray2D)):
            geo = Ray3D.from_ray2d(geo) if isinstance(geo, Ray2D) else geo
            pt = plane.project_point(geo.p)
            vec = plane.project_point(Point3D(geo.v.x, geo.v.y, geo.v.z))
            projected_geos.append(Ray3D(pt, Vector3D(vec.x, vec.y, vec.z)))
        elif isinstance(geo, (Vector3D, Vector2D)):
            geo = Vector3D.from_vector2d(geo) if isinstance(geo, Vector2D) else geo
            vec = plane.project_point(Point3D(geo.x, geo.y, geo.z))
            projected_geos.append(Vector3D(vec.x, vec.y, vec.z))
        elif isinstance(geo, Plane):
            origin = plane.project_point(geo.o)
            normal = plane.project_point(Point3D(geo.n.x, geo.n.y, geo.n.z))
            normal = Vector3D(normal.x, normal.y, normal.z)
            projected_geos.append(Plane(normal, origin))
        elif isinstance(geo, Sphere):
            center = plane.project_point(geo.center)
            projected_geos.append(Sphere(center, geo.radius))
        elif isinstance(geo, Cone):
            pt = plane.project_point(geo.vertex)
            vec = plane.project_point(Point3D(geo.axis.x, geo.axis.y, geo.axis.z))
            projected_geos.append(Cone(pt, Vector3D(vec.x, vec.y, vec.z), geo.angle))
        elif isinstance(geo, Cylinder):
            pt = plane.project_point(geo.center)
            vec = plane.project_point(Point3D(geo.axis.x, geo.axis.y, geo.axis.z))
            projected_geos.append(Cone(pt, Vector3D(vec.x, vec.y, vec.z), geo.radius))
        else:
            raise ValueError('Unrecognized geometry type {}: {}'.format(type(geo), geo))
    return projected_geos


def project_geometry_2d(plane, geometries):
    """"Project multiple geometries into a plane to get them in the plane's 2D system.

    Args:
        plane: The Plane into which the geometries will be projected.
        geometries: An array of any ladybug_geometry objects which will be projected
            into the plane. 2D geometry objects will retain their 2D classes as
            they are projected into the new system.

    Returns:
        The input geometries projected into the Plane. All coordinate values will
        be in the 2D system of the input plane.
    """
    projected_geos = []
    for geo in geometries:
        if isinstance(geo, (Point3D, Point2D)):
            geo = Point3D.from_point2d(geo) if isinstance(geo, Point2D) else geo
            projected_geos.append(plane.xyz_to_xy(plane.project_point(geo)))
        elif isinstance(geo, (LineSegment3D, LineSegment2D)):
            geo = LineSegment3D.from_line_segment2d(geo) \
                if isinstance(geo, LineSegment2D) else geo
            st, end = plane.project_points(geo.endpoints)
            st, end = plane.xyz_to_xy(st), plane.xyz_to_xy(end)
            projected_geos.append(LineSegment2D.from_end_points(st, end))
        elif isinstance(geo, (Polyline3D, Polyline2D)):
            geo = Polyline3D.from_polyline2d(geo) if isinstance(geo, Polyline2D) else geo
            vertices = plane.project_points(geo.vertices)
            vertices = [plane.xyz_to_xy(pt) for pt in vertices]
            projected_geos.append(Polyline2D(vertices, interpolated=geo.interpolated))
        elif isinstance(geo, (Arc3D, Arc2D)):
            geo = Arc3D.from_arc2d(geo) if isinstance(geo, Arc2D) else geo
            p_line = geo.to_polyline(30, interpolated=True)
            vertices = plane.project_points(p_line.vertices)
            vertices = [plane.xyz_to_xy(pt) for pt in vertices]
            projected_geos.append(Polyline2D(vertices, interpolated=True))
        elif isinstance(geo, Polygon2D):
            vertices = plane.project_points(geo.vertices)
            vertices = [plane.xyz_to_xy(pt) for pt in vertices]
            projected_geos.append(Polygon2D(vertices))
        elif isinstance(geo, Face3D):
            boundary = plane.project_points(geo.boundary)
            boundary = [Point3D.from_point2d(plane.xyz_to_xy(pt)) for pt in boundary]
            holes = None
            if geo.has_holes:
                holes = []
                for h in geo.holes:
                    h = plane.project_points(h)
                    h = [Point3D.from_point2d(plane.xyz_to_xy(pt)) for pt in h]
                    holes.append(h)
            projected_geos.append(Face3D(boundary, geo.plane, holes))
        elif isinstance(geo, (Mesh3D, Mesh2D)):
            geo = Mesh3D.from_mesh2d(geo) if isinstance(geo, Mesh2D) else geo
            vertices = plane.project_points(geo.vertices)
            vertices = [plane.xyz_to_xy(pt) for pt in vertices]
            projected_geos.append(Mesh2D(vertices, geo.faces, geo.colors))
        elif isinstance(geo, Polyface3D):
            vertices = plane.project_points(geo.vertices)
            vertices = [Point3D.from_point2d(plane.xyz_to_xy(pt)) for pt in vertices]
            proj_p_face = Polyface3D(vertices, geo.face_indices, geo.edge_information)
            projected_geos.append(proj_p_face)
        elif isinstance(geo, (Ray3D, Ray2D)):
            geo = Ray3D.from_ray2d(geo) if isinstance(geo, Ray2D) else geo
            pt = plane.xyz_to_xy(plane.project_point(geo.p))
            vec = plane.project_point(Point3D(geo.v.x, geo.v.y, geo.v.z))
            vec = plane.xyz_to_xy(vec)
            projected_geos.append(Ray2D(pt, Vector2D(vec.x, vec.y)))
        elif isinstance(geo, (Vector3D, Vector2D)):
            geo = Vector3D.from_vector2d(geo) if isinstance(geo, Vector2D) else geo
            vec = plane.project_point(Point3D(geo.x, geo.y, geo.z))
            vec = plane.xyz_to_xy(vec)
            projected_geos.append(Vector2D(vec.x, vec.y))
        elif isinstance(geo, Plane):
            origin = plane.xyz_to_xy(plane.project_point(geo.o))
            normal = plane.project_point(Point3D(geo.n.x, geo.n.y, geo.n.z))
            normal = plane.xyz_to_xy(normal)
            normal = Vector3D(normal.x, normal.y)
            projected_geos.append(Plane(normal, Point3D(origin.x, origin.y)))
        elif isinstance(geo, Sphere):
            center = plane.xyz_to_xy(plane.project_point(geo.center))
            center = Point3D.from_point2d(center)
            projected_geos.append(Sphere(center, geo.radius))
        elif isinstance(geo, Cone):
            pt = Point3D.from_point2d(plane.xyz_to_xy(plane.project_point(geo.vertex)))
            vec = plane.project_point(Point3D(geo.axis.x, geo.axis.y, geo.axis.z))
            vec = Point3D.from_point2d(plane.xyz_to_xy(vec))
            projected_geos.append(Cone(pt, Vector3D(vec.x, vec.y, vec.z), geo.angle))
        elif isinstance(geo, Cylinder):
            pt = Point3D.from_point2d(plane.xyz_to_xy(plane.project_point(geo.center)))
            vec = plane.project_point(Point3D(geo.axis.x, geo.axis.y, geo.axis.z))
            vec = Point3D.from_point2d(plane.xyz_to_xy(vec))
            projected_geos.append(Cone(pt, Vector3D(vec.x, vec.y, vec.z), geo.radius))
        else:
            raise ValueError('Unrecognized geometry type {}: {}'.format(type(geo), geo))
    return projected_geos
