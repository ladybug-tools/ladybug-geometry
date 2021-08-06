# coding=utf-8
"""Utilities to convert any Ladybug Geometry dictionary to Python objects.

Note that importing this module will import almost all modules within the
Ladybug_geometry library in order to be able to re-serialize almost any 
dictionary produced from the library.
"""

from ladybug_geometry.geometry2d import Vector2D, Point2D, Ray2D, \
                            LineSegment2D, Arc2D, Polyline2D, Polygon2D, Mesh2D
from ladybug_geometry.geometry3d import Vector3D, Point3D, Ray3D, \
                            LineSegment3D, Arc3D, Polyline3D, Polyface3D, Mesh3D,\
                            Plane, Face3D, Sphere, Cone, Cylinder

def geometry_dict_to_object(ladybug_geom_dict, raise_exception=True):
    """
    Args:
        ladybug_geom_dict (dict): A dictionary of any Ladybug Geometry object.
        raise_exception (bool): Boolean to note whether an exception should be raised
            if the object is not identified as a part of ladybug_geometry.
            Default: True.

    Returns:
        A Python object derived from the input ladybug_geom_dict. 
    """

    lbt_types = {
        'Vector2D': Vector2D,
        'Point2D': Point2D,
        'Ray2D': Ray2D,
        'LineSegment2D': LineSegment2D,
        'Arc2D': Arc2D,
        'Polyline2D': Polyline2D,
        'Polygon2D': Polygon2D,
        'Mesh2D': Mesh2D,
        'Vector3D': Vector3D,
        'Point3D': Point3D,
        'Ray3D': Ray3D,
        'LineSegment3D': LineSegment3D,
        'Arc3D': Arc3D,
        'Polyline3D': Polyline3D,
        'Mesh3D': Mesh3D,
        'Plane': Plane,
        'Polyface3D': Polyface3D,
        'Face3D':Face3D,
        'Sphere':Sphere,
        'Cone':Cone,
        'Cylinder':Cylinder,
    }

    # Get the ladybug_geometry object 'Type'
    try:
        obj_type = ladybug_geom_dict['type']
    except KeyError:
        raise ValueError('Ladybug dictionary lacks required "type" key.')

    # Build a new Ladybug Python Object based on the "Type"
    try:
        lbt_class = lbt_types[obj_type]
        return lbt_class.from_dict( ladybug_geom_dict )
    except KeyError:
        if raise_exception:
            raise ValueError('{} is not a recognized ladybug geometry type'.format(obj_type))
        else:
            return None