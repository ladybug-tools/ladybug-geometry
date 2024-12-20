# coding=utf-8
"""Utility functions for computing bounding boxes and extents around geometry."""
from __future__ import division

from ladybug_geometry.geometry2d.pointvector import Point2D
from ladybug_geometry.geometry3d.pointvector import Point3D


def bounding_domain_x(geometries):
    """Get minimum and maximum X coordinates of multiple geometries.

    Args:
        geometries: An array of any ladybug_geometry objects for which the extents
            of the X domain will be computed. Note that all objects must have
            a min and max property.

    Returns:
        A tuple with the min and the max X coordinates around the geometry.
    """
    min_x, max_x = geometries[0].min.x, geometries[0].max.x
    for geom in geometries[1:]:
        if geom.min.x < min_x:
            min_x = geom.min.x
        if geom.max.x > max_x:
            max_x = geom.max.x
    return min_x, max_x


def bounding_domain_y(geometries):
    """Get minimum and maximum Y coordinates of multiple geometries.

    Args:
        geometries: An array of any ladybug_geometry objects for which the extents
            of the Y domain will be computed. Note that all objects must have
            a min and max property.

    Returns:
        A tuple with the min and the max Y coordinates around the geometry.
    """
    min_y, max_y = geometries[0].min.y, geometries[0].max.y
    for geom in geometries[1:]:
        if geom.min.y < min_y:
            min_y = geom.min.y
        if geom.max.y > max_y:
            max_y = geom.max.y
    return min_y, max_y


def bounding_domain_z(geometries):
    """Get minimum and maximum Z coordinates of multiple geometries.

    Args:
        geometries: An array of any 3D ladybug_geometry objects for which the extents
            of the Z domain will be computed. Note that all objects must have
            a min and max property and they cannot be 2D objects.

    Returns:
        A tuple with the min and the max Z coordinates around the geometry.
    """
    min_z, max_z = geometries[0].min.z, geometries[0].max.z
    for geom in geometries:
        if geom.max.z > max_z:
            max_z = geom.max.z
        if geom.min.z < min_z:
            min_z = geom.min.z
    return min_z, max_z


def bounding_domain_z_2d_safe(geometries):
    """Get minimum and maximum Z coordinates in a manner that is safe for 2D geometries.

    Args:
        geometries: An array of any ladybug_geometry objects for which the extents
            of the Z domain will be computed. Any 2D objects within this list will
            be assumed to have a Z-value of zero.

    Returns:
        A tuple with the min and the max Z coordinates around the geometry.
    """
    try:
        min_z, max_z = geometries[0].min.z, geometries[0].max.z
    except AttributeError:
        min_z, max_z = 0, 0
    for geom in geometries:
        try:
            if geom.max.z > max_z:
                max_z = geom.max.z
            if geom.min.z < min_z:
                min_z = geom.min.z
        except AttributeError:
            if 0 > max_z:
                max_z = 0
            if 0 < min_z:
                min_z = 0
    return min_z, max_z


def _orient_geometry(geometries, axis_angle, center):
    """Orient both 2D and 3D geometry to a given axis angle and center point.

    This is used by the methods that compute bounding rectangles.
    """
    new_geometries = []
    for geom in geometries:
        try:  # assume that it is a 2D geometry object
            new_geometries.append(geom.rotate(-axis_angle, center))
        except TypeError:  # it's a 3D geometry object
            new_geometries.append(geom.rotate_xy(-axis_angle, center))
    return new_geometries


def bounding_rectangle(geometries, axis_angle=0):
    """Get the min and max of an oriented bounding rectangle around 2D or 3D geometry.

    Args:
        geometries: An array of 2D or 3D geometry objects. Note that all objects
            must have a min and max property.
        axis_angle: The counter-clockwise rotation angle in radians in the XY plane
            to represent the orientation of the bounding rectangle extents. (Default: 0).

    Returns:
        A tuple with two Point2D objects representing the min point and max point
        of the bounding rectangle respectively.
    """
    if axis_angle != 0:  # rotate geometry to the bounding box
        cpt = geometries[0].vertices[0]
        geometries = _orient_geometry(geometries, axis_angle, cpt)
    xx = bounding_domain_x(geometries)
    yy = bounding_domain_y(geometries)
    min_pt = Point2D(xx[0], yy[0])
    max_pt = Point2D(xx[1], yy[1])
    if axis_angle != 0:  # rotate the points back
        cpt = Point2D(cpt.x, cpt.y)  # cast Point3D to Point2D
        min_pt = min_pt.rotate(axis_angle, cpt)
        max_pt = max_pt.rotate(axis_angle, cpt)
    return min_pt, max_pt


def bounding_rectangle_extents(geometries, axis_angle=0):
    """Get the width and length of an oriented bounding rectangle around 2D or 3D geometry.

    Args:
        geometries: An array of 2D or 3D geometry objects. Note that all objects
            must have a min and max property.
        axis_angle: The counter-clockwise rotation angle in radians in the XY plane
            to represent the orientation of the bounding rectangle extents. (Default: 0).

    Returns:
        A tuple with 2 values corresponding to the width and length of the bounding
        rectangle.
    """
    if axis_angle != 0:
        cpt = geometries[0].vertices[0]
        geometries = _orient_geometry(geometries, axis_angle, cpt)
    xx = bounding_domain_x(geometries)
    yy = bounding_domain_y(geometries)
    return xx[1] - xx[0], yy[1] - yy[0]


def bounding_box(geometries, axis_angle=0):
    """Get the min and max of an oriented bounding box around 3D geometry.

    Args:
        geometries: An array of 3D geometry objects. Note that all objects must
            have a min and max property.
        axis_angle: The counter-clockwise rotation angle in radians in the XY plane
            to represent the orientation of the bounding box extents. (Default: 0).

    Returns:
        A tuple with two Point3D objects representing the min point and max point
        of the bounding box respectively.
    """
    if axis_angle != 0:  # rotate geometry to the bounding box
        cpt = geometries[0].vertices[0]
        geometries = [geom.rotate_xy(-axis_angle, cpt) for geom in geometries]
    xx = bounding_domain_x(geometries)
    yy = bounding_domain_y(geometries)
    zz = bounding_domain_z_2d_safe(geometries)
    min_pt = Point3D(xx[0], yy[0], zz[0])
    max_pt = Point3D(xx[1], yy[1], zz[1])
    if axis_angle != 0:  # rotate the points back
        min_pt = min_pt.rotate_xy(axis_angle, cpt)
        max_pt = max_pt.rotate_xy(axis_angle, cpt)
    return min_pt, max_pt


def bounding_box_extents(geometries, axis_angle=0):
    """Get the width, length and height of an oriented bounding box around 3D geometry.

    Args:
        geometries: An array of 3D geometry objects. Note that all objects must
            have a min and max property.
        axis_angle: The counter-clockwise rotation angle in radians in the XY plane
            to represent the orientation of the bounding box extents. (Default: 0).

    Returns:
        A tuple with 3 values corresponding to the width, length and height of
        the bounding box.
    """
    if axis_angle != 0:
        cpt = geometries[0].vertices[0]
        geometries = [geom.rotate_xy(-axis_angle, cpt) for geom in geometries]
    xx = bounding_domain_x(geometries)
    yy = bounding_domain_y(geometries)
    zz = bounding_domain_z_2d_safe(geometries)
    return xx[1] - xx[0], yy[1] - yy[0], zz[1] - zz[0]


def overlapping_bounding_rect(geometry_1, geometry_2, distance):
    """Check if the bounding rectangles of two geometries overlap within a tolerance.

    Args:
        geometry_1: The first geometry to check.
        geometry_2: The second geometry to check.
        distance: The maximum distance at which the geometries are
            considered overlapping.

    Return:
        True if the geometries overlap. False if they do not.
    """
    # Bounding box check using the Separating Axis Theorem
    geo1_width = geometry_1.max.x - geometry_1.min.x
    geo2_width = geometry_2.max.x - geometry_2.min.x
    dist_btwn_x = abs(geometry_1.center.x - geometry_2.center.x)
    x_gap_btwn_box = dist_btwn_x - (0.5 * geo1_width) - (0.5 * geo2_width)
    if x_gap_btwn_box > distance:
        return False   # overlap impossible

    geo1_depth = geometry_1.max.y - geometry_1.min.y
    geo2_depth = geometry_2.max.y - geometry_2.min.y
    dist_btwn_y = abs(geometry_1.center.y - geometry_2.center.y)
    y_gap_btwn_box = dist_btwn_y - (0.5 * geo1_depth) - (0.5 * geo2_depth)
    if y_gap_btwn_box > distance:
        return False   # overlap impossible

    return True  # overlap exists


def overlapping_bounding_boxes(geometry_1, geometry_2, distance):
    """Check if the bounding boxes around two geometries overlap within a distance.

    Args:
        geometry_1: The first geometry to check.
        geometry_2: The second geometry to check.
        distance: The maximum distance at which the geometries are
            considered overlapping.

    Return:
        True if the geometries overlap. False if they do not.
    """
    # Bounding box check using the Separating Axis Theorem
    geo1_width = geometry_1.max.x - geometry_1.min.x
    geo2_width = geometry_2.max.x - geometry_2.min.x
    dist_btwn_x = abs(geometry_1.center.x - geometry_2.center.x)
    x_gap_btwn_box = dist_btwn_x - (0.5 * geo1_width) - (0.5 * geo2_width)
    if x_gap_btwn_box > distance:
        return False   # overlap impossible

    geo1_depth = geometry_1.max.y - geometry_1.min.y
    geo2_depth = geometry_2.max.y - geometry_2.min.y
    dist_btwn_y = abs(geometry_1.center.y - geometry_2.center.y)
    y_gap_btwn_box = dist_btwn_y - (0.5 * geo1_depth) - (0.5 * geo2_depth)
    if y_gap_btwn_box > distance:
        return False   # overlap impossible

    geo1_height = geometry_1.max.z - geometry_1.min.z
    geo2_height = geometry_2.max.z - geometry_2.min.z
    dist_btwn_z = abs(geometry_1.center.z - geometry_2.center.z)
    z_gap_btwn_box = dist_btwn_z - (0.5 * geo1_height) - (0.5 * geo2_height)
    if z_gap_btwn_box > distance:
        return False   # overlap impossible

    return True  # overlap exists
