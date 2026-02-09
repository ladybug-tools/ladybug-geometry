"""Utility functions used by several different geometry methods."""
from __future__ import division
import math


def coordinates_hash(point, tolerance):
    """Convert XY coordinates of a Point2D into a string useful for hashing.

    Points that are co-located within the tolerance will receive the same string value
    from this function, which helps convert line segments that contain duplicated
    vertex references them into a singular network object where co-located vertices
    are referenced only once.

    Args:
        point: A Point2D object.
        tolerance: floating point precision tolerance.

    Returns:
        A string of rounded coordinates.
    """
    # get the relative tolerance using a log function
    try:
        rtol = int(math.log10(tolerance)) * -1
    except ValueError:
        rtol = 0  # the tol is equal to 1 (out of range for log)
    # account for the fact that the tolerance may not be base 10
    base = int(tolerance * 10 ** (rtol + 1))
    if base == 10 or base == 0:  # tolerance is base 10 (eg. 0.001)
        base = 1
    else:  # tolerance is not base 10 (eg. 0.003)
        rtol += 1
    # avoid cases of signed zeros messing with the hash
    z_tol = tolerance / 2
    x_val = 0.0 if abs(point.x) < z_tol else point.x
    y_val = 0.0 if abs(point.y) < z_tol else point.y
    # convert the coordinate values to a hash
    return str((
        base * round(x_val / base, rtol),
        base * round(y_val / base, rtol)
    ))


def coordinates_hash_3d(point, tolerance):
    """Convert XY coordinates of a Point3D into a string useful for hashing.

    Points that are co-located within the tolerance will receive the same string value
    from this function, which helps convert line segments that contain duplicated
    vertex references them into a singular network object where co-located vertices
    are referenced only once.

    Args:
        point: A Point3D object.
        tolerance: floating point precision tolerance.

    Returns:
        A string of rounded coordinates.
    """
    # get the relative tolerance using a log function
    try:
        rtol = int(math.log10(tolerance)) * -1
    except ValueError:
        rtol = 0  # the tol is equal to 1 (out of range for log)
    # account for the fact that the tolerance may not be base 10
    base = int(tolerance * 10 ** (rtol + 1))
    if base == 10 or base == 0:  # tolerance is base 10 (eg. 0.001)
        base = 1
    else:  # tolerance is not base 10 (eg. 0.003)
        rtol += 1
    # avoid cases of signed zeros messing with the hash
    z_tol = tolerance / 2
    x_val = 0.0 if abs(point.x) < z_tol else point.x
    y_val = 0.0 if abs(point.y) < z_tol else point.y
    z_val = 0.0 if abs(point.z) < z_tol else point.z
    # convert the coordinate values to a hash
    return str((
        base * round(x_val / base, rtol),
        base * round(y_val / base, rtol),
        base * round(z_val / base, rtol)
    ))
