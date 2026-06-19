"""Utility functions used by several different geometry methods."""
from __future__ import division
import math


def rounding_tolerance(tolerance):
    """Get the number of digits to round coordinate value to given a tolerance.

    To get coordinate values that are perfectly equal to one another within floating
    point tolerance using this method, the outputs should be used in the following way.

    .. code-block:: python

        rounded_coordinate_value = base * round(coordinate_value / base, rtol)

    Args:
        tolerance: A number for the smallest difference in coordinate values
            considered meaningful.

    Returns:
        A tuple with two elements

            -   rtol: An integer for the number of digits to round values to.

            -   base: A number to account for cases where the input tolerance
                is not base 10.
    """
    # get the relative tolerance using a log function
    try:
        rtol = int(math.log10(tolerance)) * -1
    except ValueError:
        rtol = 0  # the tolerance is equal to 1 (out of range for log)
    # account for the fact that the tolerance may not be base 10
    base = int(tolerance * 10 ** (rtol + 1))
    if base == 10 or base == 0:  # tolerance is base 10 (eg. 0.001)
        base = 1
    else:  # tolerance is not base 10 (eg. 0.003)
        rtol += 1
    return rtol, base


def coordinates_hash(point, tolerance):
    """Convert XY coordinates of a Point2D into a string useful for hashing.

    Points that are co-located within the tolerance will receive the same string value
    from this function, which helps convert line segments that contain duplicated
    vertex references them into a singular network object where co-located vertices
    are referenced only once.

    Args:
        point: A Point2D object.
        tolerance: A number for the smallest difference in coordinate values
            considered meaningful.

    Returns:
        A string of rounded coordinates.
    """
    # get the relative tolerance using a log function
    rtol, base = rounding_tolerance(tolerance)
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
        tolerance: A number for the smallest difference in coordinate values
            considered meaningful.
    Returns:
        A string of rounded coordinates.
    """
    # get the relative tolerance using a log function
    rtol, base = rounding_tolerance(tolerance)
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
