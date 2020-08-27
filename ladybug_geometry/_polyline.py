"""Hidden utility functions used by both Polyline2D and Polyline3D classes."""


def _group_vertices(segments, tolerance):
    """Get lists of joined polyline vertices from segments.

    Args:
        segments: An array of LineSegment objects.
        tolerance: The minimum difference in X, Y, and Z values at which Points
            are considred equivalent. Segments with points that match within the
            tolerance will be joined.

    Returns:
        A list of lists vertices that represent joined polylines.
    """
    grouped_verts = []
    base_seg = segments[0]
    remain_segs = list(segments[1:])
    while len(remain_segs) > 0:
        grouped_verts.append(_build_polyline(base_seg, remain_segs, tolerance))
        if len(remain_segs) > 1:
            base_seg = remain_segs[0]
            del remain_segs[0]
        elif len(remain_segs) == 1:  # lone last segment
            grouped_verts.append([remain_segs[0].p1, remain_segs[0].p2])
            del remain_segs[0]
    return grouped_verts


def _build_polyline(base_seg, other_segs, tol):
    """Attempt to build a list of polyline vertices from a base segment.

    Args:
        base_seg: A LineSegment to serve as the base of the Polyline.
        other_segs: A list of other LineSegment objects to attempt to
            connect to the base_seg. This method will delete any segments
            that are successfully connected to the output from this list.
        tol: The tolerance to be used for connecting the line.

    Returns:
        A list of vertices that represent the longest Polyline to which the
        base_seg can be a part of given the other_segs as connections.
    """
    poly_verts = [base_seg.p1, base_seg.p2]
    more_to_check = True
    while more_to_check:
        for i, r_seg in enumerate(other_segs):
            if _connect_seg_to_poly(poly_verts, r_seg, tol):
                del other_segs[i]
                break
        else:
            more_to_check = False
    return poly_verts


def _connect_seg_to_poly(poly_verts, seg, tol):
    """Connect a LineSegment to a list of polyline vertices.

    If successful, a Point will be appended to the poly_verts list and True
    will be returned. If not successful, the poly_verts list will remain unchanged
    and False will be returned.

    Args:
        poly_verts: An ordered list of Points to which the segment should
            be connected.
        seg: A LineSegment to connect to the poly_verts.
        tol: The tolerance to be used for connecting the line.
    """
    p1, p2 = seg.p1, seg.p2
    if poly_verts[-1].is_equivalent(p1, tol):
        poly_verts.append(p2)
        return True
    elif poly_verts[0].is_equivalent(p2, tol):
        poly_verts.insert(0, p1)
        return True
    elif poly_verts[-1].is_equivalent(p2, tol):
        poly_verts.append(p1)
        return True
    elif poly_verts[0].is_equivalent(p1, tol):
        poly_verts.insert(0, p2)
        return True
    return False
