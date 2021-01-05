"""Core triangulation functions used by various geometry modules.

The functions here are derived from the earcut-python library available at
https://github.com/joshuaskelly/earcut-python

The earcut-python library is, itself, a pure Python port of the earcut JavaScript
triangulation library maintained by Mapbox. The original project can be found at
https://github.com/mapbox/earcut

The version here is based off of the JavaScript earcut 2.1.1 release, and is
functionally identical.
"""
import math


def earcut(data, hole_indices=None, dim=2):
    """Triangulate a list of vertices that make up a shape, either with or without holes.
    
    Args:
        data: A flat array of vertex coordinates like [x0,y0, x1,y1, x2,y2, ...].
        hole_indices: A flat array of the starting indices for each hole. For example,
            [5, 8] for a 12-vertex input would mean one hole with vertices 5-7
            and another with 8-11. If a single vertex is passed as as a hole,
            Earcut treats it as a Steiner point. If None, no holes will be assumed
            for the shape. (Default: None).
        dim: An integer for the number of coordinates per vertex in the input
            array. For example, 3 means each vertex exists in 3D space with
            XX, Y, Z coordinates. (Default: 2 for 2D coordinates).
    """
    dim = dim or 2
    hasHoles = hole_indices and len(hole_indices)
    outerLen =  hole_indices[0] * dim if hasHoles else len(data)
    outerNode = _linked_list(data, 0, outerLen, dim, True)
    triangles = []

    if not outerNode:
        return triangles

    minX = None
    minY = None
    maxX = None
    maxY = None
    x = None
    y = None
    size = None

    if hasHoles:
        outerNode = _eliminate_holes(data, hole_indices, outerNode, dim)

    # if the shape is not too simple, we'll use z-order curve hash later
    if (len(data) > 80 * dim):  # calculate polygon bbox
        minX = maxX = data[0]
        minY = maxY = data[1]

        for i in range(dim, outerLen, dim):
            x = data[i]
            y = data[i + 1]
            if x < minX:
                minX = x
            if y < minY:
                minY = y
            if x > maxX:
                maxX = x
            if y > maxY:
                maxY = y

        # minX, minY and size are later used to transform coords into integers
        # integers are used for z-order calculation
        size = max(maxX - minX, maxY - minY)

    _earcut_linked(outerNode, triangles, dim, minX, minY, size)

    return triangles


def _linked_list(data, start, end, dim, clockwise):
    """Create a circular doubly linked list from polygon points.

    Points will be in the specified winding order.
    """
    i = None
    last = None

    if (clockwise == (_signed_area(data, start, end, dim) > 0)):
        for i in range(start, end, dim):
            last = _insert_node(i, data[i], data[i + 1], last)

    else:
        for i in reversed(range(start, end, dim)):
            last = _insert_node(i, data[i], data[i + 1], last)

    if (last and _equals(last, last.next)):
        _remove_node(last)
        last = last.next

    return last


def _signed_area(data, start, end, dim):
    """Get the signed area from a list of coordinate data."""
    sum = 0
    j = end - dim

    for i in range(start, end, dim):
        sum += (data[j] - data[i]) * (data[i + 1] + data[j + 1])
        j = i

    return sum


def _filter_points(start, end=None):
    """Eliminate colinear or duplicate points."""
    if not start:
        return start
    if not end:
        end = start

    p = start
    again = True

    while again or p != end:
        again = False

        if (not p.steiner and (_equals(p, p.next) or _area(p.prev, p, p.next) == 0)):
            _remove_node(p)
            p = end = p.prev
            if (p == p.next):
                return None

            again = True

        else:
            p = p.next

    return end


def _earcut_linked(ear, triangles, dim, minX, minY, size, _pass=None):
    """Main ear slicing loop which triangulates a polygon (given as a linked list)."""
    if not ear:
        return

    # interlink polygon nodes in z-order
    if not _pass and size:
        _index_curve(ear, minX, minY, size)

    stop = ear
    prev = None
    next = None

    # iterate through ears, slicing them one by one
    while ear.prev != ear.next:
        prev = ear.prev
        next = ear.next

        if _is_ear_hashed(ear, minX, minY, size) if size else _is_ear(ear):
            # cut off the triangle
            triangles.append(prev.i // dim)
            triangles.append(ear.i // dim)
            triangles.append(next.i // dim)

            _remove_node(ear)

            # skipping the next vertice leads to less sliver triangles
            ear = next.next
            stop = next.next

            continue

        ear = next

        # if we looped through the whole remaining polygon and can't find any more ears
        if ear == stop:
            # try filtering points and slicing again
            if not _pass:
                _earcut_linked(_filter_points(ear), triangles, dim, minX, minY, size, 1)

                # if this didn't work, try curing all small self-intersections locally
            elif _pass == 1:
                ear = _cure_local_intersections(ear, triangles, dim)
                _earcut_linked(ear, triangles, dim, minX, minY, size, 2)

                # as a last resort, try splitting the remaining polygon into two
            elif _pass == 2:
                _split_earcut(ear, triangles, dim, minX, minY, size)

            break


def _is_ear(ear):
    """Check whether a polygon node forms a valid ear with adjacent nodes."""
    a = ear.prev
    b = ear
    c = ear.next

    if _area(a, b, c) >= 0:
        return False # reflex, can't be an ear

    # now make sure we don't have other points inside the potential ear
    p = ear.next.next

    while p != ear.prev:
        if _point_in_triangle(a.x, a.y, b.x, b.y, c.x, c.y, p.x, p.y) and \
                _area(p.prev, p, p.next) >= 0:
            return False
        p = p.next

    return True


def _is_ear_hashed(ear, minX, minY, size):
    """Check whether a polygon node forms a valid ear using hashes."""
    a = ear.prev
    b = ear
    c = ear.next

    if _area(a, b, c) >= 0:
        return False # reflex, can't be an ear

    # triangle bbox; min & max are calculated like this for speed
    minTX = (a.x if a.x < c.x else c.x) if a.x < b.x else (b.x if b.x < c.x else c.x)
    minTY = (a.y if a.y < c.y else c.y) if a.y < b.y else (b.y if b.y < c.y else c.y)
    maxTX = (a.x if a.x > c.x else c.x) if a.x > b.x else (b.x if b.x > c.x else c.x)
    maxTY = (a.y if a.y > c.y else c.y) if a.y > b.y else (b.y if b.y > c.y else c.y)

    # z-order range for the current triangle bbox;
    minZ = _z_order(minTX, minTY, minX, minY, size)
    maxZ = _z_order(maxTX, maxTY, minX, minY, size)

    # first look for points inside the triangle in increasing z-order
    p = ear.nextZ

    while p and p.z <= maxZ:
        if p != ear.prev and p != ear.next and \
                _point_in_triangle(a.x, a.y, b.x, b.y, c.x, c.y, p.x, p.y) and \
                _area(p.prev, p, p.next) >= 0:
            return False
        p = p.nextZ

    # then look for points in decreasing z-order
    p = ear.prevZ

    while p and p.z >= minZ:
        if p != ear.prev and p != ear.next and \
                _point_in_triangle(a.x, a.y, b.x, b.y, c.x, c.y, p.x, p.y) and \
                _area(p.prev, p, p.next) >= 0:
            return False
        p = p.prevZ

    return True


def _cure_local_intersections(start, triangles, dim):
    """Go through all polygon nodes and cure small local self-intersections."""
    do = True
    p = start

    while do or p != start:
        do = False

        a = p.prev
        b = p.next.next

        if not _equals(a, b) and _intersects(a, p, p.next, b) and \
                _locally_inside(a, b) and _locally_inside(b, a):
            triangles.append(a.i // dim)
            triangles.append(p.i // dim)
            triangles.append(b.i // dim)

            # remove two nodes involved
            _remove_node(p)
            _remove_node(p.next)

            p = start = b

        p = p.next

    return p


def _split_earcut(start, triangles, dim, minX, minY, size):
    """try splitting polygon into two and triangulate them independently."""
    # look for a valid diagonal that divides the polygon into two
    do = True
    a = start

    while do or a != start:
        do = False
        b = a.next.next

        while b != a.prev:
            if a.i != b.i and _is_valid_diagonal(a, b):
                # split the polygon in two by the diagonal
                c = _split_polygon(a, b)

                # filter colinear points around the cuts
                a = _filter_points(a, a.next)
                c = _filter_points(c, c.next)

                # run earcut on each half
                _earcut_linked(a, triangles, dim, minX, minY, size)
                _earcut_linked(c, triangles, dim, minX, minY, size)
                return

            b = b.next

        a = a.next


def _eliminate_holes(data, hole_indices, outerNode, dim):
    """Link holes into the outer loop, producing a single-ring polygon without holes."""
    queue = []
    i = None
    _len = len(hole_indices)
    start = None
    end = None
    _list = None

    for i in range(len(hole_indices)):
        start = hole_indices[i] * dim
        end =  hole_indices[i + 1] * dim if i < _len - 1 else len(data)
        _list = _linked_list(data, start, end, dim, False)

        if (_list == _list.next):
            _list.steiner = True

        queue.append(_get_leftmost(_list))

    queue = sorted(queue, key=lambda i: i.x)

    # process holes from left to right
    for i in range(len(queue)):
        _eliminate_hole(queue[i], outerNode)
        outerNode = _filter_points(outerNode, outerNode.next)

    return outerNode


def _eliminate_hole(hole, outerNode):
    """Find a bridge between vertices that connects hole with an outer ring.
    
    Return a shape with the hole linked into it."""
    outerNode = _find_hole_bridge(hole, outerNode)
    if outerNode:
        b = _split_polygon(outerNode, hole)
        _filter_points(b, b.next)


def _find_hole_bridge(hole, outerNode):
    """David Eberly's algorithm for finding a bridge between hole and outer polygon."""
    do = True
    p = outerNode
    hx = hole.x
    hy = hole.y
    qx = float('-inf')
    m = None

    # find a segment intersected by a ray from the hole's leftmost point to the left;
    # segment's endpoint with lesser x will be potential connection point
    while do or p != outerNode:
        do = False
        if hy <= p.y and hy >= p.next.y and p.next.y - p.y != 0:
            x = p.x + (hy - p.y) * (p.next.x - p.x) / (p.next.y - p.y)

            if x <= hx and x > qx:
                qx = x

                if (x == hx):
                    if hy == p.y:
                        return p
                    if hy == p.next.y:
                        return p.next

                m = p if p.x < p.next.x else p.next

        p = p.next

    if not m:
        return None

    if hx == qx:
        return m.prev # hole touches outer segment; pick lower endpoint

    # check points inside the triangle of hole point, segment intersection and endpoint
    # if there are no points found, we have a valid connection
    # otherwise choose the point of the minimum angle with the ray as connection point

    stop = m
    mx = m.x
    my = m.y
    tanMin = float('inf')
    tan = None

    p = m.next

    while p != stop:
        hx_or_qx = hx if hy < my else qx
        qx_or_hx = qx if hy < my else hx

        if hx >= p.x and p.x >= mx and \
                _point_in_triangle(hx_or_qx, hy, mx, my, qx_or_hx, hy, p.x, p.y):
            tan = abs(hy - p.y) / (hx - p.x) # tangential

            if (tan < tanMin or (tan == tanMin and p.x > m.x)) and \
                    _locally_inside(p, hole):
                m = p
                tanMin = tan

        p = p.next

    return m


def _index_curve(start, minX, minY, size):
    """Interlink polygon nodes in z-order."""
    do = True
    p = start

    while do or p != start:
        do = False

        if p.z == None:
            p.z = _z_order(p.x, p.y, minX, minY, size)

        p.prevZ = p.prev
        p.nextZ = p.next
        p = p.next

    p.prevZ.nextZ = None
    p.prevZ = None

    _sort_linked(p)


def _sort_linked(_list):
    """Simon Tatham's linked list merge sort algorithm.

    More information available at https://www.chiark.greenend.org.uk/
    """
    do = True
    i = None
    p = None
    q = None
    e = None
    tail = None
    numMerges = None
    pSize = None
    qSize = None
    inSize = 1

    while do or numMerges > 1:
        do = False
        p = _list
        _list = None
        tail = None
        numMerges = 0

        while p:
            numMerges += 1
            q = p
            pSize = 0
            for i in range(inSize):
                pSize += 1
                q = q.nextZ
                if not q:
                    break

            qSize = inSize

            while pSize > 0 or (qSize > 0 and q):

                if pSize == 0:
                    e = q
                    q = q.nextZ
                    qSize -= 1

                elif (qSize == 0 or not q):
                    e = p
                    p = p.nextZ
                    pSize -= 1

                elif (p.z <= q.z):
                    e = p
                    p = p.nextZ
                    pSize -= 1

                else:
                    e = q
                    q = q.nextZ
                    qSize -= 1

                if tail:
                    tail.nextZ = e

                else:
                    _list = e

                e.prevZ = tail
                tail = e

            p = q

        tail.nextZ = None
        inSize *= 2

    return _list


def _z_order(x, y, minX, minY, size):
    """Z-order of a point given coords and size of the data bounding box."""
    # coords are transformed into non-negative 15-bit integer range
    x = int(32767 * (x - minX) // size)
    y = int(32767 * (y - minY) // size)

    x = (x | (x << 8)) & 0x00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F
    x = (x | (x << 2)) & 0x33333333
    x = (x | (x << 1)) & 0x55555555

    y = (y | (y << 8)) & 0x00FF00FF
    y = (y | (y << 4)) & 0x0F0F0F0F
    y = (y | (y << 2)) & 0x33333333
    y = (y | (y << 1)) & 0x55555555

    return x | (y << 1)


def _get_leftmost(start):
    """Find the leftmost node of a polygon ring."""
    do = True
    p = start
    leftmost = start

    while do or p != start:
        do = False
        if p.x < leftmost.x:
            leftmost = p
        p = p.next

    return leftmost


def _point_in_triangle(ax, ay, bx, by, cx, cy, px, py):
    """Check if a point lies within a convex triangle."""
    return (cx - px) * (ay - py) - (ax - px) * (cy - py) >= 0 and \
        (ax - px) * (by - py) - (bx - px) * (ay - py) >= 0 and \
        (bx - px) * (cy - py) - (cx - px) * (by - py) >= 0


def _is_valid_diagonal(a, b):
    """Check if a diagonal between two polygon nodes is valid.

    A valid diagonal is defined as one that lies in polygon interior.
    """
    return a.next.i != b.i and a.prev.i != b.i and not _intersects_polygon(a, b) and \
        _locally_inside(a, b) and _locally_inside(b, a) and _middle_inside(a, b)


def _area(p, q, r):
    """Get the signed area of a triangle."""
    return (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)


def _equals(p1, p2):
    """Check if two points are equal."""
    return p1.x == p2.x and p1.y == p2.y


def _intersects(p1, q1, p2, q2):
    """Check if two segments intersect."""
    if (_equals(p1, q1) and _equals(p2, q2)) or (_equals(p1, q2) and _equals(p2, q1)):
        return True

    return _area(p1, q1, p2) > 0 != _area(p1, q1, q2) > 0 and \
        _area(p2, q2, p1) > 0 != _area(p2, q2, q1) > 0


def _intersects_polygon(a, b):
    """Check if a polygon diagonal intersects any polygon segments."""
    do = True
    p = a

    while do or p != a:
        do = False
        if (p.i != a.i and p.next.i != a.i and p.i != b.i and p.next.i != b.i and \
                _intersects(p, p.next, a, b)):
            return True

        p = p.next

    return False


def _locally_inside(a, b):
    """Check if a polygon diagonal is locally inside the polygon."""
    if _area(a.prev, a, a.next) < 0:
        return _area(a, b, a.next) >= 0 and _area(a, a.prev, b) >= 0
    else:
        return _area(a, b, a.prev) < 0 or _area(a, a.next, b) < 0


def _middle_inside(a, b):
    """Check if the middle point of a polygon diagonal is inside a polygon."""
    do = True
    p = a
    inside = False
    px = (a.x + b.x) / 2
    py = (a.y + b.y) / 2

    while do or p != a:
        do = False
        if ((p.y > py) != (p.next.y > py)) and \
                (px < (p.next.x - p.x) * (py - p.y) / (p.next.y - p.y) + p.x):
            inside = not inside

        p = p.next

    return inside


def _split_polygon(a, b):
    """Link two polygon vertices with a bridge.

    If the vertices belong to the same ring, the polygon will be split into two.
    If one belongs to the outer ring and another to a hole, the hole will be merged
    into a single ring.
    """
    a2 = _Node(a.i, a.x, a.y)
    b2 = _Node(b.i, b.x, b.y)
    an = a.next
    bp = b.prev

    a.next = b
    b.prev = a

    a2.next = an
    an.prev = a2

    b2.next = a2
    a2.prev = b2

    bp.next = b2
    b2.prev = bp

    return b2


def _insert_node(i, x, y, last):
    """Create a node and optionally link it with previous one.

    Linking is done in a circular doubly linked list.
    """
    p = _Node(i, x, y)

    if not last:
        p.prev = p
        p.next = p

    else:
        p.next = last.next
        p.prev = last
        last.next.prev = p
        last.next = p

    return p


def _remove_node(p):
    """Remove a node from a list."""
    p.next.prev = p.prev
    p.prev.next = p.next

    if p.prevZ:
        p.prevZ.nextZ = p.nextZ

    if p.nextZ:
        p.nextZ.prevZ = p.prevZ


class _Node(object):
    """Node within a coordinate array."""

    def __init__(self, i, x, y):
        # vertex index in coordinates array
        self.i = i

        # vertex coordinates
        self.x = x
        self.y = y

        # previous and next vertice nodes in a polygon ring
        self.prev = None
        self.next = None

        # z-order curve value
        self.z = None

        # previous and next nodes in z-order
        self.prevZ = None
        self.nextZ = None

        # indicates whether this is a steiner point
        self.steiner = False
