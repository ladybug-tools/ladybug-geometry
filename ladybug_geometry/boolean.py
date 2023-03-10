# coding=utf-8
"""Module for boolean operations on 2D polygons (union, intersection, difference, xor).

The functions here are derived from the pypolybool python library available at
https://github.com/KaivnD/pypolybool

The pypolybool library is, itself, a pure Python port of the polybooljs JavaScript
library maintained by Sean Mconnelly available at https://github.com/velipso/polybooljs

Full documentation of the method is available at
https://sean.cm/a/polygon-clipping-pt2

Based somewhat on the F. Martinez (2013) algorithm.

Francisco Martínez, Carlos Ogayar, Juan R. Jiménez, Antonio J. Rueda (2013),
"A simple algorithm for Boolean operations on polygons",
Advances in Engineering Software, Volume 64, Pages 11-19, ISSN 0965-9978,
https://doi.org/10.1016/j.advengsoft.2013.04.004.
"""
from __future__ import division


"""____________OBJECTS FOR INPUT/OUTPUT FROM BOOLEAN OPERATIONS____________"""


class BooleanPoint:
    """2D Point class used in polygon boolean operations.

    Args:
        x: Float for X coordinate.
        y: Float for Y coordinate
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def is_equivalent(self, other_pt, tol):
        """Check if this point is equivalent ot another within tolerance."""
        if not isinstance(other_pt, BooleanPoint):
            return False
        return abs(self.x - other_pt.x) < tol and abs(self.y - other_pt.y) < tol

    @staticmethod
    def collinear(pt1, pt2, pt3, tolerance):
        """Get a boolean for whether 3 points are colinear."""
        dx1 = pt1.x - pt2.x
        dy1 = pt1.y - pt2.y
        dx2 = pt2.x - pt3.x
        dy2 = pt2.y - pt3.y
        return abs(dx1 * dy2 - dx2 * dy1) < tolerance

    @staticmethod
    def compare(pt1, pt2, tolerance):
        """Get an integer for the relationship between two points.

        Zero indicates equal. Positive 1 is to the right. Negative 1 is to the left.
        """
        if abs(pt1.x - pt2.x) < tolerance:
            return 0 if abs(pt1.y - pt2.y) < tolerance else -1 if pt1.y < pt2.y else 1
        return -1 if pt1.x < pt2.x else 1

    @staticmethod
    def point_above_or_on_line(point, left, right, tolerance):
        """Get a boolean for whether a point is above or on a line.

        Args:
            point: BooleanPoint to be evaluated.
            left: BooleanPoint for the left of the line segment.
            right: BooleanPoint for the right of the line segment
        """
        return (right.x - left.x) * (point.y - left.y) - (right.y - left.y) * (
            point.x - left.x
        ) >= -tolerance

    @staticmethod
    def between(point, left, right, tolerance):
        """Get a boolean for whether a point is between two points.

        Args:
            point: BooleanPoint to be evaluated.
            left: BooleanPoint for the left.
            right: BooleanPoint for the right.
        """
        dPyLy = point.y - left.y
        dRxLx = right.x - left.x
        dPxLx = point.x - left.x
        dRyLy = right.y - left.y

        dot = dPxLx * dRxLx + dPyLy * dRyLy
        if dot < tolerance:
            return False

        sqlen = dRxLx * dRxLx + dRyLy * dRyLy
        if dot - sqlen > -tolerance:
            return False

        return True

    @staticmethod
    def _lines_intersect(a0, a1, b0, b1, tolerance):
        """Get an _IntersectionPoint object for the intersection of two line segments.

        Args:
            a0: BooleanPoint for the first point of the first line segment.
            a1: BooleanPoint for the second point of the first line segment.
            b0: BooleanPoint for the first point of the second line segment.
            b1: BooleanPoint for the second point of the second line segment.
        """
        adx = a1.x - a0.x
        ady = a1.y - a0.y
        bdx = b1.x - b0.x
        bdy = b1.y - b0.y

        axb = adx * bdy - ady * bdx

        if abs(axb) < tolerance:
            return None

        dx = a0.x - b0.x
        dy = a0.y - b0.y

        a = (bdx * dy - bdy * dx) / axb
        b = (adx * dy - ady * dx) / axb

        return _IntersectionPoint(
            BooleanPoint.__calc_along_using_value(a, tolerance),
            BooleanPoint.__calc_along_using_value(b, tolerance),
            BooleanPoint(a0.x + a * adx, a0.y + a * ady),
        )

    @staticmethod
    def __calc_along_using_value(value, tolerance):
        if value <= -tolerance:
            return -2
        elif value < tolerance:
            return -1
        elif value - 1 <= -tolerance:
            return 0
        elif value - 1 < tolerance:
            return 1
        else:
            return 2

    def __repr__(self):
        return "{},{}".format(self.x, self.y)

    def __str__(self):
        return "{},{}".format(self.x, self.y)


class BooleanPolygon:
    """Polygon class used in polygon boolean operations.

    Args:
        regions: A list of lists of BooleanPoints representing the 2D points defining
            the regions of the Polygon. The first sub-list is typically the
            boundary of the polygon and each successive list represents a hole
            within the boundary. It is also permissable for the holes
            to lie outside the first polygon, in which case the shape is
            interpreted as a MultiPolygon. As an alternative to BooleanPoints,
            tuples of two float values are also permissable in which case the
            values represent the X and Y coordinates of each vertex.
        is_inverted: A boolean for whether the Polygon is inverted or not. For
            polygons input to the boolean methods, this value should always be
            False. (Default: False)
    """

    def __init__(self, regions, is_inverted=False):
        _regions = []
        for region in regions:
            tmp = []
            for pt in region:
                if isinstance(pt, BooleanPoint):
                    tmp.append(pt)
                elif isinstance(pt, tuple):
                    x, y = pt
                    tmp.append(BooleanPoint(x, y))
            _regions.append(tmp)

        self.regions = _regions
        self.is_inverted = is_inverted


"""____________PRIMARY COMPUTATION CLASSES AND METHODS____________"""


class _Fill:
    def __init__(self, below=None, above=None):
        self.below = below
        self.above = above

    def __repr__(self):
        return "{},{}".format(self.above, self.below)

    def __str__(self):
        return "{},{}".format(self.above, self.below)


class _Segment:
    def __init__(self, start, end, myfill=None, otherfill=None):
        self.start = start
        self.end = end
        self.myfill = myfill
        self.otherfill = otherfill

    def __repr__(self):
        return "S: {}, E: {}".format(self.start, self.end)

    def __str__(self):
        return "S: {}, E: {}".format(self.start, self.end)


class _PolySegments:
    def __init__(self, segments=None, is_inverted=False):
        self.segments = segments
        self.is_inverted = is_inverted


class _CombinedPolySegments:
    def __init__(self, combined=None, is_inverted1=False, is_inverted2=False):
        self.combined = combined
        self.is_inverted1 = is_inverted1
        self.is_inverted2 = is_inverted2


class _Matcher:
    def __init__(self, index, matchesHead, matchesPt1):
        self.index = index
        self.matchesHead = matchesHead
        self.matchesPt1 = matchesPt1


class _IntersectionPoint:
    def __init__(self, alongA, alongB, pt):
        self.alongA = alongA
        self.alongB = alongB
        self.pt = pt


class _Node:
    def __init__(
        self, isRoot=False, isStart=False, pt=None, seg=None, primary=False,
        next=None, previous=None, other=None, ev=None, status=None, remove=None,
    ):
        self.status = status
        self.other = other
        self.ev = ev
        self.previous = previous
        self.next = next
        self.isRoot = isRoot
        self.remove = remove
        self.isStart = isStart
        self.pt = pt
        self.seg = seg
        self.primary = primary


class _Transition:
    def __init__(self, after, before, insert):
        self.after = after
        self.before = before
        self.insert = insert


class _LinkedList:
    def __init__(self):
        self.__root = _Node(isRoot=True)

    def exists(self, node):
        if node is None or node is self.__root:
            return False
        return True

    def isEmpty(self):
        return self.__root.next is None

    def getHead(self):
        return self.__root.next

    def insertBefore(self, node, check):
        last = self.__root
        here = self.__root.next

        while here is not None:
            if check(here):
                node.previous = here.previous
                node.next = here
                here.previous.next = node
                here.previous = node
                return
            last = here
            here = here.next
        last.next = node
        node.previous = last
        node.next = None

    def findTransition(self, check):
        previous = self.__root
        here = self.__root.next

        while here is not None:
            if check(here):
                break
            previous = here
            here = here.next

        def insert_func(node):
            node.previous = previous
            node.next = here
            previous.next = node
            if here is not None:
                here.previous = node
            return node

        return _Transition(
            before=(None if previous is self.__root else previous),
            after=here,
            insert=insert_func,
        )

    @staticmethod
    def node(data):
        data.previous = None
        data.next = None

        def remove_func():
            data.previous.next = data.next
            if data.next is not None:
                data.next.previous = data.previous
            data.previous = None
            data.next = None

        data.remove = remove_func
        return data


class _Intersecter:
    """Primary intersection class."""

    def __init__(self, selfIntersection, tol):
        self.selfIntersection = selfIntersection
        self.tol = tol
        self.__eventRoot = _LinkedList()

    def newsegment(self, start, end):
        return _Segment(start=start, end=end, myfill=_Fill())

    def segmentCopy(self, start, end, seg):
        return _Segment(
            start=start, end=end, myfill=_Fill(seg.myfill.below, seg.myfill.above)
        )

    def __eventCompare(self, p1IsStart, p11, p12, p2IsStart, p21, p22):
        comp = BooleanPoint.compare(p11, p21, self.tol)
        if comp != 0:
            return comp

        if p12.is_equivalent(p22, self.tol):
            return 0

        if p1IsStart != p2IsStart:
            return 1 if p1IsStart else -1

        return (
            1
            if BooleanPoint.point_above_or_on_line(
                p12, p21 if p2IsStart else p22, p22 if p2IsStart else p21, self.tol
            )
            else -1
        )

    def __eventAdd(self, ev, otherPt):
        def check_func(here):
            comp = self.__eventCompare(
                ev.isStart, ev.pt, otherPt, here.isStart, here.pt, here.other.pt
            )
            return comp < 0

        self.__eventRoot.insertBefore(ev, check_func)

    def __eventAddSegmentStart(self, segment, primary):
        evStart = _LinkedList.node(
            _Node(
                isStart=True,
                pt=segment.start,
                seg=segment,
                primary=primary,
            )
        )
        self.__eventAdd(evStart, segment.end)
        return evStart

    def __eventAddSegmentEnd(self, evStart, segment, primary):
        evEnd = _LinkedList.node(
            _Node(
                isStart=False,
                pt=segment.end,
                seg=segment,
                primary=primary,
                other=evStart,
            )
        )
        evStart.other = evEnd
        self.__eventAdd(evEnd, evStart.pt)

    def eventAddSegment(self, segment, primary):
        evStart = self.__eventAddSegmentStart(segment, primary)
        self.__eventAddSegmentEnd(evStart, segment, primary)
        return evStart

    def __eventUpdateEnd(self, ev, end):
        ev.other.remove()
        ev.seg.end = end
        ev.other.pt = end
        self.__eventAdd(ev.other, ev.pt)

    def __eventDivide(self, ev, pt):
        ns = self.segmentCopy(pt, ev.seg.end, ev.seg)
        self.__eventUpdateEnd(ev, pt)
        return self.eventAddSegment(ns, ev.primary)

    def __statusCompare(self, ev1, ev2):
        a1 = ev1.seg.start
        a2 = ev1.seg.end
        b1 = ev2.seg.start
        b2 = ev2.seg.end

        if BooleanPoint.collinear(a1, b1, b2, self.tol):
            if BooleanPoint.collinear(a2, b1, b2, self.tol):
                return 1
            return 1 if BooleanPoint.point_above_or_on_line(a2, b1, b2, self.tol) else -1
        return 1 if BooleanPoint.point_above_or_on_line(a1, b1, b2, self.tol) else -1

    def __statusFindSurrounding(self, statusRoot, ev):
        def check_func(here):
            return self.__statusCompare(ev, here.ev) > 0

        return statusRoot.findTransition(check_func)

    def __checkIntersection(self, ev1, ev2):
        seg1 = ev1.seg
        seg2 = ev2.seg
        a1 = seg1.start
        a2 = seg1.end
        b1 = seg2.start
        b2 = seg2.end

        i = BooleanPoint._lines_intersect(a1, a2, b1, b2, self.tol)
        if i is None:
            if not BooleanPoint.collinear(a1, a2, b1, self.tol):
                return None
            if a1.is_equivalent(b2, self.tol) or a2.is_equivalent(b1, self.tol):
                return None
            a1EquB1 = a1.is_equivalent(b1, self.tol)
            a2EquB2 = a2.is_equivalent(b2, self.tol)
            if a1EquB1 and a2EquB2:
                return ev2

            a1Between = not a1EquB1 and BooleanPoint.between(a1, b1, b2, self.tol)
            a2Between = not a2EquB2 and BooleanPoint.between(a2, b1, b2, self.tol)

            if a1EquB1:
                if a2Between:
                    self.__eventDivide(ev2, a2)
                else:
                    self.__eventDivide(ev1, b2)

                return ev2
            elif a1Between:
                if not a2EquB2:
                    if a2Between:
                        self.__eventDivide(ev2, a2)
                    else:
                        self.__eventDivide(ev1, b2)
                self.__eventDivide(ev2, a1)
        else:
            if i.alongA == 0:
                if i.alongB == -1:
                    self.__eventDivide(ev1, b1)
                elif i.alongB == 0:
                    self.__eventDivide(ev1, i.pt)
                elif i.alongB == 1:
                    self.__eventDivide(ev1, b2)
            if i.alongB == 0:
                if i.alongA == -1:
                    self.__eventDivide(ev2, a1)
                elif i.alongA == 0:
                    self.__eventDivide(ev2, i.pt)
                elif i.alongA == 1:
                    self.__eventDivide(ev2, a2)
        return None

    def __checkBothIntersections(self, above, ev, below):
        if above is not None:
            eve = self.__checkIntersection(ev, above)
            if eve is not None:
                return eve
        if below is not None:
            return self.__checkIntersection(ev, below)

        return None

    def calculate(self, primaryPolyInverted, secondaryPolyInverted):
        statusRoot = _LinkedList()
        segments = []

        cnt = 0

        while not self.__eventRoot.isEmpty():
            cnt += 1
            ev = self.__eventRoot.getHead()
            if ev.isStart:
                surrounding = self.__statusFindSurrounding(statusRoot, ev)
                above = (
                    surrounding.before.ev if surrounding.before is not None else None
                )
                below = surrounding.after.ev if surrounding.after is not None else None

                eve = self.__checkBothIntersections(above, ev, below)
                if eve is not None:
                    if self.selfIntersection:
                        toggle = False
                        if ev.seg.myfill.below is None:
                            toggle = True
                        else:
                            toggle = ev.seg.myfill.above != ev.seg.myfill.below

                        if toggle:
                            eve.seg.myfill.above = not eve.seg.myfill.above
                    else:
                        eve.seg.otherfill = ev.seg.myfill
                    ev.other.remove()
                    ev.remove()

                if self.__eventRoot.getHead() is not ev:
                    continue

                if self.selfIntersection:
                    toggle = False
                    if ev.seg.myfill.below is None:
                        toggle = True
                    else:
                        toggle = ev.seg.myfill.above != ev.seg.myfill.below

                    if below is None:
                        ev.seg.myfill.below = primaryPolyInverted
                    else:
                        ev.seg.myfill.below = below.seg.myfill.above

                    if toggle:
                        ev.seg.myfill.above = not ev.seg.myfill.below
                    else:
                        ev.seg.myfill.above = ev.seg.myfill.below
                else:
                    if ev.seg.otherfill is None:
                        inside = False
                        if below is None:
                            inside = (
                                secondaryPolyInverted
                                if ev.primary
                                else primaryPolyInverted
                            )
                        else:
                            if ev.primary == below.primary:
                                inside = below.seg.otherfill.above
                            else:
                                inside = below.seg.myfill.above
                        ev.seg.otherfill = _Fill(inside, inside)
                ev.other.status = surrounding.insert(_LinkedList.node(_Node(ev=ev)))
            else:
                st = ev.status
                if st is None:
                    raise Exception(
                        'PolyBool: Zero-length segment detected; '
                        'your tolerance is probably too small or too large'
                    )
                if statusRoot.exists(st.previous) and statusRoot.exists(st.next):
                    self.__checkIntersection(st.previous.ev, st.next.ev)
                st.remove()

                if not ev.primary:
                    s = ev.seg.myfill
                    ev.seg.myfill = ev.seg.otherfill
                    ev.seg.otherfill = s
                segments.append(ev.seg)
            self.__eventRoot.getHead().remove()
        return segments


class _RegionIntersecter(_Intersecter):
    def __init__(self, tol):
        _Intersecter.__init__(self, True, tol)

    def addRegion(self, region):
        pt2 = region[-1]
        for i in range(len(region)):
            pt1 = pt2
            pt2 = region[i]
            forward = BooleanPoint.compare(pt1, pt2, self.tol)

            if forward == 0:
                continue

            seg = self.newsegment(
                pt1 if forward < 0 else pt2, pt2 if forward < 0 else pt1
            )

            self.eventAddSegment(seg, True)

    def calculate(self, inverted):
        return _Intersecter.calculate(self, inverted, False)


class _SegmentIntersecter(_Intersecter):
    def __init__(self, tol):
        _Intersecter.__init__(self, False, tol)

    def calculate(
        self, segments1, is_inverted1, segments2, is_inverted2
    ):
        for seg in segments1:
            self.eventAddSegment(self.segmentCopy(seg.start, seg.end, seg), True)

        for seg in segments2:
            self.eventAddSegment(self.segmentCopy(seg.start, seg.end, seg), False)

        return _Intersecter.calculate(self, is_inverted1, is_inverted2)


class _SegmentChainerMatcher:
    def __init__(self):
        self.firstMatch = _Matcher(0, False, False)
        self.secondMatch = _Matcher(0, False, False)

        self.nextMatch = self.firstMatch

    def setMatch(self, index, matchesHead, matchesPt1):
        self.nextMatch.index = index
        self.nextMatch.matchesHead = matchesHead
        self.nextMatch.matchesPt1 = matchesPt1
        if self.nextMatch is self.firstMatch:
            self.nextMatch = self.secondMatch
            return False
        self.nextMatch = None
        return True


def _list_shift(list):
    list.pop(0)


def _list_pop(list):
    list.pop()


def _list_splice(list, index, count):
    del list[index: index + count]


def _list_unshift(list, element):
    list.insert(0, element)


def _segmentChainer(segments, tol):
    regions = []
    chains = []

    for seg in segments:
        pt1 = seg.start
        pt2 = seg.end
        if pt1.is_equivalent(pt2, tol):
            continue

        scm = _SegmentChainerMatcher()

        for i in range(len(chains)):
            chain = chains[i]
            head = chain[0]
            tail = chain[-1]

            if head.is_equivalent(pt1, tol):
                if scm.setMatch(i, True, True):
                    break
            elif head.is_equivalent(pt2, tol):
                if scm.setMatch(i, True, False):
                    break
            elif tail.is_equivalent(pt1, tol):
                if scm.setMatch(i, False, True):
                    break
            elif tail.is_equivalent(pt2, tol):
                if scm.setMatch(i, False, False):
                    break

        if scm.nextMatch is scm.firstMatch:
            chains.append([pt1, pt2])
            continue

        if scm.nextMatch is scm.secondMatch:
            index = scm.firstMatch.index
            pt = pt2 if scm.firstMatch.matchesPt1 else pt1
            addToHead = scm.firstMatch.matchesHead

            chain = chains[index]
            grow = chain[0] if addToHead else chain[-1]
            grow2 = chain[1] if addToHead else chain[-2]
            oppo = chain[-1] if addToHead else chain[0]
            oppo2 = chain[-2] if addToHead else chain[1]

            if BooleanPoint.collinear(grow2, grow, pt, tol):
                if addToHead:
                    _list_shift(chain)
                else:
                    _list_pop(chain)
                grow = grow2
            if oppo.is_equivalent(pt, tol):
                _list_splice(chains, index, 1)
                if BooleanPoint.collinear(oppo2, oppo, grow, tol):
                    if addToHead:
                        _list_pop(chain)
                    else:
                        _list_shift(chain)
                regions.append(chain)
                continue
            if addToHead:
                _list_unshift(chain, pt)
            else:
                chain.append(pt)
            continue

        def reverseChain(index):
            chains[index].reverse()

        def appendChain(index1, index2):
            chain1 = chains[index1]
            chain2 = chains[index2]
            tail = chain1[-1]
            tail2 = chain1[-2]
            head = chain2[0]
            head2 = chain2[1]

            if BooleanPoint.collinear(tail2, tail, head, tol):
                _list_pop(chain1)
                tail = tail2
            if BooleanPoint.collinear(tail, head, head2, tol):
                _list_shift(chain2)

            chains[index1] = chain1 + chain2
            _list_splice(chains, index2, 1)

        f = scm.firstMatch.index
        s = scm.secondMatch.index

        reverseF = len(chains[f]) < len(chains[s])
        if scm.firstMatch.matchesHead:
            if scm.secondMatch.matchesHead:
                if reverseF:
                    reverseChain(f)
                    appendChain(f, s)
                else:
                    reverseChain(s)
                    appendChain(s, f)
            else:
                appendChain(s, f)
        else:
            if scm.secondMatch.matchesHead:
                appendChain(f, s)
            else:
                if reverseF:
                    reverseChain(f)
                    appendChain(s, f)
                else:
                    reverseChain(s)
                    appendChain(f, s)

    return regions


def __select(segments, selection):
    result = []
    for seg in segments:
        index = (
            (8 if seg.myfill.above else 0)
            + (4 if seg.myfill.below else 0)
            + (2 if seg.otherfill is not None and seg.otherfill.above else 0)
            + (1 if seg.otherfill is not None and seg.otherfill.below else 0)
        )

        if selection[index] != 0:
            result.append(
                _Segment(
                    start=seg.start,
                    end=seg.end,
                    myfill=_Fill(selection[index] == 2, above=selection[index] == 1),
                )
            )
    return result


"""____________CORE INTERFACE FOR MANAGING INTERSECTIONS____________"""


def _segments(poly, tol):
    """Get the intersected PolySegments of a BooleanPolygon.

    Args:
        poly: A BooleanPolygon for which PolySegments will be computed.
        tol: The intersection tolerance.
    """
    i = _RegionIntersecter(tol)
    for region in poly.regions:
        i.addRegion(region)
    return _PolySegments(i.calculate(poly.is_inverted), poly.is_inverted)


def _combine(segments1, segments2, tol):
    """Combine intersected PolySegments into a CombinedPolySegments object.

    Args:
        segments1: The first PolySegments object to be combined.
        segments2: The second PolySegments to be combined.
        tol: The intersection tolerance.
    """
    i = _SegmentIntersecter(tol)
    return _CombinedPolySegments(
        i.calculate(
            segments1.segments,
            segments1.is_inverted,
            segments2.segments,
            segments2.is_inverted,
        ),
        segments1.is_inverted,
        segments2.is_inverted,
    )


def _select_union(polyseg):
    """Select the union from the PolySegments.

    above1 below1 above2 below2    Keep?               Value
      0      0      0      0   =>   no                  0
      0      0      0      1   =>   yes filled below    2
      0      0      1      0   =>   yes filled above    1
      0      0      1      1   =>   no                  0
      0      1      0      0   =>   yes filled below    2
      0      1      0      1   =>   yes filled below    2
      0      1      1      0   =>   no                  0
      0      1      1      1   =>   no                  0
      1      0      0      0   =>   yes filled above    1
      1      0      0      1   =>   no                  0
      1      0      1      0   =>   yes filled above    1
      1      0      1      1   =>   no                  0
      1      1      0      0   =>   no                  0
      1      1      0      1   =>   no                  0
      1      1      1      0   =>   no                  0
      1      1      1      1   =>   no                  0
    """
    return _PolySegments(
        segments=__select(
            # fmt:off
            polyseg.combined, [
                0, 2, 1, 0,
                2, 2, 0, 0,
                1, 0, 1, 0,
                0, 0, 0, 0,
            ]
            # fmt:on
        ),
        is_inverted=(polyseg.is_inverted1 or polyseg.is_inverted2),
    )


def _select_intersect(polyseg):
    """Select the intersection from the PolySegments.

    above1 below1 above2 below2    Keep?               Value
      0      0      0      0   =>   no                  0
      0      0      0      1   =>   no                  0
      0      0      1      0   =>   no                  0
      0      0      1      1   =>   no                  0
      0      1      0      0   =>   no                  0
      0      1      0      1   =>   yes filled below    2
      0      1      1      0   =>   no                  0
      0      1      1      1   =>   yes filled below    2
      1      0      0      0   =>   no                  0
      1      0      0      1   =>   no                  0
      1      0      1      0   =>   yes filled above    1
      1      0      1      1   =>   yes filled above    1
      1      1      0      0   =>   no                  0
      1      1      0      1   =>   yes filled below    2
      1      1      1      0   =>   yes filled above    1
      1      1      1      1   =>   no                  0
    """
    return _PolySegments(
        segments=__select(
            # fmt:off
            polyseg.combined, [
                0, 0, 0, 0,
                0, 2, 0, 2,
                0, 0, 1, 1,
                0, 2, 1, 0
            ]
            # fmt:on
        ),
        is_inverted=(polyseg.is_inverted1 and polyseg.is_inverted2),
    )


def _select_difference(polyseg):
    """Select the difference from the PolySegments.

    above1 below1 above2 below2    Keep?               Value
      0      0      0      0   =>   no                  0
      0      0      0      1   =>   no                  0
      0      0      1      0   =>   no                  0
      0      0      1      1   =>   no                  0
      0      1      0      0   =>   yes filled below    2
      0      1      0      1   =>   no                  0
      0      1      1      0   =>   yes filled below    2
      0      1      1      1   =>   no                  0
      1      0      0      0   =>   yes filled above    1
      1      0      0      1   =>   yes filled above    1
      1      0      1      0   =>   no                  0
      1      0      1      1   =>   no                  0
      1      1      0      0   =>   no                  0
      1      1      0      1   =>   yes filled above    1
      1      1      1      0   =>   yes filled below    2
      1      1      1      1   =>   no                  0
    """
    return _PolySegments(
        segments=__select(
            # fmt:off
            polyseg.combined, [
                0, 0, 0, 0,
                2, 0, 2, 0,
                1, 1, 0, 0,
                0, 1, 2, 0
            ]
            # fmt:on
        ),
        is_inverted=(polyseg.is_inverted1 and not polyseg.is_inverted2),
    )


def _select_difference_rev(polyseg):
    """Select the reversed difference from the PolySegments.

    above1 below1 above2 below2    Keep?               Value
      0      0      0      0   =>   no                  0
      0      0      0      1   =>   yes filled below    2
      0      0      1      0   =>   yes filled above    1
      0      0      1      1   =>   no                  0
      0      1      0      0   =>   no                  0
      0      1      0      1   =>   no                  0
      0      1      1      0   =>   yes filled above    1
      0      1      1      1   =>   yes filled above    1
      1      0      0      0   =>   no                  0
      1      0      0      1   =>   yes filled below    2
      1      0      1      0   =>   no                  0
      1      0      1      1   =>   yes filled below    2
      1      1      0      0   =>   no                  0
      1      1      0      1   =>   no                  0
      1      1      1      0   =>   no                  0
      1      1      1      1   =>   no                  0
    """
    return _PolySegments(
        segments=__select(
            # fmt:off
            polyseg.combined, [
                0, 2, 1, 0,
                0, 0, 1, 1,
                0, 2, 0, 2,
                0, 0, 0, 0
            ]
            # fmt:on
        ),
        is_inverted=(not polyseg.is_inverted1 and polyseg.is_inverted2),
    )


def _select_xor(polyseg):
    """Select the exclusive disjunction from the PolySegments.

    above1 below1 above2 below2    Keep?               Value
      0      0      0      0   =>   no                  0
      0      0      0      1   =>   yes filled below    2
      0      0      1      0   =>   yes filled above    1
      0      0      1      1   =>   no                  0
      0      1      0      0   =>   yes filled below    2
      0      1      0      1   =>   no                  0
      0      1      1      0   =>   no                  0
      0      1      1      1   =>   yes filled above    1
      1      0      0      0   =>   yes filled above    1
      1      0      0      1   =>   no                  0
      1      0      1      0   =>   no                  0
      1      0      1      1   =>   yes filled below    2
      1      1      0      0   =>   no                  0
      1      1      0      1   =>   yes filled above    1
      1      1      1      0   =>   yes filled below    2
      1      1      1      1   =>   no                  0
    """
    return _PolySegments(
        segments=__select(
            # fmt:off
            polyseg.combined, [
                0, 2, 1, 0,
                2, 0, 0, 1,
                1, 0, 0, 2,
                0, 1, 2, 0
            ]
            # fmt:on
        ),
        is_inverted=(polyseg.is_inverted1 != polyseg.is_inverted2),
    )


def _polygon(segments, tol):
    return BooleanPolygon(_segmentChainer(segments.segments, tol), segments.is_inverted)


def __operate(poly1, poly2, selector, tol):
    firstPolygonRegions = _segments(poly1, tol)
    secondPolygonRegions = _segments(poly2, tol)
    combinedSegments = _combine(firstPolygonRegions, secondPolygonRegions, tol)
    seg = selector(combinedSegments)
    return _polygon(seg, tol)


"""____________PUBLIC FUNCTIONS FOR BOOLEAN OPERATIONS____________"""


def union_all(polygons, tolerance):
    """Get a BooleanPolygon for the union of multiple polygons.

    Using this method is more computationally efficient than calling the union()
    method multiple times as this method will only compute the intersection of
    the segments once.

    Args:
        polygons: An array of BooleanPolygons for which the union will be computed.
        tolerance: The minimum distance between points before they are
            considered distinct from one another.

    Returns:
        A BooleanPolygon for the union across all of the input polygons.
    """
    seg1 = _segments(polygons[0], tolerance)
    for i in range(1, len(polygons)):
        seg2 = _segments(polygons[i], tolerance)
        comb = _combine(seg1, seg2, tolerance)
        seg1 = _select_union(comb)
    return _polygon(seg1, tolerance)


def intersect_all(polygons, tolerance):
    """Get a BooleanPolygon for the intersection of multiple polygons.

    Using this method is more computationally efficient than calling the intersect()
    method multiple times as this method will only compute the intersection of
    the segments once.

    Args:
        polygons: An array of BooleanPolygons for which the intersection will
            be computed.
        tolerance: The minimum distance between points before they are
            considered distinct from one another.

    Returns:
        A BooleanPolygon for the intersection across all of the input polygons.
    """
    seg1 = _segments(polygons[0], tolerance)
    for i in range(1, len(polygons)):
        seg2 = _segments(polygons[i], tolerance)
        comb = _combine(seg1, seg2, tolerance)
        seg1 = _select_intersect(comb)
    return _polygon(seg1, tolerance)


def split(poly1, poly2, tolerance):
    """Split two BooleanPolygons with one another to get the intersection and difference.

    Using this method is more computationally efficient than calling the intersect()
    and difference() methods individually as this method will only compute the
    intersection of the segments once.

    Args:
        poly1: A BooleanPolygon for the first polygon that will be split with
            the second polygon.
        poly2: A BooleanPolygon for the second polygon that will be split with
            the first polygon.
        tolerance: The minimum distance between points before they are
            considered distinct from one another.

    Returns:
        A tuple with three elements

        -   intersection: A BooleanPolygon for the intersection of the two
            input polygons.

        -   poly1_difference: A BooleanPolygon for the portion of poly1 that does
            not overlap with poly2. When combined with the intersection, this
            makes a split version of poly1.

        -   poly2_difference: A BooleanPolygon for the portion of poly2 that does
            not overlap with poly1. When combined with the intersection, this
            makes a split version of poly2.
    """
    first_regions = _segments(poly1, tolerance)
    second_regions = _segments(poly2, tolerance)
    comb = _combine(first_regions, second_regions, tolerance)
    intersection = _polygon(_select_intersect(comb), tolerance)
    poly1_difference = _polygon(_select_difference(comb), tolerance)
    poly2_difference = _polygon(_select_difference_rev(comb), tolerance)
    return intersection, poly1_difference, poly2_difference


def union(poly1, poly2, tolerance):
    """Get a BooleanPolygon for the union of two polygons.

    Note that the result will not differentiate hole polygons from boundary polygons.

    Args:
        poly1: A BooleanPolygon for the first polygon for which the union will
            be computed.
        poly2: A BooleanPolygon for the second polygon for which the union will
            be computed.
        tolerance: The minimum distance between points before they are
            considered distinct from one another.

    Returns:
        A BooleanPolygon for the union of the two polygons.
    """
    return __operate(poly1, poly2, _select_union, tolerance)


def intersect(poly1, poly2, tolerance):
    """Get a BooleanPolygon for the intersection of two polygons.

    Args:
        poly1: A BooleanPolygon for the first polygon for which the intersection
            will be computed.
        poly2: A BooleanPolygon for the second polygon for which the intersection
            will be computed.
        tolerance: The minimum distance between points before they are
            considered distinct from one another.

    Returns:
        A BooleanPolygon for the intersection of the two polygons.
    """
    return __operate(poly1, poly2, _select_intersect, tolerance)


def difference(poly1, poly2, tolerance):
    """Get a BooleanPolygon for the subtraction of poly2 from poly1.

    Args:
        poly1: A BooleanPolygon for the the polygon that will be subtracted from.
        poly2: A BooleanPolygon for the polygon to subtract with.
        tolerance: The minimum distance between points before they are
            considered distinct from one another.

    Returns:
        A BooleanPolygon for the difference of poly1 - poly2.
    """
    return __operate(poly1, poly2, _select_difference, tolerance)


def difference_reversed(poly1, poly2, tolerance):
    """Get a BooleanPolygon for the subtraction of poly1 from poly2.

    Args:
        poly1: A BooleanPolygon for the polygon to subtract with.
        poly2: A BooleanPolygon for the the polygon that will be subtracted from.
        tolerance: The minimum distance between points before they are
            considered distinct from one another.

    Returns:
        A BooleanPolygon for the difference of poly2 - poly1.
    """
    return __operate(poly1, poly2, _select_difference_rev, tolerance)


def xor(poly1, poly2, tolerance):
    """Get a BooleanPolygon for the exclusive disjunction of two Polygons.

    Note that this method is prone to merging holes that may exist in the
    result into the boundary to create a single list of joined vertices,
    which may not always be desirable. In this case, it may be desirable
    to do two separate difference calculations instead or use the split method.

    Also note that, when the result includes separate polygons for holes,
    it will not differentiate hole polygons from boundary polygons.

    Args:
        poly1: A BooleanPolygon for the first polygon for which the exclusive
            disjunction will be computed.
        poly2: A BooleanPolygon for the second polygon for which the exclusive
            disjunction will be computed.
        tolerance: The minimum distance between points before they are
            considered distinct from one another.

    Returns:
        A BooleanPolygon for the exclusive disjunction of the two polygons.
    """
    return __operate(poly1, poly2, _select_xor, tolerance)
