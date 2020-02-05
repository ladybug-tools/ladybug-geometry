# coding=utf-8
"""
Implementation of the straight skeleton algorithm as described by Felkel and Obdržálek in their 1998 conference paper Straight skeleton implementation.
"""

from __future__ import division

import logging
import heapq
from itertools import *
from collections import namedtuple
import operator

# Geometry classes
from ladybug_geometry.geometry2d.polygon import Polygon2D
from ladybug_geometry.geometry2d.pointvector import Point2D, Vector2D
from ladybug_geometry.geometry2d.line import LineSegment2D
from ladybug_geometry.geometry2d.ray import Ray2D
from ladybug_geometry import intersection2d

log = logging.getLogger("__name__")


# For Debugging

class Debug:
	def __init__(self, image):
		if image is not None:
			self.im = image[0]
			self.draw = image[1]
			self.do = True
		else:
			self.do = False

	def line(self, *args, **kwargs):
		if self.do:
			self.draw.line(*args, **kwargs)

	def rectangle(self, *args, **kwargs):
		if self.do:
			self.draw.rectangle(*args, **kwargs)

	def show(self):
		if self.do:
			self.im.show()

_debug = Debug(None)

def set_debug(image):
	global _debug
	_debug = Debug(image)


# Skeleton Code

def _window(lst):
	prevs, items, nexts = tee(lst, 3)
	prevs = islice(cycle(prevs), len(lst)-1, None)
	nexts = islice(cycle(nexts), 1, None)
	return zip(prevs, items, nexts)

def _determinant(a, b):
	res = a.x*b.y - b.x*a.y
	return res

def _approximately_equals(a, b):
	return a==b or (abs(a-b) <= max(abs(a), abs(b)) * 0.001)

def _approximately_same(point_a, point_b):
	return _approximately_equals(point_a.x, point_b.x) and _approximately_equals(point_a.y, point_b.y)

def _normalize_contour(contour):
	"""
	Consumes list of x,y coordinate tuples and returns list of Point2Ds.

	Args:
		contour: list of x,y tuples from contour.
	Return:
		 list of Point2Ds of contour.
	"""
	contour = [Point2D(float(x), float(y)) for (x,y) in contour]
	normed_contour = []
	for prev, point, next in _window(contour):
		normed_prev = Point2D.normalize(point-prev)
		normed_next = Point2D.normalize(next-point)
		if not (point == next or normed_prev == normed_next):
			normed_contour.append(point)

	return normed_contour


class _SplitEvent(namedtuple("_SplitEvent", "distance, intersection_point, vertex, opposite_edge")):
	__slots__ = ()
	def __str__(self):
		return "{} Split event @ {} from {} to {}".format(self.distance, self.intersection_point, self.vertex, self.opposite_edge)

class _EdgeEvent(namedtuple("_EdgeEvent", "distance intersection_point vertex_a vertex_b")):
	__slots__ = ()
	def __str__(self):
		return "{} Edge event @ {} between {} and {}".format(self.distance, self.intersection_point, self.vertex_a, self.vertex_b)

_OriginalEdge = namedtuple("_OriginalEdge", "edge bisector_left, bisector_right")

Subtree = namedtuple("Subtree", "source, height, sinks")

def _side(point, line):
	a = line.p.x
	b = line.p.y

class _LAVertex:
	def __init__(self, point, edge_left, edge_right, direction_vectors=None):
		self.point = point
		self.edge_left = edge_left
		self.edge_right = edge_right
		self.prev = None
		self.next = None
		self.lav = None
		self._valid = True; # this should be handled better. Maybe membership in lav implies validity?
		edge_left._v = Point2D.normalize(edge_left.v)
		edge_right._v = Point2D.normalize(edge_right.v)
		creator_vectors = (edge_left.v * -1, edge_right.v)
		if direction_vectors is None:
			direction_vectors = creator_vectors

		# The determinant of two 2d vectors equals the sign of their cross product
		# If second vector is to left, then sign of det is pos, else neg
		# So if cw polygon, convex angle will be neg (since second vector to right)
		# In this case, since we flip first vector, concave will be neg
		self._is_reflex = _determinant(*direction_vectors) < 0.0
		self._bisector = Ray2D(
			self.point,
			operator.add(*creator_vectors) * (-1 if self.is_reflex else 1)
			)
		log.info("Created vertex %s", self.__repr__())
		_debug.line((self.bisector.p.x, self.bisector.p.y, self.bisector.p.x+self.bisector.v.x*100, self.bisector.p.y+self.bisector.v.y*100), fill="blue")

	@property
	def bisector(self):
		return self._bisector

	@property
	def is_reflex(self):
		return self._is_reflex

	@property
	def original_edges(self):
		return self.lav._slav._original_edges

	def next_event(self):
		events = []
		if self.is_reflex:
			# a reflex vertex may generate a split event
			# split events happen when a vertex hits an opposite edge, splitting the polygon in two.
			log.debug("looking for split candidates for vertex %s", self)
			for edge in self.original_edges:
				if edge.edge == self.edge_left or edge.edge == self.edge_right:
					continue

				log.debug("\tconsidering EDGE %s", edge)

				# A potential b is at the intersection of between our own bisector and
				# the bisector of the angle between the tested edge and any one of our
				# own edges.

				# We choose the "less parallel" edge (in order to exclude a
				# potentially parallel edge)

				# Make normalized copies of vectors
				norm_edge_left_v  = Vector2D.normalize(self.edge_left.v)
				norm_edge_right_v = Vector2D.normalize(self.edge_right.v)
				norm_edge_v = Vector2D.normalize(edge.edge.v)

				# Compute dot
				leftdot  = abs(norm_edge_left_v.dot(norm_edge_v))
				rightdot = abs(norm_edge_right_v.dot(norm_edge_v))
				selfedge = self.edge_left if leftdot < rightdot else self.edge_right
				otheredge = self.edge_left if leftdot > rightdot else self.edge_right

				# Make copies of edges and compute intersection
				self_edge_copy = LineSegment2D(selfedge.p, selfedge.v)
				edge_edge_copy = LineSegment2D(edge.edge.p, edge.edge.v)
				i = intersection2d.intersect_line2d(self_edge_copy, edge_edge_copy)
				if i is not None and not _approximately_equals(i, self.point):
					#locate candidate b
					linvec = Point2D.normalize(self.point - i)
					edvec  = Point2D.normalize(edge.edge.v)
					if linvec.dot(edvec)<0:
						edvec = -edvec

					bisecvec = edvec + linvec
					if abs(bisecvec) == 0:
						continue
					bisector = LineSegment2D(i, bisecvec)
					b = bisector.intersect(self.bisector)

					if b is None:
						continue

					#check eligibility of b
					# a valid b should lie within the area limited by the edge and the bisectors of its two vertices:
					xleft = _determinant(edge.bisector_left.v.normalized(), (b - edge.bisector_left.p).normalized()) > 0
					xright = _determinant(edge.bisector_right.v.normalized(), (b - edge.bisector_right.p).normalized()) < 0
					xedge = _determinant(edge.edge.v.normalized(), (b - edge.edge.p).normalized()) < 0

					if not (xleft and xright and xedge):
						log.debug("\t\tDiscarded candidate %s (%s-%s-%s)", b, xleft, xright, xedge)
						continue

					log.debug("\t\tFound valid candidate %s", b)
					events.append( _SplitEvent(Line2(edge.edge).distance(b), b, self, edge.edge) )

		# Intersect line2d with line2d (does not assume lines are infinite)
		i_prev = intersection2d.intersect_line2d(self.bisector, self.prev.bisector)
		i_next = intersection2d.intersect_line2d(self.bisector, self.prev.bisector)

		# Calc distances from edge to intersection
		dist_to_i_prev = LineSegment2D(
			self.edge_left.p.duplicate(),
			self.edge_left.v.duplicate())
		dist_to_i_prev = dist_to_i_prev.distance_to_point(i_prev)
		dist_to_i_next = LineSegment2D(
			self.edge_right.p.duplicate(),
			self.edge_right.v.duplicate())
		dist_to_i_next = dist_to_i_next.distance_to_point(i_next)

		# Make EdgeEvent and append to events
		if i_prev is not None:
			events.append(_EdgeEvent(dist_to_i_prev, i_prev, self.prev, self))
		if i_next is not None:
			events.append(_EdgeEvent(dist_to_i_next, i_next, self, self.next))

		if not events:
			return None

		ev = min(
			events,
			key=lambda event: self.point.distance_to_point(event.intersection_point)
			)

		log.info("Generated new event for %s: %s", self, ev)
		return ev

	def invalidate(self):
		if self.lav is not None:
			self.lav.invalidate(self)
		else:
			self._valid = False

	@property
	def is_valid(self):
		return self._valid

	def __str__(self):
		return "Vertex ({:.2f};{:.2f})".format(self.point.x, self.point.y)

	def __lt__(self, other):
		if isinstance(other, _LAVertex):
			return self.point.x < other.point.x

	def __repr__(self):
		return "Vertex ({}) ({:.2f};{:.2f}), bisector {}, edges {} {}".format("reflex" if self.is_reflex else "convex", self.point.x, self.point.y, self.bisector, self.edge_left, self.edge_right)

class _SLAV:
	def __init__(self, polygon, holes):
		contours = [_normalize_contour(polygon)]
		contours.extend([_normalize_contour(hole) for hole in holes])

		self._lavs = [ _LAV.from_polygon(contour, self) for contour in contours ]

		#store original polygon edges for calculating split events
		self._original_edges = [
			_OriginalEdge(
				LineSegment2D.from_end_points(vertex.prev.point, vertex.point),
				vertex.prev.bisector,
				vertex.bisector
				) for vertex in chain.from_iterable(self._lavs)
			]

	def __iter__(self):
		for lav in self._lavs:
			yield lav

	def __len__(self):
		return len(self._lavs)

	def empty(self):
		return len(self._lavs) == 0

	def handle_edge_event(self, event):
		sinks = []
		events = []

		lav = event.vertex_a.lav
		if event.vertex_a.prev == event.vertex_b.next:
			log.info("%.2f Peak event at intersection %s from <%s,%s,%s> in %s", event.distance, event.intersection_point, event.vertex_a, event.vertex_b, event.vertex_a.prev, lav)
			self._lavs.remove(lav)
			for vertex in list(lav):
				sinks.append(vertex.point)
				vertex.invalidate()
		else:
			log.info("%.2f Edge event at intersection %s from <%s,%s> in %s", event.distance, event.intersection_point, event.vertex_a, event.vertex_b,lav)
			new_vertex = lav.unify(event.vertex_a, event.vertex_b, event.intersection_point)
			if lav.head in (event.vertex_a, event.vertex_b):
				lav.head = new_vertex
			sinks.extend((event.vertex_a.point, event.vertex_b.point))
			next_event = new_vertex.next_event()
			if next_event is not None:
				events.append(next_event)

		return (Subtree(event.intersection_point, event.distance, sinks), events)

	def handle_split_event(self, event):
		lav = event.vertex.lav
		log.info("%.2f Split event at intersection %s from vertex %s, for edge %s in %s", event.distance, event.intersection_point, event.vertex, event.opposite_edge, lav)

		sinks = [event.vertex.point]
		vertices = []
		x = None   # right vertex
		y = None   # left vertex
		norm = event.opposite_edge.v.normalized()
		for v in chain.from_iterable(self._lavs):
			log.debug("%s in %s", v, v.lav)
			if norm == v.edge_left.v.normalized() and event.opposite_edge.p == v.edge_left.p:
				x = v
				y = x.prev
			elif norm == v.edge_right.v.normalized() and event.opposite_edge.p == v.edge_right.p:
				y=v
				x=y.next

			if x:
				xleft = _determinant(y.bisector.v.normalized(), (event.intersection_point - y.point).normalized()) >= 0
				xright = _determinant(x.bisector.v.normalized(), (event.intersection_point - x.point).normalized()) <= 0
				log.debug("Vertex %s holds edge as %s edge (%s, %s)", v, ("left" if x==v else "right"), xleft, xright)

				if xleft and xright:
					break
				else:
					x = None
					y = None

		if x is None:
			log.info("Failed split event %s (equivalent edge event is expected to follow)", event)
			return (None,[])

		v1 = _LAVertex(event.intersection_point, event.vertex.edge_left, event.opposite_edge)
		v2 = _LAVertex(event.intersection_point, event.opposite_edge, event.vertex.edge_right)

		v1.prev = event.vertex.prev
		v1.next = x
		event.vertex.prev.next = v1
		x.prev = v1

		v2.prev = y
		v2.next = event.vertex.next
		event.vertex.next.prev = v2
		y.next = v2

		new_lavs = None
		self._lavs.remove(lav)
		if lav != x.lav:
			# the split event actually merges two lavs
			self._lavs.remove(x.lav)
			new_lavs = [_LAV.from_chain(v1, self)]
		else:
			new_lavs = [_LAV.from_chain(v1, self), _LAV.from_chain(v2, self)]

		for l in new_lavs:
			log.debug(l)
			if len(l) > 2:
				self._lavs.append(l)
				vertices.append(l.head)
			else:
				log.info("LAV %s has collapsed into the line %s--%s", l, l.head.point, l.head.next.point)
				sinks.append(l.head.next.point)
				for v in list(l):
					v.invalidate()

		events = []
		for vertex in vertices:
			next_event = vertex.next_event()
			if next_event is not None:
				events.append(next_event)

		event.vertex.invalidate()
		return (Subtree(event.intersection_point, event.distance, sinks), events)

class _LAV:
	def __init__(self, slav):
		self.head = None
		self._slav = slav
		self._len = 0
		log.debug("Created LAV %s", self)


	@classmethod
	def from_polygon(cls, polygon, slav):
		lav = cls(slav)
		for prev, point, next in _window(polygon):
			lav._len += 1
			vertex = _LAVertex(
				point,
				LineSegment2D.from_end_points(prev, point),
				LineSegment2D.from_end_points(point, next)
				)

			vertex.lav = lav
			if lav.head == None:
				lav.head = vertex
				vertex.prev = vertex.next = vertex
			else:
				vertex.next = lav.head
				vertex.prev = lav.head.prev
				vertex.prev.next = vertex
				lav.head.prev = vertex
		return lav

	@classmethod
	def from_chain(cls, head, slav):
		lav = cls(slav)
		lav.head = head
		for vertex in lav:
			lav._len += 1
			vertex.lav = lav
		return lav

	def invalidate(self, vertex):
		assert vertex.lav is self, "Tried to invalidate a vertex that's not mine"
		log.debug("Invalidating %s", vertex)
		vertex._valid = False
		if self.head == vertex:
			self.head = self.head.next
		vertex.lav = None

	def unify(self, vertex_a, vertex_b, point):
		normed_b_bisector = Vector2D.normalize(vertex_b.bisector.v)
		normed_a_bisector = Vector2D.normalize(vertex_a.bisector.v)
		replacement =_LAVertex(
			point,
			vertex_a.edge_left,
			vertex_b.edge_right,
		 	(normed_b_bisector, normed_a_bisector)
			)
		replacement.lav = self

		if self.head in [vertex_a, vertex_b]:
			self.head = replacement

		vertex_a.prev.next = replacement
		vertex_b.next.prev = replacement
		replacement.prev = vertex_a.prev
		replacement.next = vertex_b.next

		vertex_a.invalidate()
		vertex_b.invalidate()

		self._len -= 1
		return replacement

	def __str__(self):
		return "LAV {}".format(id(self))

	def __repr__(self):
		return "{} = {}".format(str(self), [vertex for vertex in self])

	def __len__(self):
		return self._len

	def __iter__(self):
		cur = self.head
		while True:
			yield cur
			cur = cur.next
			if cur == self.head:
				raise StopIteration

	def _show(self):
		cur = self.head
		while True:
			print(cur.__repr__())
			cur = cur.next
			if cur == self.head:
				break

class _EventQueue:
	def __init__(self):
		self.__data = []

	def put(self, item):
		if item is not None:
			heapq.heappush(self.__data, item)

	def put_all(self, iterable):
		for item in iterable:
			heapq.heappush(self.__data, item)

	def get(self):
		return heapq.heappop(self.__data)

	def empty(self):
		return len(self.__data)==0

	def peek(self):
		return self.__data[0]

	def show(self):
		for item in self.__data:
			print(item)

def skeletonize(polygon, holes=None):
	"""
	Compute the straight skeleton of a polygon.

	The polygon should be given as a list of vertices in counter-clockwise order.
	Holes is a list of the contours of the holes, the vertices of which should be in clockwise order.

	Returns the straight skeleton as a list of "subtrees", which are in the form of (source, height, sinks),
	where source is the highest points, height is its height, and sinks are the point connected to the source.
	"""

	# Current code actually works on cw and ccw order for polygons and holes, respectively
	# Sol reverse vertice order for both inputs
	polygon = polygon[::-1]
	if holes is not None:
		holes = [hole[::-1] for hole in holes]

	slav = _SLAV(polygon, holes)
	output = []
	prioque = _EventQueue()

	for lav in slav:
		for vertex in lav:
			v = vertex.next_event()
			prioque.put(v)

	while not (prioque.empty() or slav.empty()):
		log.debug("SLAV is %s", [repr(lav) for lav in slav])
		i = prioque.get()
		if isinstance(i, _EdgeEvent):
			if not i.vertex_a.is_valid or not i.vertex_b.is_valid:
				log.info("%.2f Discarded outdated edge event %s", i.distance, i)
				continue

			(arc, events) = slav.handle_edge_event(i)
		elif isinstance(i, _SplitEvent):
			if not i.vertex.is_valid:
				log.info("%.2f Discarded outdated split event %s", i.distance, i)
				continue
			(arc, events) = slav.handle_split_event(i)

		prioque.put_all(events)

		if arc is not None:
			output.append(arc)
			for sink in arc.sinks:
				_debug.line((arc.source.x, arc.source.y, sink.x, sink.y), fill="red")

			_debug.show()
	return output