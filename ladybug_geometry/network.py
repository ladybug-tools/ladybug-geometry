# coding=utf-8
"""A 2D Directed Graph Network data structure used for splitting polygons.

A directed graph (or digraph) is a graph made up of a set of vertices (aka. nodes)
connected by directed edges, often into a network where each node can have
multiple connections.

This class is used in all operations where a polygon or face is split using
line segments or polylines.

The overall strategies used in this module are inspired by operations performed by
NetworkX but were constructed from scratch without working from any particular
module or class in the package More information on NetworkX can be found here:
https://github.com/networkx/networkx
"""
from __future__ import division
import math

from ladybug_geometry.intersection2d import intersect_line_segment2d
from ladybug_geometry.geometry2d import LineSegment2D


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


class Node(object):
    """A Node within in DirectedGraphNetwork, optionally connected to other Nodes.

    Args:
        pt: A Point2D object for the node
        key: String representation of the Point2D object which accounts for tolerance.
        order: Integer for the order of the Node (based on directed graph propagation).
        adj_lst: List of Node objects that are adjacent to this node.
        exterior: Optional boolean to indicate if the Node is on the exterior of
            the graph. If None, this value can be computed later based on the
            position within the overall graph.

    Properties:
        * pt
        * key
        * adj_lst
        * exterior
        * adj_count
    """
    __slots__ = ('key', 'pt', '_order', 'adj_lst', 'exterior')

    def __init__(self, pt, key, order, adj_lst, exterior=None):
        """Initialize Node."""
        self.pt = pt
        self.key = key
        self._order = order
        self.adj_lst = adj_lst
        self.exterior = exterior

    @property
    def adj_count(self):
        """Number of adjacent nodes"""
        return len(self.adj_lst)

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return isinstance(other, Node) and \
            self.key == other.key

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return '{}: {}'.format(self._order, self.key)


class DirectedGraphNetwork(object):
    """A 2D Directed Graph Network data structure used for splitting polygons.

    A directed graph (or digraph) is a graph made up of a set of vertices (aka. nodes)
    connected by directed edges, often into a network where each node can have
    multiple connections. This class contains for finding the shortest pathways
    through the graph. It also helps differentiate interior from exterior parts
    of the graph. Typically, interior pathways are bi-directional in the graph
    while exterior pathways are uni-directional.

    Args:
        tolerance: The tolerance used to determine point equivalence throughout
            the graph. This is used for hashing points within the network.

    Properties:
        * node_count: Integer for the number of nodes in graph.
        * nodes: An iterable of nodes in graph.
        * ordered_nodes: An iterable of nodes in graph in order they were added.
        * connection_segments: List of LineSegment2D for the node connections.
        * outer_root_node: A node for the outer root key.
        * hole_root_nodes: A list of nodes for the hole root keys.
    """
    __slots__ = ('_directed_graph', '_tolerance', 'outer_root_key', 'hole_root_keys')

    def __init__(self, tolerance):
        """Initialize a PolygonDirectedGraph."""
        self._directed_graph = {}  # will be used to hold the nodes of the network
        # multiply tolerance by 2 to catch both positive and negative point equivalence
        self._tolerance = tolerance * 2
        self.outer_root_key = None  # will be set during network creation
        self.hole_root_keys = []  # will be set during network creation

    @classmethod
    def from_point_array(cls, point_array, tolerance, loop=True):
        """Create a DirectedGraphNetwork for a 1-dimensional array of points.

        Args:
            point_array: Array of Point2D objects.
            tolerance: The tolerance used to determine point equivalence throughout
                the graph. This is used for hashing points within the network.
            loop: Optional boolean, which will ensure that the input point_array
                is connected into a loop. (Default: True).
        """
        ext = True if loop else None
        dg = cls(tolerance)
        for i in range(len(point_array) - 1):
            k = dg.add_node(point_array[i], [point_array[i + 1]], exterior=ext)
            if i == 0:
                dg.outer_root_key = k
        if loop:
            dg.add_node(point_array[-1], [point_array[0]], exterior=ext)
        return dg

    @classmethod
    def from_polygon(cls, polygon, tolerance):
        """Create a DirectedGraphNetwork from a polygon.

        Args:
            polygon: A Polygon2D object.
            tolerance: The tolerance used to determine point equivalence throughout
                the graph. This is used for hashing points within the network.

        Returns:
            A PolygonDirectedGraph object that represents the polygon. The edges
            are uni-directional and counterclockwise. The nodes have the exterior
            property set to True.
        """
        # ensure the boundary and holes are oriented correctly for the graph
        if polygon.is_clockwise:
            polygon = polygon.reverse()
        return cls.from_point_array(polygon.vertices, tolerance, loop=True)

    @classmethod
    def from_shape_with_holes(cls, boundary, holes, tolerance):
        """Create a DirectedGraphNetwork for a shape with holes.

        Args:
            boundary: A Polygon2D for the boundary around the shape.
            holes: An optional list of Polygon2D for the holes within the shape.
                If None, it will be assumed that no holes exist in the shape.
            tolerance: The tolerance used to determine point equivalence throughout
                the graph. This is used for hashing points within the network.

        Returns:
            A PolygonDirectedGraph object that represents the boundary with holes.
            The edges (including the boundary and holes) are uni-directional with
            the outer boundary being counterclockwise and the holes being clockwise.
            In other words, the fill of the shape is always to the left of each edge.
            The nodes have the exterior property set to True.
        """
        # ensure the boundary and holes are oriented correctly for the graph
        if boundary.is_clockwise:
            boundary = boundary.reverse()
        loops = [boundary.vertices]
        if holes is not None:
            for hole in holes:
                if hole.is_clockwise:
                    loops.append(hole.vertices)
                else:
                    loops.append(hole.reverse().vertices)

        # make the directed graph and add the nodes for the boundary + holes
        dg = cls(tolerance=tolerance)
        for loop_count, vertices in enumerate(loops):
            for j in range(len(vertices) - 1):
                curr_v = vertices[j]
                next_v = vertices[j + 1]
                k = dg.add_node(curr_v, [next_v], exterior=True)
                if j == 0:
                    if loop_count == 0:
                        dg.outer_root_key = k
                    else:
                        dg.hole_root_keys.append(k)
            dg.add_node(vertices[-1], [vertices[0]], exterior=True)  # close loop
        return dg

    @classmethod
    def from_shape_to_split(cls, boundary, holes, split_segments, tolerance):
        """Get a DirectedGraphNetwork for a shape to be split with segments.

        The shape is composed of a boundary with optional holes and the
        split_segments are an array of any line segments to be used to
        split the shape.

        Args:
            boundary: A Polygon2D for the boundary around the shape.
            holes: An optional list of Polygon2D for the holes within the shape.
                If None, it will be assumed that no holes exist in the shape.
            split_segments: An array of LineSegment2D to be used to split the shape.
            tolerance: The tolerance used to determine point equivalence throughout
                the graph. This is used for hashing points within the network.

        Returns:
            A PolygonDirectedGraph object that represents the network formed by
            the boundary, holes, and split segments. Portions of the split_segments
            that are outside of the boundary are excluded from the graph. All interior
            connections between the split_segments in the graph are bi-directional.
            The exterior edges (including the boundary and holes) are uni-directional
            with the outer boundary being counterclockwise and the holes being
            clockwise. In other words, the fill of the shape is always to the left
            of each exterior edge. The nodes at the boundary and the holes have
            the exterior property set to True.
        """
        # first split the boundary and holes with the split_segments
        boundary = boundary.remove_colinear_vertices(tolerance)
        bound_sgs = cls._intersect_segments(boundary.segments, split_segments, tolerance)
        bound_pts = [seg.p1 for seg in bound_sgs]
        split_boundary = boundary.__class__(bound_pts)
        split_holes = None
        if holes is not None:
            split_holes = []
            for hole in holes:
                hole = hole.remove_colinear_vertices(tolerance)
                hole_sgs = cls._intersect_segments(
                    hole.segments, split_segments, tolerance)
                hole_pts = [seg.p1 for seg in hole_sgs]
                split_holes.append(boundary.__class__(hole_pts))

        # make the directed graph for the boundary + holes
        dg = cls.from_shape_with_holes(split_boundary, split_holes, tolerance)

        # process the split_segments for intersection
        add_segs = list(boundary.segments)
        if holes is not None:
            for hole in holes:
                add_segs.extend(hole.segments)
        split_seg = cls._intersect_segments(split_segments, add_segs, tolerance)
        split_seg = cls._remove_segments_outside_boundary(split_seg, boundary, tolerance)
        if len(split_seg) == 0:  # none of the segments are inside the shape
            return dg

        # add the intersection segments to the graph
        for seg in split_seg:  # add a bidirectional edge to represent interior edges
            dg.add_node(seg.p2, [seg.p1], exterior=False)
            dg.add_node(seg.p1, [seg.p2], exterior=False)
        return dg

    @property
    def node_count(self):
        return len(self.nodes)

    @property
    def nodes(self):
        """Get an iterable of pt nodes"""
        return self._directed_graph.values()

    @property
    def ordered_nodes(self):
        """Get an iterable of pt nodes in order of addition"""
        nodes = list(self.nodes)
        nodes.sort(key=lambda v: v._order)
        return nodes

    @property
    def outer_root_node(self):
        """Get the node of the outer boundary root."""
        return self.node(self.outer_root_key)

    @property
    def hole_root_nodes(self):
        """Get a list of nodes for the roots of the holes."""
        return [self.node(hole_key) for hole_key in self.hole_root_keys]

    @property
    def connection_segments(self):
        """Get a list of LineSegment2D for the node connections in the graph."""
        traversed = set()
        connections = []
        for node in self.nodes:
            for conn_node in node.adj_lst:
                if (conn_node.key, node.key) not in traversed:
                    conn_seg = LineSegment2D.from_end_points(node.pt, conn_node.pt)
                    connections.append(conn_seg)
                    traversed.add((node.key, conn_node.key))
        return connections

    def node(self, key):
        """Retrieves the node based on passed value.

        Args:
            val: The key for a node in the directed graph.

        Returns:
            The node for the passed key.
        """
        try:
            return self._directed_graph[key]
        except KeyError:
            return None  # broken connection

    def add_adj(self, node, adj_val_lst):
        """Adds nodes to node.adj_lst.

        This method will ensure no repetitions will occur in adj_lst.

        Args:
            node: Node to add adjacencies to.
            adj_val_lst: List of Point2D objects to add as adjacent nodes.
        """
        adj_keys = {n.key: None for n in node.adj_lst}
        adj_keys[node.key] = None
        for adj_val in adj_val_lst:
            adj_key = coordinates_hash(adj_val, self._tolerance)
            if adj_key in adj_keys:
                continue

            self._add_node(adj_key, adj_val, exterior=None)
            adj_keys[adj_key] = None
            node.adj_lst.append(self.node(adj_key))

    def remove_adj(self, node, adj_key_lst):
        """Removes nodes in node.adj_lst.

        Args:
            node: Node to remove adjacencies to.
            adj_val_lst: List of adjacency keys to remove as adjacent nodes.
        """
        node.adj_lst = [n for n in node.adj_lst if n.key not in set(adj_key_lst)]

    def add_node(self, val, adj_lst, exterior=None):
        """Add a node into the PolygonDirectedGraph.

        This method consumes a Point2D, computes its key value, and adds it in the
        graph if it doesn't exist. If it does exist it appends adj_lst to existing pt.

        Args:
            val: A Point2D object.
            adj_lst: A list of Point2D objects adjacent to the node.
            exterior: Optional boolean for whether the Node is exterior.

        Returns:
            The hashed key from the existing or new node.
        """
        key = coordinates_hash(val, self._tolerance)  # get key
        self._add_node(key, val, exterior)  # get node if it exists
        node = self._directed_graph[key]
        self.add_adj(node, adj_lst)  # add the adj_lst to dg
        # if the exterior boolean was passed, change the node attribute
        if exterior is not None:
            node.exterior = exterior
        return node.key

    def _add_node(self, key, val, exterior=None):
        """Helper function for add_node.

        If key doesn't currently exist in the graph, it is added with a new key.
        """
        if key not in self._directed_graph:
            self._directed_graph[key] = Node(val, key, self.node_count, [], exterior)
        return self._directed_graph[key]

    def insert_node(self, base_node, new_val, next_node, exterior=None):
        """Insert node in the middle of an edge defined by node and next_node.

        Args:
            base_node: Node object to the left.
            new_val:  A Point2D object for the new node in the middle.
            next_node: Node object to the right.
            exterior: Optional boolean for exterior attribute.

        Returns:
            key of new_val node.
        """
        # add new_val as a node, with next_node as an adjacency
        new_key = self.add_node(new_val, [next_node.pt], exterior=exterior)
        # update parent by adding new adjacency, and removing old adjacency
        self.add_adj(base_node, [self.node(new_key).pt])

        # catch the edge case where the new point is coincident to parent or next_point.
        # this occurs when intersection passes through a corner.
        if (new_key == next_node.key) or (new_key == base_node.key):
            return new_key
        self.remove_adj(base_node, [next_node.key])
        return new_key

    def node_exists(self, key):
        """Check if a node is in the graph. True if node in directed graph else False."""
        return key in self._directed_graph

    def pt_exists(self, pt):
        """True if a point (as Point2D) in directed graph exists as node else False.
        """
        return self.node_exists(coordinates_hash(pt, self._tolerance))

    def polygon_exists(self, polygon):
        """Check if a polygon is in the directed graph.

        Args:
            polygons: A Polygon2D object.

        Return:
            True if exists, else False.
        """
        vertices_loop = list(polygon.vertices)
        vertices_loop = vertices_loop + [vertices_loop[0]]

        for i in range(len(vertices_loop) - 1):
            pt1 = vertices_loop[i]
            pt2 = vertices_loop[i + 1]

            if not self.pt_exists(pt1):
                return False

            node1 = self.node(coordinates_hash(pt1, self._tolerance))
            node2 = self.node(coordinates_hash(pt2, self._tolerance))
            if node2.key in [n.key for n in node1.adj_lst]:
                return False

        return True

    def adj_matrix(self):
        """Gets an adjacency matrix of the directed graph where:

        * 1 = adjacency from row node to col node.
        * 0 = no adjacency.

        Returns:
            N x N square matrix where N is number of nodes.
        """
        nodes = self.ordered_nodes
        # initialize a mtx with no adjacencies
        amtx = [[0 for i in range(self.node_count)]
                for j in range(self.node_count)]

        for i in range(self.node_count):
            adj_indices = [adj._order for adj in nodes[i].adj_lst]
            for adj_idx in adj_indices:
                amtx[i][adj_idx] = 1

        return amtx

    def adj_matrix_labels(self):
        """Returns a dictionary where label key corresponds to index in adj_matrix
        and value is node key"""
        return {i: node.key for i, node in enumerate(self.ordered_nodes)}

    def min_cycle(self, base_node, goal_node, ccw_only=False):
        """Identify the shortest interior cycle between two exterior nodes.

        Args:
            base_node: The first exterior node of the edge.
            goal_node: The end exterior node of the cycle that, together with
                the base_node, constitutes an exterior edge.
            ccw_only: A boolean to note whether the search should be limited
                to the counter-clockwise direction only. (Default: False).

        Returns:
            A list of nodes that form a polygon if the cycle exists, else None.
        """
        # set up a queue for exploring the graph
        explored = []
        queue = [[base_node]]
        orig_dir = base_node.pt - goal_node.pt \
            if base_node.key != goal_node.key else None
        # loop to traverse the graph  with the help of the queue
        while queue:
            path = queue.pop(0)
            node = path[-1]
            # make sure that the current node has not been visited
            if node not in explored:
                prev_dir = node.pt - path[-2].pt if len(path) > 1 else orig_dir
                # iterate over the neighbors to determine relevant nodes
                rel_neighbors, rel_angles = [], []
                last_resort_neighbors, last_resort_angles = [], []
                for neighbor in node.adj_lst:
                    if neighbor == goal_node:  # the shortest path was found!
                        path.append(goal_node)
                        return path
                    edge_dir = neighbor.pt - node.pt
                    cw_angle = prev_dir.angle_clockwise(edge_dir * -1) \
                        if prev_dir is not None else math.pi
                    if 1e-5 < cw_angle < (2 * math.pi) - 1e-5:
                        rel_neighbors.append(neighbor)
                        rel_angles.append(cw_angle)
                    else:  # try to avoid back-tracking along the search
                        last_resort_neighbors.append(neighbor)
                        last_resort_angles.append(cw_angle)
                if len(rel_neighbors) == 0:  # back tracking is the only option
                    rel_neighbors = last_resort_neighbors
                    rel_angles = last_resort_angles
                # sort the neighbors by clockwise angle
                if len(rel_neighbors) > 1:
                    rel_neighbors = [n for _, n in sorted(zip(rel_angles, rel_neighbors),
                                                          key=lambda pair: pair[0])]
                # add the relevant neighbors to the path and the queue
                if ccw_only:
                    new_path = list(path)
                    new_path.append(rel_neighbors[0])
                    queue.append(new_path)
                else:  # add all neighbors to the search
                    for neighbor in rel_neighbors:
                        new_path = list(path)
                        new_path.append(neighbor)
                        queue.append(new_path)
                explored.append(node)
        # if we reached the end of the queue, then no path was found
        return None

    def all_min_cycles(self):
        """Get a list of lists where each sub-list is a minimum cycle of Nodes.

        The combination of all min cycles should account for the full area of
        the input shape if the DirectedGraphNetwork was made using any of the
        class methods that work from polygons. If the DirectedGraphNetwork was made
        using the from_shape_to_split method, the resulting cycles here represent
        the input shape split with the split_segments.
        """
        # first, figure out how many loops each node should be a part of
        node_cycle_counts = {}
        for node in self.nodes:
            node_cycle_counts[node.key] = len(node.adj_lst)

        # loop through the nodes until all min cycles have been identified
        all_cycles = []
        iter_count = 0
        max_iter = len(self.nodes)
        remaining_nodes = self.ordered_nodes
        explored_nodes = set()

        while len(remaining_nodes) > 1 and iter_count < max_iter:
            # try to identify two connected nodes which we can use to build a cycle
            cycle_root = remaining_nodes[0]
            next_node = cycle_root  # if we can't find a connected node, connect to self
            ext_cycle = False
            if cycle_root.exterior:  # exterior cycles tend to have clear connections
                next_node = DirectedGraphNetwork.next_exterior_node(cycle_root)
                if next_node is not None:
                    ext_cycle = True
                else:
                    next_node = cycle_root
            if not ext_cycle:  # see if we can connect it to another incomplete node
                for _next_node in cycle_root.adj_lst:
                    if node_cycle_counts[_next_node.key] != 0:
                        next_node = _next_node
                        ext_cycle = True
                        break

            # find the minimum cycle by searching counter-clockwise
            min_cycle = self.min_cycle(next_node, cycle_root, True)

            # if we found a minimum cycle, evaluate its validity by node connections
            if min_cycle is not None and len(min_cycle) >= 3:
                if not ext_cycle:
                    min_cycle.pop(-1)  # take out the last duplicated node
                is_valid_cycle = True
                for node in min_cycle:
                    if node_cycle_counts[node.key] - 1 < 0:  # we are re-traversing
                        is_valid_cycle = False  # not a valid cycle

                # add the valid cycle to the list to be returned
                if is_valid_cycle:
                    for node in min_cycle:
                        node_cycle_counts[node.key] = node_cycle_counts[node.key] - 1
                        if node_cycle_counts[node.key] == 0:  # all cycles for node found
                            for i, r_node in enumerate(remaining_nodes):
                                if r_node.key == node.key:
                                    remaining_nodes.pop(i)
                                    break
                    all_cycles.append(min_cycle)
                    for node in min_cycle:
                        explored_nodes.add(node.key)

            # reorder the remaining nodes so unexplored nodes get prioritized
            if len(remaining_nodes) != 0:
                for j, node in enumerate(remaining_nodes):
                    if node.key not in explored_nodes:
                        break
                remaining_nodes.insert(0, remaining_nodes.pop(j))
            iter_count += 1

        # if we were not able to address all nodes, see if they are all in the same loop
        if len(remaining_nodes) >= 3:
            current_node = remaining_nodes.pop(0)
            current_node_adj = [node.key for node in node.adj_lst]
            last_cycle = [current_node]
            iter_count, max_iter = 0, len(remaining_nodes)
            while len(remaining_nodes) > 0 and iter_count < max_iter:
                for k, node in enumerate(remaining_nodes):
                    if node.key in current_node_adj:
                        current_node = remaining_nodes.pop(k)
                        current_node_adj = [node.key for node in node.adj_lst]
                        last_cycle.append(current_node)
                        break
                iter_count += 1
            if len(last_cycle) > 2:
                all_cycles.append(last_cycle)

        return all_cycles

    def exterior_cycle(self, cycle_root):
        """Compute exterior boundary from a given node.

        This method assumes that exterior edges are naked (unidirectional) and
        interior edges are bidirectional.

        Args:
            cycle_root: Starting Node in exterior cycle.

        Returns:
            List of nodes on exterior if a cycle exists, else None.
        """
        # Get the first exterior edge
        curr_node = cycle_root
        next_node = DirectedGraphNetwork.next_exterior_node(curr_node)
        if not next_node:
            return None

        # loop through the cycle until we get it all or run out of points
        max_iter = self.node_count + 1  # maximum length a cycle can be
        ext_cycle = [curr_node]
        iter_count = 0
        while next_node.key != cycle_root.key:
            ext_cycle.append(next_node)
            next_node = DirectedGraphNetwork.next_exterior_node(next_node)
            if not next_node:
                return None  # we have hit a dead end in the cycle
            iter_count += 1
            if iter_count > max_iter:
                break  # we have gotten stuck in a loop

        return ext_cycle

    def exterior_cycles(self):
        """Get a list of lists where each sub-list is an exterior cycle of Nodes.

        Exterior cycles refer to the cycles of both the boundary and the holes
        of the DirectedGraphNetwork was created using the from_shape_to_split
        class method.
        """
        exterior_poly_lst = []  # list to store cycles
        explored_nodes = set()  # set to note explored exterior nodes
        max_iter = self.node_count + 1  # maximum length a cycle can be

        # loop through all of the nodes of the graph and find cycles
        for root_node in self.ordered_nodes:
            # make a note that the current node has been explored
            explored_nodes.add(root_node.key)  # mark the node as explored
            # get next exterior adjacent node and check that it's valid
            next_node = self.next_exterior_node(root_node)  # mark the node as explored
            is_valid = (next_node is not None) and (next_node.key not in explored_nodes)
            if not is_valid:
                continue
            # make a note that the next node has been explored
            explored_nodes.add(next_node.key)

            # traverse the loop of points until we get back to start or hit a dead end
            exterior_poly = [root_node]
            prev_node = root_node
            iter_count = 0
            while next_node.key != root_node.key:
                exterior_poly.append(next_node)
                explored_nodes.add(next_node.key)  # mark the node as explored
                follow_node = self.next_exterior_node_no_backtrack(
                    next_node, prev_node, explored_nodes)
                prev_node = next_node  # set as the previous node for the next step
                next_node = follow_node
                if next_node is None:
                    break  # we have hit a dead end in the cycle
                iter_count += 1
                if iter_count > max_iter:
                    print('Extraction of core polygons hit an endless loop.')
                    break  # we have gotten stuck in a loop
            exterior_poly_lst.append(exterior_poly)

        # return all of the exterior loops that were found
        return exterior_poly_lst

    @staticmethod
    def next_exterior_node_no_backtrack(node, previous_node, explored_nodes):
        """Get the next exterior node adjacent to the input node.

        This method is similar to the next_exterior_node method but it includes
        extra checks to handle intersections with 3 or more segments in the
        graph exterior cycles. In these cases a set of previously explored_nodes
        is used to ensure that no back-tracking happens over the search of the
        network, which can lead to infinite looping through the graph. Furthermore,
        the previous_node is used to select the pathway with the smallest angle
        difference with the previous direction. This leads the result towards
        minimal polygons with fewer self-intersecting loops.

        Args:
            node: A Node object for which the next node will be returned.
            previous_node: A Node object for the node that came before
                the current one in the loop. This will be used in the event that
                multiple exterior nodes are found connecting to the input node.
                In this case, the exterior node with the smallest angle difference
                with the previous direction will be returned. This leads the
                result towards minimal polygons and away from self-intersecting
                exterior loops like a bowtie.

        Returns:
            Next node that defines exterior edge, or None if all adjacencies are
            bidirectional.
        """
        # loop through the all adjacent nodes and determine if they are exterior
        next_nodes = []
        for _next_node in node.adj_lst:
            if _next_node.exterior:  # user has labeled it as exterior; we're done!
                return _next_node
            elif _next_node.exterior is None:  # don't know if it's interior or exterior
                # if user-assigned attribute isn't defined, check bi-directionality
                if not DirectedGraphNetwork.is_edge_bidirect(node, _next_node):
                    next_nodes.append(_next_node)

        # evaluate whether there is one obvious choice for the next node
        if len(next_nodes) <= 1:
            return next_nodes[0] if len(next_nodes) == 1 else None
        next_nodes = [nn for nn in next_nodes if nn.key not in explored_nodes]
        if len(next_nodes) <= 1:
            return next_nodes[0] if len(next_nodes) == 1 else None

        # if we have multiple exterior nodes, use the previous node to find the best one
        prev_dir = previous_node.pt - node.pt  # yields a vector
        next_angles = []
        for next_node in next_nodes:
            edge_dir = next_node.pt - node.pt  # yields a vector
            next_angles.append(prev_dir.angle(edge_dir * -1))
        sorted_nodes = [n for _, n in sorted(zip(next_angles, next_nodes),
                                             key=lambda pair: pair[0])]
        return sorted_nodes[0]  # return the node making the smallest angle

    @staticmethod
    def next_exterior_node(node):
        """Get the next exterior node adjacent to consumed node.

        If there are adjacent nodes that are labeled as exterior, with True or
        False defining the Node.exterior property, the first of such nodes in
        the adjacency list will be returned as the next one. Otherwise, the
        bi-directionality will be used to determine whether the next node is
        exterior.

        Args:
            node: A Node object for which the next node will be returned.

        Returns:
            Next node that defines exterior edge, or None if all adjacencies are
            bidirectional.
        """
        # loop through the adjacency and find an exterior node
        for _next_node in node.adj_lst:
            if _next_node.exterior:  # user has labeled it as exterior; we're done!
                return _next_node
            elif _next_node.exterior is None:  # don't know if it's interior or exterior
                # if user-assigned attribute isn't defined, check bi-directionality
                if not DirectedGraphNetwork.is_edge_bidirect(node, _next_node):
                    return _next_node
        return None

    @staticmethod
    def is_edge_bidirect(node1, node2):
        """Are two nodes bidirectional.

        Args:
            node1: Node object
            node2: Node object

        Returns:
            True if node1 and node2 are in each other's adjacency list,
            else False.
        """
        return node1.key in (n.key for n in node2.adj_lst) and \
            node2.key in (n.key for n in node1.adj_lst)

    @staticmethod
    def _intersect_segments(segments, additional_segments, tolerance):
        """Intersect a list of LineSegment2D and split them.

        Args:
            segments: A list of LineSegment2D for the segments to be split/intersected.
            additional_segments: A list of additional LineSegment2Ds, which will be
                used to split the input segments but will not be included in the
                output themselves.
            tolerance: The tolerance at which the intersection will be computed.

        Returns:
            A list of LineSegment2D for the input segments split through
            self-intersection and intersection with the additional_segments.
        """
        # make sure that we are working with lists
        if not isinstance(segments, list):
            segments = list(segments)
        if not isinstance(additional_segments, list):
            additional_segments = list(additional_segments)

        # extend segments a little to ensure intersections happen
        under_tol = tolerance * 0.99
        ext_segments = []
        for seg in segments + additional_segments:
            m_v = seg.v.normalize() * under_tol
            ext_seg = LineSegment2D.from_end_points(seg.p1.move(-m_v), seg.p2.move(m_v))
            ext_segments.append(ext_seg)

        # compute all of the intersection points across the segments
        intersect_pts = [[] for _ in segments]
        for i, seg in enumerate(segments):
            try:
                for other_seg in ext_segments[:i] + ext_segments[i + 1:]:
                    int_pt = intersect_line_segment2d(seg, other_seg)
                    if int_pt is None or int_pt.is_equivalent(seg.p1, tolerance) or \
                            int_pt.is_equivalent(seg.p2, tolerance):
                        continue
                    # we have found an intersection point where segments should be split
                    intersect_pts[i].append(int_pt)
            except IndexError:
                pass  # we have reached the end of the list

        # loop through the segments and split them at the intersection points
        split_segments = []
        for seg, split_pts in zip(segments, intersect_pts):
            if len(split_pts) == 0:
                split_segments.append(seg)
            elif len(split_pts) == 1:  # split the segment in two
                int_pt = split_pts[0]
                split_segments.append(LineSegment2D.from_end_points(seg.p1, int_pt))
                split_segments.append(LineSegment2D.from_end_points(int_pt, seg.p2))
            else:  # sort the points along the segment to split it
                pt_dists = [seg.p1.distance_to_point(ipt) for ipt in split_pts]
                sort_obj = sorted(zip(pt_dists, split_pts), key=lambda pair: pair[0])
                sort_pts = [x for _, x in sort_obj]
                sort_pts.append(seg.p2)
                pr_pt = seg.p1
                for s_pt in sort_pts:
                    if not pr_pt.is_equivalent(s_pt, tolerance):
                        split_segments.append(LineSegment2D.from_end_points(pr_pt, s_pt))
                    pr_pt = s_pt

        return split_segments

    @staticmethod
    def _remove_segments_outside_boundary(segments, boundary, tolerance):
        """Remove LineSegment2D that are outside the boundary of the parent shape.

        This can be used to clean up the result after intersection of segments.

        Args:
            segments: A list of LineSegment2D to be filtered for whether they are
                inside the boundary.
            boundary: A Polygon2D for the boundary of the shape. Segments that lie
                outside of this boundary beyond the tolerance will be removed from
                the result.
            tolerance: The tolerance for distinguishing whether skeleton points lie
                outside the boundary.

        Returns:
            A list of LineSegment2D objects with segments removed that outside
            of the boundary.
        """
        clean_segments = []
        for seg in segments:
            p1, p2 = seg.p1, seg.p2
            if boundary.point_relationship(p2, tolerance) >= 0 and \
                    boundary.point_relationship(p1, tolerance) >= 0:
                clean_segments.append(seg)
        return clean_segments

    def __repr__(self):
        """Represent PolygonDirectedGraph."""
        s = ''
        for n in self.ordered_nodes:
            s += '{}, [{}]\n'.format(
                n.pt.to_array(),
                ', '.join([str(_n.pt.to_array()) for _n in n.adj_lst]))
        return s
