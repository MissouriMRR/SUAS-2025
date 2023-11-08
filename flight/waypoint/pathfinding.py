"""
Defines a function to find the shortest path between two waypoints that
stays within the mission flight boundary.
"""
import heapq
from typing import Iterable, NamedTuple, TypeAlias

from flight.waypoint.geometry import LineSegment, Point
from flight.waypoint.graph import GraphNode

Node: TypeAlias = GraphNode[Point, float]


class _SearchNode(NamedTuple):
    """
    Contains a node to visit. Used when searching a graph.

    Attributes
    ----------
    priority : float
        The priority of the node to search. A lower value means a higher
        priority.
    distance_so_far : float
        The distance traveled to get to this node.
    visiting_node: Node
        The node that is visiting.
    node : Node
        The node to visit.
    """

    priority: float
    distance_so_far: float
    visitor: Node | None
    node: Node


def _visitors(goal_node: Node) -> Iterable[Point]:
    """
    Yields the positions of the visitors in a path to a goal node, with the
    goal node being last.

    Parameters
    ----------
    goal_node : Node
        The goal node in a search.

    Yields
    -------
    Point
        The positions of the visitors in a path to the goal node, from the
        start to (and  including) the goal node.
    """
    points: list[Point] = []
    visitor: Node | None = goal_node
    while visitor is not None:
        points.append(visitor.value)
        visitor = visitor.visitor

    points.reverse()
    for point in points:
        yield point


def _search(search_queue: list[_SearchNode], goal_node: Node) -> bool:
    """
    Searches for the goal node in the graph. Mutates the search queue passed
    in and the visitor attribute of the graph nodes.

    Parameters
    ----------
    search_queue : list[_SearchNode]
        A min priority queue used to search for the goal node. Must already
        contain the starting nodes in the graph in the correct order.
    goal_node : Node
        The goal node.

    Returns
    -------
    bool
        Whether the search was successful.
    """

    while search_queue:
        search_node: _SearchNode = heapq.heappop(search_queue)
        curr_distance_so_far: float = search_node.distance_so_far
        visitor: Node | None = search_node.visitor
        node: Node = search_node.node
        if node.visitor is not None:
            continue

        if node == goal_node:
            return True

        node.visitor = visitor
        visitor = node
        for node, weight in node.edges.items():
            if node.visitor is not None:
                continue

            distance_so_far: float = curr_distance_so_far + weight
            heapq.heappush(
                search_queue,
                _SearchNode(
                    distance_so_far + LineSegment(node.value, goal_node.value).length(),
                    distance_so_far,
                    visitor,
                    node,
                ),
            )

    return False


def shortest_path_between(src: Point, dst: Point, boundary: Iterable[Node]) -> Iterable[Point]:
    """
    Finds the shortest path between two points given a graph with all possible
    paths between boundary points

    Parameters
    ----------
    src : Point
        The point we're currently at.
    dst : Point
        The point to move to.
    boundary: Iterable[Node]
        Graph nodes with all the boundary points and possible paths between
        those points. The points must be in order, but it does not matter
        whether they are in clockwise or counterclockwise order.

    Yields
    -------
    Point
        The next point to move to in order to get to the destination using
        the shorted path.

    Raises
    ------
    RuntimeError
        If no path was found to the destination. In normal usage, this should
        never occur.
    """
    boundary_nodes: list[Node] = list(boundary)
    boundary_line_segments: list[LineSegment] = list(
        LineSegment.from_points((node.value for node in boundary_nodes), True)
    )

    straight_path: LineSegment = LineSegment(src, dst)
    if not any(
        straight_path.intersects(boundary_line_segment)
        for boundary_line_segment in boundary_line_segments
    ):
        yield dst
        return

    # We use the A* search algorithm

    goal_node: Node = Node(dst)
    search_queue: list[_SearchNode] = []

    for boundary_node in boundary_nodes:
        boundary_node.visitor = None

        straight_path = LineSegment(boundary_node.value, dst)
        if not any(
            straight_path.intersects(boundary_line_segment)
            for boundary_line_segment in boundary_line_segments
        ):
            weight: float = straight_path.length()
            boundary_node.connect(goal_node, weight)

        straight_path = LineSegment(src, boundary_node.value)
        if not any(
            straight_path.intersects(boundary_line_segment)
            for boundary_line_segment in boundary_line_segments
        ):
            distance_so_far: float = straight_path.length()
            heapq.heappush(
                search_queue,
                _SearchNode(
                    distance_so_far + LineSegment(boundary_node.value, dst).length(),
                    distance_so_far,
                    None,
                    boundary_node,
                ),
            )

    success: bool = _search(search_queue, goal_node)

    for boundary_node in boundary_nodes:
        boundary_node.disconnect(goal_node)

    if success:
        yield from _visitors(goal_node)
    else:
        raise RuntimeError("no path was found to the destination")


def create_pathfinding_graph(boundary: Iterable[Point], safety_margin: float) -> list[Node]:
    """
    Create a graph suitable to be used as the boundary graph when pathfinding

    Parameters
    ----------
    boundary : Iterable[Point]
        The vertices of the boundary. They must be in order, but it does not
        matter whether they are in clockwise or counterclockwise order.
    safety_margin : float
        How far to move the boundary vertices inward. The units are the same
        as the units for `boundary`.

    Returns
    -------
    list[Node]
        A list of graph nodes constituting the pathfinding graph.
    """
    points: list[Point] = list(boundary)
    points_moved_inward: list[Point] = []
    for point, line_segment_1, line_segment_2 in zip(
        points,
        LineSegment.from_points([points[-1]] + points[:-1], True),
        LineSegment.from_points(points, True),
    ):
        # How far to move the point inward according to the safety margin
        inward_diff: Point = (line_segment_1.p_1 - line_segment_1.p_2) + (
            line_segment_2.p_2 - line_segment_2.p_1
        )
        inward_diff /= inward_diff.distance_from_origin()

        if not (point + 1e-5 * inward_diff).is_inside_shape(points):
            inward_diff *= -1.0

        inward_diff *= safety_margin
        points_moved_inward.append(point + inward_diff)

    boundary_line_segments: list[LineSegment] = list(
        LineSegment.from_points(points_moved_inward, True)
    )

    # Rather inefficient
    # Thankfully, there shouldn't be too many boundary vertices
    nodes: list[Node] = [Node(point) for point in points_moved_inward]
    for node_1 in nodes:
        for node_2 in nodes:
            if node_1 == node_2:
                continue

            straight_path: LineSegment = LineSegment(node_1.value, node_2.value)

            if not any(
                straight_path.intersects(boundary_line_segment)
                for boundary_line_segment in boundary_line_segments
            ):
                node_1.connect(node_2, straight_path.length())

    return nodes
