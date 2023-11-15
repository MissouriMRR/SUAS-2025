"""Run tests for waypoint pathfinding."""

import json
from pathlib import Path
import random
import time
from typing import Callable, Iterable, TypedDict

import utm

from flight.waypoint.geometry import lerp, Point
from flight.waypoint.graph import GraphNode
from flight.waypoint import pathfinding


class Coordinate(TypedDict):
    """A coordinate in waypoint_data.json"""

    latitude: float
    longitude: float


class Flyzones(TypedDict):
    """flyzones in waypoint_data.json"""

    altitudeMin: float
    altitudeMax: float
    boundaryPoints: list[Coordinate]


class WaypointData(TypedDict):
    """TypedDict representing waypoint_data.json"""

    flyzones: Flyzones
    waypoints: list[Coordinate]


def random_point_in_shape(vertices: Iterable[Point]) -> Point:
    """
    Generate a random point within a shape.

    Parameters
    ----------
    vertices : list[Point]
        The vertices of the boundary of the shape in the correct order.

    Returns
    -------
    Point
        A random point within the shape
    """
    xmin: float = min(point.x for point in vertices)
    xmax: float = max(point.x for point in vertices)
    ymin: float = min(point.y for point in vertices)
    ymax: float = max(point.y for point in vertices)

    while True:
        t_x: float = random.random()
        t_y: float = random.random()
        point: Point = Point(lerp(xmin, xmax, t_x), lerp(ymin, ymax, t_y))
        if point.is_inside_shape(vertices):
            return point


def draw_random_paths(
    vertices: list[Point],
    nodes: list[GraphNode[Point, float]],
    to_svg_coord: Callable[[float, float], tuple[float, float]],
) -> Iterable[str]:
    """
    Add random paths between points in a shape to an SVG drawing.

    Parameters
    ----------
    vertices : list[Point]
        The vertices of the boundary of the shape in the correct order.
    nodes : list[GraphNode[Point, float]]
        The nodes in the pathfinding search graph.
    to_svg_coord : Callable[[float, float], tuple[float, float]]
        A callable to convert points to the coordinate space of the SVG
        drawing.

    Yields
    -------
    str
        Strings to append to the SVG string.
    """
    i: int = 0
    path_count_to_generate: int = 10
    while i < path_count_to_generate:
        src: Point = random_point_in_shape(vertices)
        dst: Point = random_point_in_shape(vertices)
        src_x, src_y = to_svg_coord(src.x, src.y)
        dst_x, dst_y = to_svg_coord(dst.x, dst.y)

        begin_ns: int = time.perf_counter_ns()
        path: list[Point] = list(pathfinding.shortest_path_between(src, dst, nodes))
        end_ns: int = time.perf_counter_ns()

        if i < path_count_to_generate // 2:
            if len(path) != 1:
                continue
        else:
            if len(path) <= 1:
                continue

        time_microseconds = (end_ns - begin_ns) / 1000
        print(f"Time: {time_microseconds:.3f} microseconds")

        yield f'<path d="M {src_x:.2f},{src_y:.2f}'
        for point in path:
            x, y = to_svg_coord(point.x, point.y)
            yield f" L {x:.2f},{y:.2f}"

        yield '" stroke-width="2" stroke="magenta" fill="none"/>'

        yield f'<circle cx="{src_x:.2f}" cy="{src_y:.2f}" r="4" fill="magenta"/>'
        yield f'<circle cx="{dst_x:.2f}" cy="{dst_y:.2f}" r="4" fill="magenta"/>'

        i += 1


def draw_graph(nodes: list[GraphNode[Point, float]]) -> str:
    """
    Generate an SVG drawing of a graph.

    Parameters
    ----------
    nodes : list[GraphNode[Point, float]]
        The nodes in the pathfinding search graph.

    Returns
    -------
    str
        The SVG code representing the drawing of the graph.
    """
    svg_size: int = 1000

    string_builder: list[str] = []
    string_builder.append(
        f'<svg xmlns="http://www.w3.org/2000/svg"'
        f' width="{svg_size}" height="{svg_size}"'
        f' viewBox="0 0 {svg_size} {svg_size}">'
    )

    string_builder.append(
        f'<rect x="0" y="0" width="{svg_size}" height="{svg_size}" fill="white"/>'
    )

    xmin: float = min(node.value.x for node in nodes)
    xmax: float = max(node.value.x for node in nodes)
    ymin: float = min(node.value.y for node in nodes)
    ymax: float = max(node.value.y for node in nodes)

    size: float = max(xmax - xmin, ymax - ymin)

    center_x: float = (xmin + xmax) / 2
    center_y: float = (ymin + ymax) / 2
    xmin = center_x - 0.6 * size
    xmax = center_x + 0.6 * size
    ymin = center_y - 0.6 * size
    ymax = center_y + 0.6 * size

    def to_svg_coord(x: float, y: float) -> tuple[float, float]:
        t_x: float = (x - xmin) / (xmax - xmin)
        t_y: float = (y - ymin) / (ymax - ymin)
        return t_x * svg_size, (1 - t_y) * svg_size

    for node_1 in nodes:
        for node_2 in nodes:
            if node_1 == node_2:
                continue

            if not node_1.is_connected_to(node_2):
                continue

            x_1, y_1 = to_svg_coord(node_1.value.x, node_1.value.y)
            x_2, y_2 = to_svg_coord(node_2.value.x, node_2.value.y)

            string_builder.append(
                f'<path d="M {x_1:.2f},{y_1:.2f} L {x_2:.2f},{y_2:.2f}"'
                ' stroke-width="1" stroke="gray" fill="none"/>'
            )

    for node in nodes:
        x, y = to_svg_coord(node.value.x, node.value.y)
        string_builder.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="gray"/>')

    string_builder.append(
        "".join(draw_random_paths([node.value for node in nodes], nodes, to_svg_coord))
    )

    string_builder.append("</svg>")

    return "".join(string_builder)


def main() -> None:
    """Run tests for waypoint pathfinding."""
    waypoint_data_filepath: str = str(
        Path(__file__)
        .parent.absolute()
        .joinpath("..")
        .joinpath("data")
        .joinpath("waypoint_data.json")
    )
    with open(waypoint_data_filepath, "r", encoding="utf-8") as file:
        waypoint_data: WaypointData = json.load(file)

    boundary_coords: list[Coordinate] = waypoint_data["flyzones"]["boundaryPoints"]
    boundary_coords.pop()  # The last point is a duplicate of the first in the json
    force_zone_number: int
    force_zone_letter: str
    _, _, force_zone_number, force_zone_letter = utm.from_latlon(
        boundary_coords[0]["latitude"], boundary_coords[0]["longitude"]
    )

    boundary_vertices: list[Point] = []
    for coord in waypoint_data["flyzones"]["boundaryPoints"]:
        easting, northing, _, _ = utm.from_latlon(
            coord["latitude"], coord["longitude"], force_zone_number, force_zone_letter
        )
        boundary_vertices.append(Point(easting, northing))

    graph: list[GraphNode[Point, float]] = pathfinding.create_pathfinding_graph(
        boundary_vertices, 0.0
    )
    print(draw_graph(graph))


if __name__ == "__main__":
    main()
