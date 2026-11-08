"""Global path planners: Dijkstra, A*, and RRT."""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .environment import GridMap

Point = tuple[float, float]

_MOVES = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
          (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
          (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]


@dataclass
class PlanResult:
    name: str
    path: list[Point] = field(default_factory=list)
    expanded: int = 0
    runtime_s: float = 0.0

    @property
    def success(self) -> bool:
        return len(self.path) > 0

    @property
    def length(self) -> float:
        return path_length(self.path)


def path_length(path: list[Point]) -> float:
    return float(sum(math.dist(a, b) for a, b in zip(path, path[1:])))


def octile(a: Point, b: Point) -> float:
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)


def _grid_search(gmap: GridMap, start: tuple[int, int], goal: tuple[int, int],
                 use_heuristic: bool, name: str) -> PlanResult:
    t0 = time.perf_counter()
    res = PlanResult(name)
    if not gmap.is_free(*start) or not gmap.is_free(*goal):
        res.runtime_s = time.perf_counter() - t0
        return res

    g_cost = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    counter = 0  # tie-breaker so heap never compares tuples of cells
    heap = [(0.0, counter, start)]
    closed: set[tuple[int, int]] = set()

    while heap:
        _, _, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        closed.add(cur)
        res.expanded += 1
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            res.path = [(float(x), float(y)) for x, y in reversed(path)]
            break
        for dx, dy, cost in _MOVES:
            nx, ny = cur[0] + dx, cur[1] + dy
            if not gmap.is_free(nx, ny):
                continue
            # forbid cutting corners between two diagonal obstacles
            if dx != 0 and dy != 0 and not (gmap.is_free(cur[0] + dx, cur[1])
                                            and gmap.is_free(cur[0], cur[1] + dy)):
                continue
            nxt = (nx, ny)
            ng = g_cost[cur] + cost
            if ng < g_cost.get(nxt, float("inf")):
                g_cost[nxt] = ng
                parent[nxt] = cur
                h = octile(nxt, goal) if use_heuristic else 0.0
                counter += 1
                heapq.heappush(heap, (ng + h, counter, nxt))
    res.runtime_s = time.perf_counter() - t0
    return res


def dijkstra(gmap: GridMap, start, goal) -> PlanResult:
    return _grid_search(gmap, tuple(map(int, start)), tuple(map(int, goal)),
                        use_heuristic=False, name="Dijkstra")


def astar(gmap: GridMap, start, goal) -> PlanResult:
    return _grid_search(gmap, tuple(map(int, start)), tuple(map(int, goal)),
                        use_heuristic=True, name="A*")


def rrt(gmap: GridMap, start, goal, max_iter: int = 6000, step: float = 2.5,
        goal_bias: float = 0.1, seed: int | None = 0) -> PlanResult:
    """Basic Rapidly-exploring Random Tree (probabilistically complete)."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    res = PlanResult("RRT")
    start, goal = (float(start[0]), float(start[1])), (float(goal[0]), float(goal[1]))
    nodes = [start]
    parents = [-1]

    for _ in range(max_iter):
        res.expanded += 1
        if rng.random() < goal_bias:
            sample = goal
        else:
            sample = (rng.uniform(0, gmap.width - 1), rng.uniform(0, gmap.height - 1))
        arr = np.asarray(nodes)
        i = int(np.argmin(np.hypot(arr[:, 0] - sample[0], arr[:, 1] - sample[1])))
        near = nodes[i]
        d = math.dist(near, sample)
        if d < 1e-9:
            continue
        r = min(step, d) / d
        new = (near[0] + r * (sample[0] - near[0]), near[1] + r * (sample[1] - near[1]))
        if not gmap.segment_is_free(near, new):
            continue
        nodes.append(new)
        parents.append(i)
        if math.dist(new, goal) <= step and gmap.segment_is_free(new, goal):
            nodes.append(goal)
            parents.append(len(nodes) - 2)
            path, k = [], len(nodes) - 1
            while k != -1:
                path.append(nodes[k])
                k = parents[k]
            res.path = list(reversed(path))
            break
    res.runtime_s = time.perf_counter() - t0
    return res


PLANNERS = {"dijkstra": dijkstra, "astar": astar, "rrt": rrt}
