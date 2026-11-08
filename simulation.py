"""Closed-loop simulation of a robot following a planned path."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .environment import GridMap
from .planners import Point
from .robot import DiffDriveRobot, PurePursuit
from .smoothing import densify


@dataclass
class SimResult:
    trajectory: list[tuple[float, float, float]] = field(default_factory=list)
    reached_goal: bool = False
    collided: bool = False
    time_s: float = 0.0

    @property
    def distance_travelled(self) -> float:
        return float(sum(math.dist(a[:2], b[:2])
                         for a, b in zip(self.trajectory, self.trajectory[1:])))


def simulate(true_map: GridMap, path: list[Point], dt: float = 0.05,
             max_time: float = 120.0) -> SimResult:
    """Drive a robot along ``path``. Collisions are checked against
    ``true_map`` (the *un-inflated* map)."""
    res = SimResult()
    if len(path) < 2:
        return res
    dense = densify(path, 0.5)
    heading = math.atan2(dense[1][1] - dense[0][1], dense[1][0] - dense[0][0])
    robot = DiffDriveRobot(dense[0][0], dense[0][1], heading)
    ctrl = PurePursuit(dense)
    t = 0.0
    while t < max_time:
        res.trajectory.append((robot.x, robot.y, robot.theta))
        if not true_map.world_is_free(robot.x, robot.y):
            res.collided = True
            break
        if ctrl.reached(robot):
            res.reached_goal = True
            break
        v, w = ctrl.command(robot)
        robot.step(v, w, dt)
        t += dt
    res.time_s = t
    return res
