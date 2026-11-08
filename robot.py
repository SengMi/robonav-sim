"""Differential-drive robot kinematics and a pure-pursuit path follower."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .planners import Point


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class DiffDriveRobot:
    x: float
    y: float
    theta: float = 0.0
    max_v: float = 2.0      # cells / s
    max_w: float = 2.5      # rad / s

    def step(self, v: float, w: float, dt: float) -> None:
        v = max(-self.max_v, min(self.max_v, v))
        w = max(-self.max_w, min(self.max_w, w))
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta = wrap_angle(self.theta + w * dt)


class PurePursuit:
    """Steers towards a lookahead point on the path; slows for sharp turns
    and when approaching the goal."""

    def __init__(self, path: list[Point], lookahead: float = 2.0,
                 goal_tolerance: float = 0.4, k_heading: float = 3.0):
        self.path = path
        self.lookahead = lookahead
        self.goal_tolerance = goal_tolerance
        self.k_heading = k_heading
        self._idx = 0

    def reached(self, robot: DiffDriveRobot) -> bool:
        return math.dist((robot.x, robot.y), self.path[-1]) < self.goal_tolerance

    def command(self, robot: DiffDriveRobot) -> tuple[float, float]:
        pos = (robot.x, robot.y)
        while (self._idx < len(self.path) - 1
               and math.dist(pos, self.path[self._idx]) < self.lookahead):
            self._idx += 1
        tx, ty = self.path[self._idx]
        alpha = wrap_angle(math.atan2(ty - robot.y, tx - robot.x) - robot.theta)
        dist_goal = math.dist(pos, self.path[-1])
        v = robot.max_v * max(0.15, math.cos(alpha)) * min(1.0, dist_goal / (2 * self.lookahead))
        w = self.k_heading * alpha
        return v, w
