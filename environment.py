"""Occupancy-grid environment utilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GridMap:
    """Binary occupancy grid. ``grid[y, x] == 1`` means obstacle.

    Cell centres sit on integer coordinates, so a world point (x, y)
    belongs to cell (round(x), round(y)).
    """

    grid: np.ndarray

    @property
    def height(self) -> int:
        return self.grid.shape[0]

    @property
    def width(self) -> int:
        return self.grid.shape[1]

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.grid[y, x] == 0

    def world_is_free(self, x: float, y: float) -> bool:
        return self.is_free(int(round(x)), int(round(y)))

    def segment_is_free(self, p: tuple[float, float], q: tuple[float, float],
                        step: float = 0.25) -> bool:
        """Line-of-sight test by sampling along the segment p -> q."""
        dist = float(np.hypot(q[0] - p[0], q[1] - p[1]))
        n = max(int(dist / step), 1)
        for t in np.linspace(0.0, 1.0, n + 1):
            if not self.world_is_free(p[0] + t * (q[0] - p[0]),
                                      p[1] + t * (q[1] - p[1])):
                return False
        return True

    def inflate(self, radius: int) -> "GridMap":
        """Return a new map whose obstacles are grown by ``radius`` cells
        (configuration-space expansion for a circular robot)."""
        if radius <= 0:
            return GridMap(self.grid.copy())
        h, w = self.grid.shape
        padded = np.pad(self.grid, radius, constant_values=0)
        out = np.zeros_like(self.grid)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    out |= padded[radius + dy: radius + dy + h,
                                  radius + dx: radius + dx + w]
        return GridMap(out)

    @staticmethod
    def empty(width: int, height: int) -> "GridMap":
        return GridMap(np.zeros((height, width), dtype=np.uint8))

    @staticmethod
    def random(width: int = 60, height: int = 40, n_obstacles: int = 14,
               seed: int | None = None,
               keep_clear: list[tuple[int, int]] | None = None,
               clear_radius: int = 3) -> "GridMap":
        """Random rectangular obstacles; border cells are walls."""
        rng = np.random.default_rng(seed)
        g = np.zeros((height, width), dtype=np.uint8)
        g[0, :] = g[-1, :] = 1
        g[:, 0] = g[:, -1] = 1
        for _ in range(n_obstacles):
            w = int(rng.integers(3, 10))
            h = int(rng.integers(3, 10))
            x = int(rng.integers(1, max(2, width - w - 1)))
            y = int(rng.integers(1, max(2, height - h - 1)))
            g[y:y + h, x:x + w] = 1
        for cx, cy in keep_clear or []:
            g[max(cy - clear_radius, 1):cy + clear_radius + 1,
              max(cx - clear_radius, 1):cx + clear_radius + 1] = 0
        return GridMap(g)
