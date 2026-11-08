"""Path post-processing: shortcut smoothing and densification."""
from __future__ import annotations

import math

from .environment import GridMap
from .planners import Point


def shortcut(path: list[Point], gmap: GridMap) -> list[Point]:
    """Greedy line-of-sight shortcutting: from each anchor jump to the
    farthest waypoint that is directly visible."""
    if len(path) < 3:
        return list(path)
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not gmap.segment_is_free(path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def densify(path: list[Point], spacing: float = 0.5) -> list[Point]:
    """Insert interpolated points so consecutive points are <= spacing apart."""
    if not path:
        return []
    out = [path[0]]
    for a, b in zip(path, path[1:]):
        n = max(int(math.ceil(math.dist(a, b) / spacing)), 1)
        for k in range(1, n + 1):
            t = k / n
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return out
