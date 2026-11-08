"""Monte-Carlo benchmark of planners across random maps."""
from __future__ import annotations

import csv
import statistics as st

from .environment import GridMap
from .planners import astar, dijkstra, path_length, rrt
from .smoothing import shortcut


def run_benchmark(n_maps: int = 30, width: int = 60, height: int = 40,
                  robot_radius: int = 1, base_seed: int = 100,
                  csv_file: str | None = None):
    start, goal = (4, 4), (width - 5, height - 5)
    rows = []
    for k in range(n_maps):
        raw = GridMap.random(width, height, seed=base_seed + k,
                             keep_clear=[start, goal])
        gmap = raw.inflate(robot_radius)
        for planner in (dijkstra, astar, rrt):
            r = planner(gmap, start, goal)
            length = r.length if r.success else float("nan")
            sm_len = path_length(shortcut(r.path, gmap)) if r.success else float("nan")
            rows.append(dict(map_seed=base_seed + k, planner=r.name, success=int(r.success),
                             length=length, smoothed_length=sm_len,
                             runtime_ms=r.runtime_s * 1000, expanded=r.expanded))
    if csv_file:
        with open(csv_file, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    summary = {}
    for name in sorted({r["planner"] for r in rows}):
        sub = [r for r in rows if r["planner"] == name]
        ok = [r for r in sub if r["success"]]
        summary[name] = dict(
            success_rate=len(ok) / len(sub),
            mean_length=st.mean(r["length"] for r in ok) if ok else float("nan"),
            mean_smoothed_length=st.mean(r["smoothed_length"] for r in ok) if ok else float("nan"),
            mean_runtime_ms=st.mean(r["runtime_ms"] for r in sub),
            mean_expanded=st.mean(r["expanded"] for r in sub),
        )
    return rows, summary


def format_summary(summary: dict) -> str:
    head = f"{'Planner':10s} {'Success':>8s} {'Length':>8s} {'Smoothed':>9s} {'ms':>8s} {'Expanded':>9s}"
    lines = [head, "-" * len(head)]
    for n, s in summary.items():
        lines.append(f"{n:10s} {s['success_rate']:8.0%} {s['mean_length']:8.1f} "
                     f"{s['mean_smoothed_length']:9.1f} {s['mean_runtime_ms']:8.1f} "
                     f"{s['mean_expanded']:9.0f}")
    return "\n".join(lines)
