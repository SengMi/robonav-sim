"""Matplotlib plots and GIF animation."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from .environment import GridMap
from .planners import PlanResult
from .simulation import SimResult

COLORS = {"Dijkstra": "tab:blue", "A*": "tab:green", "RRT": "tab:orange"}


def plot_plans(gmap: GridMap, start, goal, results: list[PlanResult],
               out_file: str, smoothed: dict[str, list] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(gmap.grid, cmap="Greys", origin="lower")
    for r in results:
        if not r.success:
            continue
        xs, ys = zip(*r.path)
        ax.plot(xs, ys, color=COLORS.get(r.name, "k"), lw=2,
                label=f"{r.name} ({r.length:.1f})")
        if smoothed and r.name in smoothed:
            sx, sy = zip(*smoothed[r.name])
            ax.plot(sx, sy, "--", color=COLORS.get(r.name, "k"), lw=1.5,
                    label=f"{r.name} smoothed")
    ax.plot(*start, "go", ms=10, label="start")
    ax.plot(*goal, "r*", ms=14, label="goal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Global path planning comparison")
    fig.tight_layout()
    fig.savefig(out_file, dpi=130)
    plt.close(fig)


def animate(gmap: GridMap, path, sim: SimResult, out_file: str,
            stride: int = 6) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.imshow(gmap.grid, cmap="Greys", origin="lower")
    px, py = zip(*path)
    ax.plot(px, py, "c--", lw=1.2, label="planned path")
    trail, = ax.plot([], [], "r-", lw=1.5, label="robot trail")
    body, = ax.plot([], [], "ro", ms=9)
    heading, = ax.plot([], [], "k-", lw=2)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Pure-pursuit path following")
    traj = sim.trajectory[::stride] + [sim.trajectory[-1]]

    import math

    def update(i):
        xs = [p[0] for p in traj[:i + 1]]
        ys = [p[1] for p in traj[:i + 1]]
        trail.set_data(xs, ys)
        x, y, th = traj[i]
        body.set_data([x], [y])
        heading.set_data([x, x + 1.5 * math.cos(th)], [y, y + 1.5 * math.sin(th)])
        return trail, body, heading

    anim = FuncAnimation(fig, update, frames=len(traj), interval=40, blit=True)
    anim.save(out_file, writer=PillowWriter(fps=25))
    plt.close(fig)


def plot_benchmark(summary: dict[str, dict[str, float]], out_file: str) -> None:
    names = list(summary)
    metrics = [("success_rate", "Success rate"), ("mean_length", "Mean path length"),
               ("mean_runtime_ms", "Mean runtime (ms)"), ("mean_expanded", "Mean nodes expanded")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (key, title) in zip(axes, metrics):
        ax.bar(names, [summary[n][key] for n in names],
               color=[COLORS.get(n, "gray") for n in names])
        ax.set_title(title)
    fig.suptitle("Planner benchmark over random maps")
    fig.tight_layout()
    fig.savefig(out_file, dpi=130)
    plt.close(fig)
