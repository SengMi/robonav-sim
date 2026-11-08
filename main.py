
from __future__ import annotations

import argparse
import os

from robonav.benchmark import format_summary, run_benchmark
from robonav.environment import GridMap
from robonav.planners import PLANNERS, astar, dijkstra, rrt
from robonav.simulation import simulate
from robonav.smoothing import shortcut
from robonav.visualize import animate, plot_benchmark, plot_plans


def cmd_demo(args) -> None:
    os.makedirs(args.out, exist_ok=True)
    w, h = 60, 40
    start, goal = (4, 4), (w - 5, h - 5)
    raw = GridMap.random(w, h, seed=args.seed, keep_clear=[start, goal])
    gmap = raw.inflate(args.radius)

    results = [dijkstra(gmap, start, goal), astar(gmap, start, goal), rrt(gmap, start, goal)]
    for r in results:
        status = f"length={r.length:.1f}" if r.success else "NO PATH"
        print(f"{r.name:9s} {status:16s} expanded={r.expanded:6d} time={r.runtime_s*1000:7.1f} ms")

    smoothed = {r.name: shortcut(r.path, gmap) for r in results if r.success}
    plot_plans(gmap, start, goal, results, f"{args.out}/plans.png", smoothed)

    chosen = next(r for r in results if r.name.lower().replace("*", "star") == args.planner
                  or r.name.lower() == args.planner)
    if not chosen.success:
        print("Chosen planner found no path; try another --seed.")
        return
    path = smoothed[chosen.name]
    sim = simulate(raw, path)       # collisions checked on the TRUE map
    print(f"Simulation: reached={sim.reached_goal} collided={sim.collided} "
          f"time={sim.time_s:.1f}s distance={sim.distance_travelled:.1f}")
    animate(raw, path, sim, f"{args.out}/robot.gif")
    print(f"Saved plans.png and robot.gif in {args.out}/")


def cmd_benchmark(args) -> None:
    os.makedirs(args.out, exist_ok=True)
    rows, summary = run_benchmark(args.maps, csv_file=f"{args.out}/benchmark.csv")
    print(format_summary(summary))
    plot_benchmark(summary, f"{args.out}/benchmark.png")
    print(f"Saved benchmark.csv and benchmark.png in {args.out}/")


def main() -> None:
    p = argparse.ArgumentParser(description="RoboNav-Sim")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="plan + simulate on one random map")
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--radius", type=int, default=1, help="robot radius in cells")
    d.add_argument("--planner", choices=["dijkstra", "astar", "rrt"], default="astar")
    d.add_argument("--out", default="results")
    d.set_defaults(func=cmd_demo)
    b = sub.add_parser("benchmark", help="compare planners on many random maps")
    b.add_argument("--maps", type=int, default=30)
    b.add_argument("--out", default="results")
    b.set_defaults(func=cmd_benchmark)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
