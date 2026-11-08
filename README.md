# RoboNav-Sim: 2D Autonomous Mobile Robot Navigation Simulator

A pure-Python (NumPy + Matplotlib) simulator that plans collision-free paths for a
differential-drive robot and drives it along them in a closed control loop.

## Features
- **Occupancy-grid environment** with random map generation and obstacle inflation
  (configuration-space expansion for a circular robot).
- **Global planners**: Dijkstra, A* (octile heuristic, no diagonal corner-cutting), and RRT.
- **Path post-processing**: line-of-sight shortcut smoothing and path densification.
- **Robot model**: differential-drive kinematics with velocity limits.
- **Controller**: pure-pursuit path tracking with speed scaling for turns and goal approach.
- **Safety check**: collisions are tested on the *true* map, not the inflated planning map.
- **Benchmark**: Monte-Carlo comparison over random maps (success rate, length, runtime,
  nodes expanded) exported to CSV and plots.
- **Unit tests** covering optimality, unreachable goals, collision-free paths, and the controller.

## Quick start
```bash
pip install -r requirements.txt
python main.py demo --seed 7 --planner astar     # plans.png + robot.gif in results/
python main.py benchmark --maps 30               # benchmark.csv + benchmark.png
python -m unittest discover -s tests -v          # run tests
```

## Sample results (30 random 60x40 maps)
| Planner  | Mean length | Smoothed | Runtime (ms) | Nodes expanded |
|----------|-------------|----------|--------------|----------------|
| A*       | 69.4        | 66.2     | 3.8          | 370            |
| Dijkstra | 69.4        | 66.4     | 10.6         | 1117           |
| RRT      | 82.3        | 69.2     | 100.0        | 1459           |

Success rate is below 100% because some random maps have no valid path after inflation.
A* returns the same optimal cost as Dijkstra while expanding about 3x fewer nodes.
RRT is not optimal, but smoothing recovers most of the length gap.

## Project layout
```
robonav/environment.py   grid map, inflation, random maps
robonav/planners.py      Dijkstra, A*, RRT
robonav/smoothing.py     shortcut smoothing, densify
robonav/robot.py         kinematics + pure pursuit
robonav/simulation.py    closed-loop simulation
robonav/benchmark.py     Monte-Carlo evaluation
robonav/visualize.py     plots and GIF
main.py                  CLI
tests/                   unit tests
```

## Future work
RRT*, dynamic obstacles with D* Lite replanning, LiDAR simulation, ROS 2 bridge.
