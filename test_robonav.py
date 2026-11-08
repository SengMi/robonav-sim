import math
import unittest

import numpy as np

from robonav.environment import GridMap
from robonav.planners import astar, dijkstra, path_length, rrt
from robonav.simulation import simulate
from robonav.smoothing import densify, shortcut


class TestPlanners(unittest.TestCase):
    def test_astar_straight_line_optimal(self):
        m = GridMap.empty(20, 20)
        r = astar(m, (0, 0), (10, 0))
        self.assertTrue(r.success)
        self.assertAlmostEqual(r.length, 10.0)

    def test_diagonal_cost(self):
        r = astar(GridMap.empty(20, 20), (0, 0), (5, 5))
        self.assertAlmostEqual(r.length, 5 * math.sqrt(2))

    def test_no_path_when_walled_off(self):
        m = GridMap.empty(20, 20)
        m.grid[:, 10] = 1
        self.assertFalse(astar(m, (0, 0), (19, 19)).success)
        self.assertFalse(dijkstra(m, (0, 0), (19, 19)).success)

    def test_astar_matches_dijkstra_cost_but_expands_fewer(self):
        m = GridMap.random(50, 40, seed=3, keep_clear=[(3, 3), (46, 36)])
        a, d = astar(m, (3, 3), (46, 36)), dijkstra(m, (3, 3), (46, 36))
        self.assertEqual(a.success, d.success)
        if a.success:
            self.assertAlmostEqual(a.length, d.length, places=6)
            self.assertLessEqual(a.expanded, d.expanded)

    def test_start_in_obstacle(self):
        m = GridMap.empty(10, 10)
        m.grid[0, 0] = 1
        self.assertFalse(astar(m, (0, 0), (5, 5)).success)

    def test_rrt_path_is_collision_free(self):
        m = GridMap.empty(40, 30)
        m.grid[5:25, 15:18] = 1
        r = rrt(m, (2, 2), (37, 27), seed=1)
        self.assertTrue(r.success)
        for a, b in zip(r.path, r.path[1:]):
            self.assertTrue(m.segment_is_free(a, b))


class TestSmoothingAndSim(unittest.TestCase):
    def test_shortcut_never_longer_and_still_free(self):
        m = GridMap.random(50, 40, seed=5, keep_clear=[(3, 3), (46, 36)])
        r = astar(m, (3, 3), (46, 36))
        if r.success:
            s = shortcut(r.path, m)
            self.assertLessEqual(path_length(s), r.length + 1e-9)
            for a, b in zip(s, s[1:]):
                self.assertTrue(m.segment_is_free(a, b))

    def test_densify_spacing(self):
        d = densify([(0, 0), (10, 0)], 0.5)
        self.assertTrue(all(math.dist(a, b) <= 0.5 + 1e-9 for a, b in zip(d, d[1:])))

    def test_inflate_grows_obstacles(self):
        m = GridMap.empty(11, 11)
        m.grid[5, 5] = 1
        inf = m.inflate(2)
        self.assertEqual(inf.grid[5, 7], 1)
        self.assertEqual(inf.grid[5, 8], 0)

    def test_robot_reaches_goal_on_empty_map(self):
        m = GridMap.empty(30, 30)
        res = simulate(m, [(2.0, 2.0), (25.0, 2.0), (25.0, 25.0)])
        self.assertTrue(res.reached_goal)
        self.assertFalse(res.collided)


if __name__ == "__main__":
    unittest.main()
