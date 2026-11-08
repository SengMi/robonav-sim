import math
import unittest

import numpy as np

from circuitsim import opamp, rc_filter
from circuitsim.dc_motor import DCMotor
from circuitsim.pid import PID


class TestRC(unittest.TestCase):
    def test_cutoff(self):
        self.assertAlmostEqual(rc_filter.cutoff_frequency(10e3, 100e-9), 159.155, places=2)

    def test_minus_3db_at_cutoff(self):
        fc = rc_filter.cutoff_frequency(10e3, 100e-9)
        self.assertAlmostEqual(float(rc_filter.gain_db(fc, 10e3, 100e-9)), -3.0103, places=3)

    def test_phase_minus_45_at_cutoff(self):
        fc = rc_filter.cutoff_frequency(10e3, 100e-9)
        self.assertAlmostEqual(float(rc_filter.phase_deg(fc, 10e3, 100e-9)), -45.0, places=6)

    def test_step_reaches_63_percent_at_tau(self):
        r, c, dt = 10e3, 100e-9, 1e-5
        x = np.ones(int(5 * r * c / dt))
        x[0] = 0.0
        y = rc_filter.filter_signal(x, dt, r, c)
        k = int(r * c / dt)
        self.assertAlmostEqual(y[k], 1 - math.exp(-1), delta=0.01)


class TestOpAmp(unittest.TestCase):
    def test_gain(self):
        self.assertEqual(opamp.noninverting_gain(9e3, 1e3), 10.0)
        self.assertEqual(opamp.inverting_gain(10e3, 1e3), -10.0)

    def test_bandwidth(self):
        self.assertAlmostEqual(opamp.closed_loop_bandwidth(1e6, 10), 1e5)

    def test_clipping(self):
        out = opamp.amplify([1.0], 9e3, 1e3, v_rail=5.0)
        self.assertEqual(float(out[0]), 5.0)


class TestMotor(unittest.TestCase):
    def test_converges_to_analytic_steady_state(self):
        m = DCMotor()
        for _ in range(int(1.5 / 1e-4)):
            m.step(12.0, 1e-4)
        self.assertAlmostEqual(m.w, m.steady_state_speed(12.0), delta=0.5)

    def test_load_reduces_speed(self):
        self.assertLess(DCMotor().steady_state_speed(12.0, 0.03),
                        DCMotor().steady_state_speed(12.0, 0.0))

    def test_current_matches_back_emf_balance(self):
        m = DCMotor()
        for _ in range(int(1.5 / 1e-4)):
            m.step(12.0, 1e-4)
        self.assertAlmostEqual(m.i, (12.0 - m.p.Ke * m.w) / m.p.R, delta=1e-2)


class TestPID(unittest.TestCase):
    def test_output_saturates(self):
        pid = PID(10, 0, 0, out_min=-12, out_max=12)
        self.assertEqual(pid.update(1000, 0, 0.01), 12)

    def test_tracks_setpoint_and_rejects_load(self):
        m, pid, dt = DCMotor(), PID(0.3, 3.0), 1e-3
        for k in range(2000):
            tl = 0.03 if k >= 1000 else 0.0
            m.step(pid.update(150.0, m.w, dt), dt, tl)
        self.assertAlmostEqual(m.w, 150.0, delta=1.5)


if __name__ == "__main__":
    unittest.main()
