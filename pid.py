"""PID controller with output saturation and anti-windup (clamping)."""
from __future__ import annotations


class PID:
    def __init__(self, kp: float, ki: float, kd: float = 0.0,
                 out_min: float = -12.0, out_max: float = 12.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.integral = 0.0
        self._prev_err = None

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        err = setpoint - measurement
        deriv = 0.0 if self._prev_err is None else (err - self._prev_err) / dt
        self._prev_err = err
        candidate = self.integral + err * dt
        u = self.kp * err + self.ki * candidate + self.kd * deriv
        if u > self.out_max:
            u = self.out_max
            if err < 0:               # only integrate if it helps unsaturate
                self.integral = candidate
        elif u < self.out_min:
            u = self.out_min
            if err > 0:
                self.integral = candidate
        else:
            self.integral = candidate
        return u
