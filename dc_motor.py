"""Permanent-magnet DC motor model (electrical + mechanical).

    L di/dt = V - R i - Ke w
    J dw/dt = Kt i - b w - T_load
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MotorParams:
    R: float = 2.0        # armature resistance [ohm]
    L: float = 5e-3       # armature inductance [H]
    Kt: float = 0.05      # torque constant [N*m/A]
    Ke: float = 0.05      # back-EMF constant [V*s/rad]
    J: float = 1e-4       # rotor inertia [kg*m^2]
    b: float = 1e-5       # viscous friction [N*m*s/rad]


class DCMotor:
    def __init__(self, p: MotorParams | None = None):
        self.p = p or MotorParams()
        self.i = 0.0   # current [A]
        self.w = 0.0   # speed [rad/s]

    def _deriv(self, i, w, v, t_load):
        p = self.p
        di = (v - p.R * i - p.Ke * w) / p.L
        dw = (p.Kt * i - p.b * w - t_load) / p.J
        return di, dw

    def step(self, v: float, dt: float, t_load: float = 0.0) -> float:
        """Advance one RK4 step; returns the new speed."""
        i, w = self.i, self.w
        k1 = self._deriv(i, w, v, t_load)
        k2 = self._deriv(i + dt / 2 * k1[0], w + dt / 2 * k1[1], v, t_load)
        k3 = self._deriv(i + dt / 2 * k2[0], w + dt / 2 * k2[1], v, t_load)
        k4 = self._deriv(i + dt * k3[0], w + dt * k3[1], v, t_load)
        self.i += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        self.w += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        return self.w

    def steady_state_speed(self, v: float, t_load: float = 0.0) -> float:
        """Analytic w_ss from setting both derivatives to zero."""
        p = self.p
        return (p.Kt * v - p.R * t_load) / (p.R * p.b + p.Kt * p.Ke)

    def mechanical_time_constant(self) -> float:
        p = self.p
        return p.J * p.R / (p.R * p.b + p.Kt * p.Ke)
