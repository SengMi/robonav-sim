"""Non-inverting op-amp amplifier with finite gain-bandwidth and supply rails."""
from __future__ import annotations

import numpy as np


def noninverting_gain(rf: float, rg: float) -> float:
    """G = 1 + Rf/Rg"""
    return 1.0 + rf / rg


def inverting_gain(rf: float, rin: float) -> float:
    """G = -Rf/Rin"""
    return -rf / rin


def closed_loop_bandwidth(gbw_hz: float, g: float) -> float:
    """Gain-bandwidth product is constant: f_-3dB = GBW / |G|."""
    return gbw_hz / abs(g)


def magnitude(freq_hz, rf: float, rg: float, gbw_hz: float):
    g = noninverting_gain(rf, rg)
    f3 = closed_loop_bandwidth(gbw_hz, g)
    f = np.asarray(freq_hz, dtype=float)
    return g / np.sqrt(1.0 + (f / f3) ** 2)


def amplify(v_in, rf: float, rg: float, v_rail: float = 5.0):
    """Ideal gain followed by output clipping at +/- supply rail."""
    return np.clip(noninverting_gain(rf, rg) * np.asarray(v_in, dtype=float),
                   -v_rail, v_rail)
