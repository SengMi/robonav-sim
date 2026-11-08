"""First-order RC low-pass filter (sensor noise filtering)."""
from __future__ import annotations

import math

import numpy as np


def cutoff_frequency(r_ohm: float, c_farad: float) -> float:
    """f_c = 1 / (2*pi*R*C)  [Hz]"""
    return 1.0 / (2.0 * math.pi * r_ohm * c_farad)


def gain(freq_hz, r_ohm: float, c_farad: float):
    """|H(jw)| = 1 / sqrt(1 + (f/fc)^2)"""
    fc = cutoff_frequency(r_ohm, c_farad)
    f = np.asarray(freq_hz, dtype=float)
    return 1.0 / np.sqrt(1.0 + (f / fc) ** 2)


def gain_db(freq_hz, r_ohm: float, c_farad: float):
    return 20.0 * np.log10(gain(freq_hz, r_ohm, c_farad))


def phase_deg(freq_hz, r_ohm: float, c_farad: float):
    fc = cutoff_frequency(r_ohm, c_farad)
    return -np.degrees(np.arctan(np.asarray(freq_hz, dtype=float) / fc))


def filter_signal(x, dt: float, r_ohm: float, c_farad: float):
    """Time-domain simulation: dVout/dt = (Vin - Vout) / (R*C), discretised
    exactly for a zero-order-hold input."""
    tau = r_ohm * c_farad
    a = 1.0 - math.exp(-dt / tau)
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for k in range(1, len(x)):
        y[k] = y[k - 1] + a * (x[k] - y[k - 1])
    return y
