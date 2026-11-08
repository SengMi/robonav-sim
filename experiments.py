"""Experiments that generate the plots and the numbers reported in the README."""
from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import opamp, rc_filter
from .dc_motor import DCMotor
from .pid import PID


def exp_rc_bode(out):
    r, c = 10e3, 100e-9
    f = np.logspace(0, 5, 400)
    fc = rc_filter.cutoff_frequency(r, c)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    a1.semilogx(f, rc_filter.gain_db(f, r, c))
    a1.axvline(fc, color="r", ls="--", label=f"fc = {fc:.0f} Hz")
    a1.axhline(-3, color="gray", ls=":")
    a1.set_ylabel("Gain [dB]"); a1.legend(); a1.grid(True, which="both", alpha=.3)
    a1.set_title("RC low-pass (R = 10 kΩ, C = 100 nF)")
    a2.semilogx(f, rc_filter.phase_deg(f, r, c))
    a2.set_ylabel("Phase [deg]"); a2.set_xlabel("Frequency [Hz]")
    a2.grid(True, which="both", alpha=.3)
    fig.tight_layout(); fig.savefig(f"{out}/rc_bode.png", dpi=130); plt.close(fig)
    return {"rc_fc_hz": fc}


def exp_opamp(out):
    rf, rg, gbw = 9e3, 1e3, 1e6
    f = np.logspace(1, 7, 400)
    g = opamp.noninverting_gain(rf, rg)
    f3 = opamp.closed_loop_bandwidth(gbw, g)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(f, 20 * np.log10(opamp.magnitude(f, rf, rg, gbw)))
    ax.axvline(f3, color="r", ls="--", label=f"-3 dB at {f3/1e3:.0f} kHz")
    ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("Gain [dB]")
    ax.set_title(f"Non-inverting amp, G = 1 + Rf/Rg = {g:.0f} (GBW 1 MHz)")
    ax.legend(); ax.grid(True, which="both", alpha=.3)
    fig.tight_layout(); fig.savefig(f"{out}/opamp_gain.png", dpi=130); plt.close(fig)
    return {"opamp_gain": g, "opamp_bw_hz": f3}


def exp_sensor_chain(out, seed=1):
    """Noisy sensor -> RC filter -> op-amp. Reports RMS error before/after."""
    rng = np.random.default_rng(seed)
    dt, T = 1e-4, 1.0
    t = np.arange(0, T, dt)
    clean = 0.2 * np.sin(2 * math.pi * 2 * t)                    # 2 Hz sensor signal
    noisy = clean + 0.05 * np.sin(2 * math.pi * 50 * t) + 0.03 * rng.standard_normal(len(t))
    r, c = 160e3, 100e-9                                         # fc ~ 10 Hz
    filt = rc_filter.filter_signal(noisy, dt, r, c)
    amp = opamp.amplify(filt, 9e3, 1e3, v_rail=5.0)
    settle = int(0.3 / dt)                                       # skip start-up transient
    rms = lambda x, ref: float(np.sqrt(np.mean((x[settle:] - ref[settle:]) ** 2)))
    err_raw, err_filt = rms(noisy, clean), rms(filt, clean)
    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    ax[0].plot(t, noisy, lw=.6); ax[0].plot(t, clean, "r", lw=1.2); ax[0].set_title("Raw sensor (noise + 50 Hz hum)")
    ax[1].plot(t, filt, lw=1); ax[1].plot(t, clean, "r--", lw=1); ax[1].set_title(f"After RC filter (fc = {rc_filter.cutoff_frequency(r, c):.1f} Hz)")
    ax[2].plot(t, amp, lw=1); ax[2].set_title("After x10 op-amp"); ax[2].set_xlabel("Time [s]")
    for a in ax: a.grid(alpha=.3); a.set_ylabel("V")
    fig.tight_layout(); fig.savefig(f"{out}/sensor_chain.png", dpi=130); plt.close(fig)
    return {"rms_error_raw_V": err_raw, "rms_error_filtered_V": err_filt}


def exp_motor_open_loop(out):
    m = DCMotor()
    dt, T = 1e-4, 0.6
    n = int(T / dt)
    w = np.zeros(n); i = np.zeros(n)
    for k in range(n):
        w[k] = m.step(12.0, dt); i[k] = m.i
    t = np.arange(n) * dt
    w_ss = DCMotor().steady_state_speed(12.0)
    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax[0].plot(t, w * 60 / (2 * math.pi)); ax[0].set_ylabel("Speed [rpm]")
    ax[0].axhline(w_ss * 60 / (2 * math.pi), color="r", ls="--", label="analytic steady state"); ax[0].legend()
    ax[0].set_title("DC motor step response, 12 V")
    ax[1].plot(t, i); ax[1].set_ylabel("Current [A]"); ax[1].set_xlabel("Time [s]")
    for a in ax: a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{out}/motor_step.png", dpi=130); plt.close(fig)
    return {"w_final_sim": float(w[-1]), "w_steady_analytic": w_ss,
            "tau_mech_s": m.mechanical_time_constant(), "peak_current_A": float(i.max())}


def exp_pid_speed(out, target=150.0):
    """Speed control with a load-torque disturbance applied at t = 1 s."""
    motor, pid = DCMotor(), PID(kp=0.3, ki=3.0, kd=0.0)
    dt, T = 1e-3, 2.0
    n = int(T / dt)
    w = np.zeros(n); u = np.zeros(n)
    for k in range(n):
        t_load = 0.03 if k * dt >= 1.0 else 0.0
        u[k] = pid.update(target, motor.w, dt)
        w[k] = motor.step(u[k], dt, t_load)
    t = np.arange(n) * dt
    overshoot = max(0.0, (w[: int(1.0 / dt)].max() - target) / target * 100)
    settle_idx = next((k for k in range(n) if np.all(np.abs(w[k: int(1.0 / dt)] - target) < 0.02 * target)), None)
    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax[0].plot(t, w, label="speed"); ax[0].axhline(target, color="r", ls="--", label="setpoint")
    ax[0].axvline(1.0, color="gray", ls=":", label="load applied"); ax[0].set_ylabel("Speed [rad/s]"); ax[0].legend()
    ax[0].set_title("PID speed control with load disturbance")
    ax[1].plot(t, u); ax[1].set_ylabel("Voltage [V]"); ax[1].set_xlabel("Time [s]")
    for a in ax: a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{out}/pid_speed.png", dpi=130); plt.close(fig)
    return {"overshoot_pct": overshoot,
            "settling_time_2pct_s": None if settle_idx is None else settle_idx * dt,
            "speed_before_load": float(w[int(1.0 / dt) - 1]),
            "speed_end": float(w[-1]), "max_voltage_V": float(u.max())}


def run_all(out="results"):
    os.makedirs(out, exist_ok=True)
    res = {}
    for fn in (exp_rc_bode, exp_opamp, exp_sensor_chain, exp_motor_open_loop, exp_pid_speed):
        res[fn.__name__] = fn(out)
    return res
