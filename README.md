# Robot Electronics Simulation (Python)

Simulations of the analog and electromechanical building blocks of a small mobile robot:
sensor conditioning (RC filter + op-amp) and DC motor speed control (PID).
Built with NumPy and Matplotlib only.

## Modules
| File | What it models | Key equation |
|------|----------------|--------------|
| `rc_filter.py` | First-order RC low-pass, frequency and time response | f_c = 1 / (2πRC) |
| `opamp.py` | Non-inverting / inverting amplifier, finite GBW, supply clipping | G = 1 + Rf/Rg, f_-3dB = GBW / G |
| `dc_motor.py` | Permanent-magnet DC motor, RK4 integration | L di/dt = V - Ri - Ke·ω;  J dω/dt = Kt·i - bω - T_load |
| `pid.py` | PID with output saturation and anti-windup | u = Kp·e + Ki·∫e + Kd·de/dt |
| `experiments.py` | Generates all plots and numbers below | |

## Results
| Experiment | Result |
|------------|--------|
| RC filter, R = 10 kΩ, C = 100 nF | f_c = 159.2 Hz (-3 dB, -45° phase) |
| Op-amp stage, Rf = 9 kΩ, Rg = 1 kΩ, GBW = 1 MHz | Gain = 10, bandwidth = 100 kHz |
| Sensor chain (2 Hz signal + 50 Hz hum + white noise) | RMS error 46.4 mV → 28.0 mV after RC filter |
| Motor, 12 V step | Simulated 238.0 rad/s vs analytic 238.1 rad/s; τ_mech = 79 ms; start-up current 5.5 A |
| PID speed control (150 rad/s) | 0% overshoot, 2% settling in 0.26 s, load step rejected (back to 150 rad/s) |

Plots are in `results/`: `rc_bode.png`, `opamp_gain.png`, `sensor_chain.png`,
`motor_step.png`, `pid_speed.png`.

## Run
```bash
pip install -r requirements.txt
python main.py
python -m unittest discover -s tests -v
```
12 unit tests check the simulation against analytic results (cutoff frequency, -3 dB point,
63% rise at τ, motor steady state, back-EMF balance, saturation, load rejection).

## Notes and limitations
Parameters are typical small-motor values, not measured from real hardware. The model
ignores sensor non-linearity, op-amp offsets, PWM switching, and temperature effects.
