"""
car.py — Kinematic bicycle model dynamics.

Model equations (discrete-time, forward Euler):
  beta = arctan(tan(steer) * l_r / L)       # slip angle at CoM
  x'   = x + v * cos(heading + beta) * dt
  y'   = y + v * sin(heading + beta) * dt
  heading' = heading + v * sin(beta) / l_r * dt
  v'   = clip(v + accel * dt, 0, v_max)

where L = wheelbase, l_r = L/2 (rear axle to CoM, assumed half-wheelbase).

Reference: Kong et al., "Kinematic and Dynamic Vehicle Models for
Autonomous Driving Control Design", IV 2015.
"""

import numpy as np
from dataclasses import dataclass

# Physical constants (fixed by spec)
V_MAX: float = 30.0    # m/s
A_MAX: float = 8.0     # m/s²
STEER_MAX: float = 0.45  # rad
WHEELBASE: float = 3.0   # metres, L
DT: float = 0.05         # seconds, simulation time step


@dataclass
class CarState:
    """Full kinematic state of the car.

    Attributes
    ----------
    x : float  — Cartesian x position (metres)
    y : float  — Cartesian y position (metres)
    heading : float  — yaw angle (radians, CCW from +x axis)
    v : float  — longitudinal speed (m/s), always in [0, v_max]
    """
    x: float
    y: float
    heading: float
    v: float


def step(state: CarState, accel: float, steer: float,
         dt: float = DT) -> CarState:
    """Advance the kinematic bicycle model by one time step dt.

    The model uses the rear-axle as the reference point with the CoM
    assumed at the geometric centre (l_r = L/2).

    Equations
    ---------
    beta  = arctan( tan(steer) * l_r / L )
    x'    = x + v * cos(heading + beta) * dt
    y'    = y + v * sin(heading + beta) * dt
    psi'  = heading + (v / l_r) * sin(beta) * dt
    v'    = clip(v + accel * dt, 0, v_max)

    Parameters
    ----------
    state : CarState
        Current state.
    accel : float
        Longitudinal acceleration in m/s².  Positive = accelerate,
        negative = brake.
    steer : float
        Front-wheel steering angle in radians.
    dt : float
        Time step in seconds.

    Returns
    -------
    CarState
        Next state after one dt step.
    """
    l_r = WHEELBASE / 2.0   # rear axle to CoM (assumed at midpoint)

    # Slip angle at the centre of mass
    beta = np.arctan(np.tan(steer) * l_r / WHEELBASE)

    x_new = state.x + state.v * np.cos(state.heading + beta) * dt
    y_new = state.y + state.v * np.sin(state.heading + beta) * dt
    heading_new = state.heading + (state.v / l_r) * np.sin(beta) * dt
    v_new = float(np.clip(state.v + accel * dt, 0.0, V_MAX))

    return CarState(x=x_new, y=y_new, heading=heading_new, v=v_new)
