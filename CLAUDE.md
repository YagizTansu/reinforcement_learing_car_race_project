# Project: Car Race — Deep RL with Policy Gradients (PG-4)

You are helping me build a university RL project. Follow this spec EXACTLY.
Do not add features I did not ask for. Do not skip validation steps.
Prefer simple, readable code over clever code. I must be able to explain
every line in an oral exam.

## Goal
Train an agent to drive a car around a 2D racetrack in minimum time using
PPO with Gaussian policies, and study how policy network size affects:
(1) final performance, (2) episodes to convergence, (3) wall-clock time.

## Tech stack (fixed — do not substitute)
- Python 3.11+, numpy, matplotlib
- gymnasium (custom Env subclass)
- stable-baselines3 for PPO
- torch (comes with SB3)
- No other heavy dependencies without asking me first.

## Architecture (fixed)
src/
  track.py        # Track generation: spline centerline, arc-length resampling,
                  # curvature, tangent; constant half-width
  car.py          # Kinematic bicycle model dynamics
  frenet.py       # Cartesian -> track-relative (Frenet) conversion
  env.py          # gymnasium.Env: CarRacingEnv
  render.py       # matplotlib rendering: track + car trajectory
  manual_drive.py # keyboard / scripted-controller validation harness
  train.py        # training entry point (CLI args: net size, seed, track mode)
  evaluate.py     # deterministic evaluation of saved models
  plots.py        # aggregate results across seeds, produce report figures
experiments/      # configs + saved runs (auto-created)
tests/            # pytest unit tests

## Core design decisions (fixed — never change these silently)
1. STATE (observation), all normalized to roughly [-1, 1]:
   [ d / half_width,                       # lateral offset
     sin(theta_e), cos(theta_e),           # heading error vs track tangent
     v / v_max,                            # speed
     k_1 ... k_10 ]                        # signed curvature at lookahead
                                           # distances [5,10,15,20,30,40,55,70,90,110] m,
                                           # each multiplied by curvature_scale
   => observation space: Box, shape (14,)
   NO absolute position, NO absolute progress s in the state.
2. ACTION: Box([-1,1]^2): [throttle/brake, steering].
   Env maps to physical units: accel = a_max * action[0], steer = steer_max * action[1].
3. DYNAMICS: kinematic bicycle model, dt = 0.05 s,
   v clipped to [0, v_max], v_max = 30 m/s, a_max = 8 m/s^2,
   steer_max = 0.45 rad, wheelbase L = 3.0 m.
4. REWARD per step:
   + k_progress * delta_s   (signed arc-length progress; k_progress = 1.0)
   - 0.01                   (small time penalty)
   off-track (|d| > half_width): reward -= 80, terminated = True
   full lap completed: reward += 100, terminated = True
5. EPISODE: truncated at 2000 steps (set truncated=True, NOT terminated).
   Respect gymnasium's terminated vs truncated distinction everywhere.
6. TRACK: closed loop from 8-12 control points -> periodic cubic spline ->
   resampled at uniform 2 m arc-length spacing. half_width = 6 m.
   Precompute per point: cumulative arc length, unit tangent, signed curvature.
7. EXPERIMENTS: policy net sizes [16], [64,64], [256,256], [512,512,512];
   seeds [0,1,2,3]; identical hyperparameters across all sizes (SB3 PPO
   defaults unless I say otherwise). Log per-episode return, episode length,
   policy std, and wall-clock time to CSV per run.

## Working rules
- After EVERY phase, stop and tell me how to validate it manually.
- Write pytest tests for geometry-heavy code (frenet.py, track.py).
- Every plot must support mean ± std across seeds.
- Keep functions short; docstrings explain the math, citing the formula used.