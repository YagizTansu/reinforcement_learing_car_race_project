"""
manual_drive.py — Validation harness for CarRacingEnv.

Mode 1 (scripted, default):
  A proportional controller drives the car for one episode.
  Controller law:
      steer_norm = clip(-c1 * d_norm - c2 * theta_e, -1, 1)
      throttle   = fixed constant
  where d_norm = d / half_width (already in obs[0]) and theta_e is recovered
  from obs via atan2(sin, cos) = atan2(obs[1], obs[2]).

  Prints total reward, number of steps, and completion status.
  Saves a trajectory plot to experiments/figures/scripted_drive.png.

Mode 2 (keyboard):
  Interactive matplotlib window; arrow keys control the car.
  Only available when a display is present.

Usage
-----
  python -m src.manual_drive            # scripted mode
  python -m src.manual_drive --keyboard # keyboard mode
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")   # default; overridden to TkAgg in keyboard mode
import matplotlib.pyplot as plt

from src.track import track as default_track
from src.env import CarRacingEnv
from src.render import plot_track

# P-controller gains
C1: float = 1.2   # lateral offset gain
C2: float = 1.8   # heading error gain
THROTTLE: float = 0.4   # fixed throttle (normalised, maps to 0.4 * a_max)


# ---------------------------------------------------------------------------
# Scripted controller
# ---------------------------------------------------------------------------

def run_scripted(save_path: str = "experiments/figures/scripted_drive.png") -> None:
    """Run one episode with the P-controller and save the trajectory plot.

    The P-controller uses only obs[0] (d_norm) and obs[1:3] (sin/cos theta_e).
    It ignores curvature lookahead — this is intentional for simplicity.

    Controller:
        theta_e   = atan2(obs[1], obs[2])
        steer_cmd = clip(-C1 * d_norm - C2 * theta_e, -1, 1)
        throttle  = THROTTLE  (constant)

    Positive d_norm → car is left of centre → steer right (negative steer cmd).
    Positive theta_e → car heading is CCW of tangent → steer right.
    """
    track = default_track()

    terminated = truncated = False
    while not (terminated or truncated):
        d_norm  = float(obs[0])
        theta_e = float(np.arctan2(obs[1], obs[2]))

        steer_cmd = float(np.clip(-C1 * d_norm - C2 * theta_e, -1.0, 1.0))
        action = np.array([THROTTLE, steer_cmd], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        xs.append(info["x"])
        ys.append(info["y"])

    steps = info["lap_time_steps"]
    if terminated and info["unwrapped_s"] >= track.total_length:
        status = "LAP COMPLETED"
    elif terminated:
        status = "OFF-TRACK"
    else:
        status = "TRUNCATED (timeout)"

    print(f"Status        : {status}")
    print(f"Steps         : {steps}")
    print(f"Total reward  : {total_reward:.2f}")
    print(f"Final speed   : {info['v']:.2f} m/s")
    print(f"Final d       : {info['d']:.2f} m")

    # --- Plot ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_track(track, ax=ax, title="Scripted P-controller trajectory")

    # Colour the trajectory by time (blue = start, red = end)
    n = len(xs)
    colors = plt.cm.plasma(np.linspace(0.0, 1.0, n))
    ax.scatter(xs, ys, c=colors, s=3, zorder=4, linewidths=0)

    # Start / end markers
    ax.plot(xs[0], ys[0], "g^", markersize=9, zorder=5, label="Start")
    ax.plot(xs[-1], ys[-1], "rs", markersize=9, zorder=5, label=f"End ({status})")

    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Trajectory saved: {save_path}")


# ---------------------------------------------------------------------------
# Keyboard controller
# ---------------------------------------------------------------------------

def run_keyboard() -> None:
    """Interactive keyboard-controlled episode.

    Arrow keys:
      UP    — throttle +0.1
      DOWN  — brake (throttle -0.1)
      LEFT  — steer left  (-0.1)
      RIGHT — steer right (+0.1)
      Q     — quit

    Requires a display (e.g. won't work over SSH without X forwarding).
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    track = default_track()
    env = CarRacingEnv(track=track)
    obs, _ = env.reset(seed=0)

    throttle = [0.0]
    steer = [0.0]
    done = [False]

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_track(track, ax=ax, title="Keyboard drive — arrow keys, Q to quit")
    car_dot, = ax.plot([], [], "ro", markersize=8, zorder=5)
    traj_line, = ax.plot([], [], "b-", linewidth=1, alpha=0.6, zorder=4)
    xs, ys = [], []

    def on_key(event):
        if event.key == "up":
            throttle[0] = min(1.0, throttle[0] + 0.1)
        elif event.key == "down":
            throttle[0] = max(-1.0, throttle[0] - 0.1)
        elif event.key == "left":
            steer[0] = max(-1.0, steer[0] - 0.1)
        elif event.key == "right":
            steer[0] = min(1.0, steer[0] + 0.1)
        elif event.key == "q":
            done[0] = True
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    total_reward = 0.0

    def update(_):
        if done[0]:
            return
        action = np.array([throttle[0], steer[0]], dtype=np.float32)
        nonlocal obs
        obs, reward, terminated, truncated, info = env.step(action)
        nonlocal total_reward
        total_reward += reward
        xs.append(info["x"])
        ys.append(info["y"])
        car_dot.set_data([info["x"]], [info["y"]])
        traj_line.set_data(xs, ys)
        fig.canvas.draw_idle()
        if terminated or truncated:
            done[0] = True
            print(f"Episode over — reward={total_reward:.2f}, steps={info['lap_time_steps']}")

    from matplotlib.animation import FuncAnimation
    _anim = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual drive harness")
    parser.add_argument("--keyboard", action="store_true",
                        help="Enable interactive keyboard mode")
    args = parser.parse_args()

    if args.keyboard:
        run_keyboard()
    else:
        run_scripted()
