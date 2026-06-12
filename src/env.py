"""
env.py — CarRacingEnv: a gymnasium.Env for the 2D car racing task.

Observation space (Box, shape (14,), dtype float32):
  [0]    d / half_width                  lateral offset, normalised
  [1]    sin(theta_e)                    heading error (sine component)
  [2]    cos(theta_e)                    heading error (cosine component)
  [3]    v / v_max                       speed, normalised
  [4-13] k_i * curvature_scale           signed curvature at 10 lookahead
                                         distances [5,10,15,20,30,40,55,70,90,110] m

Action space (Box, shape (2,), dtype float32):
  [0]  throttle/brake in [-1, 1]  → accel = a_max * action[0]
  [1]  steering      in [-1, 1]  → steer = steer_max * action[1]

Reward per step:
  + k_progress * delta_s      (k_progress = 1.0, signed arc-length progress)
  - 0.01                      (time penalty)
  off-track:  reward -= 80,  terminated = True
  full lap:   reward += 100, terminated = True

Episode terminates (terminated=True) on off-track or lap completion.
Episode is truncated (truncated=True, terminated=False) at max_steps.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Optional

from src.track import Track, HALF_WIDTH, CURVATURE_SCALE
from src.car import CarState, step as car_step, V_MAX, A_MAX, STEER_MAX, DT
from src.frenet import to_frenet, lookahead_curvatures, closest_point_index

# Lookahead distances for curvature features (metres)
LOOKAHEAD_DISTANCES = np.array([5, 10, 15, 20, 30, 40, 55, 70, 90, 110],
                                dtype=float)

# Reward parameters
K_PROGRESS: float = 1.0
TIME_PENALTY: float = 0.01
OFF_TRACK_PENALTY: float = 80.0
LAP_BONUS: float = 100.0


class CarRacingEnv(gym.Env):
    """2D car racing environment using a kinematic bicycle model.

    Parameters
    ----------
    track : Track or None
        Fixed track to use every episode.  Exactly one of *track* or
        *track_factory* must be provided.
    track_factory : callable or None
        ``factory(rng) -> Track``.  Called on every ``reset()`` to generate
        a fresh random track.  Enables curriculum / domain randomisation.
    max_steps : int
        Episode is truncated after this many steps.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        track: Optional[Track] = None,
        track_factory: Optional[Callable] = None,
        max_steps: int = 2000,
    ) -> None:
        super().__init__()

        if (track is None) == (track_factory is None):
            raise ValueError(
                "Provide exactly one of 'track' or 'track_factory', not both or neither."
            )
        self._fixed_track = track
        self._track_factory = track_factory
        self.max_steps = max_steps

        # Active track (set in reset)
        self.track: Optional[Track] = None

        # Spaces
        obs_dim = 14
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(2,), dtype=np.float32,
        )

        # State (initialised in reset)
        self._car: Optional[CarState] = None
        self._step_count: int = 0
        self._closest_idx: int = 0
        # Unwrapped arc-length progress: tracks total distance driven,
        # accumulating across the start/finish crossing.
        self._unwrapped_s: float = 0.0
        self._prev_s: float = 0.0   # s from previous step (for delta_s)

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """Reset the environment to the start of a new episode.

        Places the car at s=0 on the centreline, heading aligned with the
        track tangent at that point, speed = 0.

        Returns
        -------
        obs : ndarray, shape (14,)
        info : dict
        """
        super().reset(seed=seed)   # seeds self.np_random (gymnasium convention)

        # Regenerate track if a factory was provided
        if self._track_factory is not None:
            self.track = self._track_factory(self.np_random)
        else:
            self.track = self._fixed_track

        # Place car at index 0 of the centreline
        start_pt = self.track.points[0]
        start_tangent = self.track.tangents[0]
        start_heading = float(np.arctan2(start_tangent[1], start_tangent[0]))

        self._car = CarState(
            x=float(start_pt[0]),
            y=float(start_pt[1]),
            heading=start_heading,
            v=0.0,
        )
        self._step_count = 0
        self._closest_idx = 0
        self._prev_s = 0.0
        self._unwrapped_s = 0.0

        obs = self._build_obs()
        info = self._build_info(s=0.0, d=0.0)
        return obs, info

    def step(self, action: np.ndarray):
        """Advance the environment by one time step.

        Parameters
        ----------
        action : array-like, shape (2,)
            [throttle/brake, steering] in [-1, 1].

        Returns
        -------
        obs         : ndarray, shape (14,)
        reward      : float
        terminated  : bool
        truncated   : bool
        info        : dict
        """
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        accel = A_MAX * float(action[0])
        steer = STEER_MAX * float(action[1])

        # Advance dynamics
        self._car = car_step(self._car, accel, steer, DT)
        self._step_count += 1

        # Frenet conversion (warm-start from last known index)
        s, d, theta_e = to_frenet(
            self.track,
            np.array([self._car.x, self._car.y]),
            self._car.heading,
            start_index=self._closest_idx,
        )
        # Update the cached closest index
        self._closest_idx = closest_point_index(
            self.track,
            np.array([self._car.x, self._car.y]),
            start_index=self._closest_idx,
        )

        # ------------------------------------------------------------------
        # Lap detection via unwrapped progress
        # ------------------------------------------------------------------
        # delta_s is the arc-length advance this step.  We handle the
        # wrap-around at the start/finish line (s jumps from ~total_length
        # back to ~0) by checking which direction is smaller.
        L = self.track.total_length
        raw_delta = s - self._prev_s
        # Correct for wrap: choose the signed delta with smallest |value|
        if raw_delta > L / 2:
            raw_delta -= L
        elif raw_delta < -L / 2:
            raw_delta += L
        delta_s = raw_delta

        self._unwrapped_s += delta_s
        self._prev_s = s

        # ------------------------------------------------------------------
        # Reward
        # ------------------------------------------------------------------
        reward = K_PROGRESS * delta_s - TIME_PENALTY

        terminated = False
        truncated = False

        # Off-track
        if abs(d) > self.track.half_width:
            reward -= OFF_TRACK_PENALTY
            terminated = True

        # Lap complete
        if self._unwrapped_s >= L:
            reward += LAP_BONUS
            terminated = True

        # Truncation
        if self._step_count >= self.max_steps and not terminated:
            truncated = True

        obs = self._build_obs()
        info = self._build_info(s=s, d=d)
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        """Construct the 14-dimensional observation vector.

        obs = [d/half_width,
               sin(theta_e), cos(theta_e),
               v/v_max,
               k_1*scale, ..., k_10*scale]
        """
        car = self._car
        s, d, theta_e = to_frenet(
            self.track,
            np.array([car.x, car.y]),
            car.heading,
            start_index=self._closest_idx,
        )

        kappas = lookahead_curvatures(self.track, s, LOOKAHEAD_DISTANCES)

        obs = np.empty(14, dtype=np.float32)
        obs[0] = d / self.track.half_width
        obs[1] = np.sin(theta_e)
        obs[2] = np.cos(theta_e)
        obs[3] = car.v / V_MAX
        obs[4:14] = (kappas * CURVATURE_SCALE).astype(np.float32)
        return obs

    def _build_info(self, s: float, d: float) -> dict:
        """Build the info dict returned by reset/step."""
        return {
            "s": s,
            "d": d,
            "v": self._car.v,
            "lap_time_steps": self._step_count,
            "x": self._car.x,
            "y": self._car.y,
            "unwrapped_s": self._unwrapped_s,
        }
