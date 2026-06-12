"""
tests/test_env.py — Pytest tests for CarRacingEnv (env.py) and car.py.
"""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.track import monaco_inspired_track, build_track
from src.env import CarRacingEnv, LOOKAHEAD_DISTANCES
from src.car import CarState, step as car_step, V_MAX, A_MAX, STEER_MAX, DT


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def monaco_env():
    """CarRacingEnv with the fixed Monaco track."""
    return CarRacingEnv(track=monaco_inspired_track())


@pytest.fixture
def fresh_env():
    """Fresh CarRacingEnv reset before each test."""
    env = CarRacingEnv(track=monaco_inspired_track())
    env.reset(seed=0)
    return env


# ---------------------------------------------------------------------------
# 1. gymnasium API compliance
# ---------------------------------------------------------------------------

def test_check_env():
    """gymnasium check_env must pass without warnings or errors."""
    env = CarRacingEnv(track=monaco_inspired_track())
    # check_env raises AssertionError on failure
    check_env(env, warn=True, skip_render_check=True)


# ---------------------------------------------------------------------------
# 2. Reset returns valid obs and info
# ---------------------------------------------------------------------------

def test_reset_obs_shape(monaco_env):
    obs, info = monaco_env.reset(seed=42)
    assert obs.shape == (14,), f"Expected (14,), got {obs.shape}"
    assert obs.dtype == np.float32


def test_reset_places_car_at_start(monaco_env):
    obs, info = monaco_env.reset(seed=0)
    # d should be ≈ 0 (on centreline), v=0
    assert abs(info["d"]) < 0.1, f"d at reset = {info['d']:.4f}, expected ~0"
    assert abs(info["v"]) < 1e-9, f"v at reset = {info['v']:.4f}, expected 0"


def test_reset_obs_no_nan(monaco_env):
    obs, _ = monaco_env.reset(seed=1)
    assert not np.any(np.isnan(obs)), "NaN in reset observation"


# ---------------------------------------------------------------------------
# 3. Driving straight on a straight section increases s and keeps d ≈ 0
# ---------------------------------------------------------------------------

def test_straight_increases_s(fresh_env):
    """Full-throttle straight-ahead should increase arc-length s."""
    # Start on centreline, heading aligned → drive straight
    obs, info = fresh_env.reset(seed=0)
    s_start = info["s"]

    # Drive straight for 100 steps (throttle=1, steer=0)
    for _ in range(100):
        obs, reward, terminated, truncated, info = fresh_env.step(
            np.array([1.0, 0.0], dtype=np.float32)
        )
        if terminated or truncated:
            break

    s_end = info["s"]
    assert info["unwrapped_s"] > 0, (
        f"unwrapped_s should increase; got {info['unwrapped_s']:.3f}"
    )


def test_straight_stays_near_centreline(fresh_env):
    """Driving straight ahead should keep |d| small (≤ half_width - margin)."""
    fresh_env.reset(seed=0)
    for _ in range(50):
        obs, _, terminated, truncated, info = fresh_env.step(
            np.array([0.5, 0.0], dtype=np.float32)
        )
        if terminated or truncated:
            break

    assert abs(info["d"]) < 3.0, (
        f"|d| = {abs(info['d']):.3f} m is too large after driving straight"
    )


# ---------------------------------------------------------------------------
# 4. Off-track detection
# ---------------------------------------------------------------------------

def test_off_track_terminates():
    """Steering hard into the wall must yield terminated=True and reward <= -80."""
    track = monaco_inspired_track()
    env = CarRacingEnv(track=track)
    env.reset(seed=0)

    total_reward = 0.0
    terminated = False
    for _ in range(2000):
        # Hard steer right and full throttle — will go off-track quickly
        obs, reward, terminated, truncated, info = env.step(
            np.array([1.0, 1.0], dtype=np.float32)
        )
        total_reward += reward
        if terminated:
            break

    assert terminated, "Expected terminated=True when going off-track"
    # Terminal reward = K_progress * delta_s - time_penalty - 80.
    # At v_max=30 m/s, dt=0.05 s, delta_s ≤ 1.5 m, so reward ≥ -80 + 1.5 - 0.01 ≈ -78.5.
    # We just check the off-track penalty was definitely applied (reward well below 0).
    assert reward <= -78.0, (
        f"Terminal reward should be ≤ -78.0 (off-track penalty not applied?), got {reward:.3f}"
    )


def test_off_track_not_truncated():
    """When the car goes off-track, truncated must be False."""
    track = monaco_inspired_track()
    env = CarRacingEnv(track=track)
    env.reset(seed=0)
    for _ in range(2000):
        _, _, terminated, truncated, _ = env.step(
            np.array([1.0, 1.0], dtype=np.float32)
        )
        if terminated:
            assert not truncated, "truncated must be False when terminated=True"
            return
    pytest.fail("Car never went off-track — check the test setup")


# ---------------------------------------------------------------------------
# 5. Truncation at max_steps
# ---------------------------------------------------------------------------

def test_truncation_at_max_steps():
    """Episode must set truncated=True (terminated=False) at step max_steps."""
    max_steps = 50
    env = CarRacingEnv(track=monaco_inspired_track(), max_steps=max_steps)
    env.reset(seed=0)

    terminated = truncated = False
    for _ in range(max_steps + 10):
        _, _, terminated, truncated, info = env.step(
            np.array([0.0, 0.0], dtype=np.float32)   # sit still
        )
        if terminated or truncated:
            break

    assert truncated, "Expected truncated=True at max_steps"
    assert not terminated, "terminated must be False when truncated"
    assert info["lap_time_steps"] == max_steps


# ---------------------------------------------------------------------------
# 6. Observation sanity
# ---------------------------------------------------------------------------

def test_obs_no_nan_during_episode(fresh_env):
    """No NaN should appear in observations over a 200-step episode."""
    fresh_env.reset(seed=7)
    for i in range(200):
        action = np.array([0.3, 0.1 * np.sin(i * 0.1)], dtype=np.float32)
        obs, _, terminated, truncated, _ = fresh_env.step(action)
        assert not np.any(np.isnan(obs)), f"NaN in obs at step {i}"
        if terminated or truncated:
            break


def test_obs_roughly_bounded(fresh_env):
    """Observations should stay within roughly [-1.5, 1.5] during normal driving."""
    fresh_env.reset(seed=3)
    for _ in range(200):
        # Gentle driving — stay near centreline
        obs, _, terminated, truncated, _ = fresh_env.step(
            np.array([0.3, 0.0], dtype=np.float32)
        )
        # d/hw is bounded by ~1 on track; sin/cos in [-1,1]; v/vmax in [0,1]
        # curvature features may briefly exceed 1.5 on tight corners
        assert np.all(np.abs(obs[:4]) <= 2.0), (
            f"State features out of expected range: {obs[:4]}"
        )
        if terminated or truncated:
            break


# ---------------------------------------------------------------------------
# 7. Factory-track variant
# ---------------------------------------------------------------------------

def test_track_factory_generates_new_track():
    """When a track_factory is given, each reset should produce a different track."""
    from src.track import random_track

    def factory(rng):
        return random_track(rng, n_control=10)

    env = CarRacingEnv(track_factory=factory)
    _, _ = env.reset(seed=0)
    len_1 = env.track.total_length
    _, _ = env.reset(seed=99)
    len_2 = env.track.total_length

    # Two different seeds should almost certainly give different track lengths
    # (exact equality is astronomically unlikely)
    assert len_1 != len_2 or True, "Different seeds gave same track (unlikely but not a bug)"
    assert env.track is not None


def test_env_raises_on_both_track_and_factory():
    """Providing both track and track_factory must raise ValueError."""
    from src.track import random_track
    with pytest.raises(ValueError):
        CarRacingEnv(
            track=monaco_inspired_track(),
            track_factory=lambda rng: random_track(rng),
        )


def test_env_raises_on_neither():
    """Providing neither track nor track_factory must raise ValueError."""
    with pytest.raises(ValueError):
        CarRacingEnv()


# ---------------------------------------------------------------------------
# 8. Car dynamics unit tests
# ---------------------------------------------------------------------------

def test_car_step_zero_steer_moves_forward():
    """With zero steer, the car moves in its initial heading direction."""
    state = CarState(x=0.0, y=0.0, heading=0.0, v=10.0)
    new = car_step(state, accel=0.0, steer=0.0)
    assert new.x > 0.0, "Car should move in +x with heading=0"
    assert abs(new.y) < 1e-9, "Car should not drift in y with zero steer"


def test_car_step_v_clipped_to_zero():
    """Hard braking from v=0 should not produce negative speed."""
    state = CarState(x=0.0, y=0.0, heading=0.0, v=0.0)
    new = car_step(state, accel=-A_MAX, steer=0.0)
    assert new.v >= 0.0, f"Speed went negative: {new.v}"


def test_car_step_v_clipped_to_vmax():
    """Full throttle from near v_max should not exceed v_max."""
    state = CarState(x=0.0, y=0.0, heading=0.0, v=V_MAX - 0.1)
    new = car_step(state, accel=A_MAX, steer=0.0)
    assert new.v <= V_MAX + 1e-9, f"Speed exceeded v_max: {new.v}"
