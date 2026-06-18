"""
tests/test_geometry.py — Pytest tests for track.py and frenet.py.

All geometric tests use a perfect circle track because its analytic
properties are known exactly:
  - Uniform curvature κ = 1/R everywhere.
  - Total arc length = 2πR.
  - A point placed 2 m to the left of the centreline has d = +2.
  - Heading error wraps correctly at the ±π boundary.
"""

import numpy as np
import pytest
from src.track import build_track, RESAMPLE_DS, HALF_WIDTH
from src.frenet import (
    closest_point_index,
    to_frenet,
    lookahead_curvatures,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_circle_track(radius: float = 50.0, n_ctrl: int = 32) -> "Track":  # noqa
    """Build a track whose centreline is a circle of given radius.

    Control points on a perfect circle with n_ctrl equally-spaced points.
    The spline will reproduce the circle closely (error < ds).
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_ctrl, endpoint=False)
    ctrl_pts = np.column_stack([radius * np.cos(angles),
                                radius * np.sin(angles)])
    return build_track(ctrl_pts)


# ---------------------------------------------------------------------------
# Track geometry tests
# ---------------------------------------------------------------------------

class TestCircleTrack:
    """Tests that use a perfect circle as ground truth."""

    RADIUS = 50.0   # metres

    @pytest.fixture(scope="class")
    def circle_track(self):
        return make_circle_track(radius=self.RADIUS)

    def test_total_arc_length(self, circle_track):
        """Total arc length must equal 2πR to within one resample interval."""
        expected = 2.0 * np.pi * self.RADIUS
        assert abs(circle_track.total_length - expected) < RESAMPLE_DS + 0.1, (
            f"Expected total_length ≈ {expected:.1f} m, "
            f"got {circle_track.total_length:.1f} m"
        )

    def test_cum_arc_length_strictly_increasing(self, circle_track):
        """cum_arc_length must be strictly increasing."""
        diffs = np.diff(circle_track.cum_arc_length)
        assert np.all(diffs > 0), "cum_arc_length is not strictly increasing"

    def test_curvature_magnitude(self, circle_track):
        """Mean |κ| should be ≈ 1/R; all values within 20 % of 1/R.

        The finite-difference formula has discretisation error proportional
        to ds²/R².  For R=50 m, ds=2 m the relative error is about 0.16 %,
        well within the 20 % tolerance.
        """
        kappa = circle_track.signed_curvature
        expected = 1.0 / self.RADIUS
        mean_kappa = float(np.mean(np.abs(kappa)))
        assert abs(mean_kappa - expected) / expected < 0.20, (
            f"Mean |κ| = {mean_kappa:.5f}, expected ≈ {expected:.5f}"
        )

    def test_curvature_sign_consistent(self, circle_track):
        """A CCW circle should have uniformly positive signed curvature."""
        kappa = circle_track.signed_curvature
        # Allow a tiny negative tolerance for floating-point edge cases
        assert np.all(kappa > -1e-9), (
            "CCW circle should have non-negative curvature everywhere; "
            f"min kappa = {kappa.min():.6f}"
        )

    def test_tangent_unit_vectors(self, circle_track):
        """All tangent vectors must have unit length (to within 1e-9)."""
        norms = np.hypot(circle_track.tangents[:, 0],
                         circle_track.tangents[:, 1])
        assert np.allclose(norms, 1.0, atol=1e-9), (
            f"Tangent norms not all 1.0; range [{norms.min():.10f}, {norms.max():.10f}]"
        )

    def test_points_on_circle(self, circle_track):
        """All resampled points should lie on the circle to within ds."""
        radii = np.hypot(circle_track.points[:, 0],
                         circle_track.points[:, 1])
        max_err = float(np.max(np.abs(radii - self.RADIUS)))
        assert max_err < RESAMPLE_DS, (
            f"Max radial error = {max_err:.4f} m (ds = {RESAMPLE_DS} m)"
        )


class TestRandomTrack:
    """Procedural tracks from random_track(seed)."""

    def test_random_track_valid_range(self):
        from src.track import random_track, _RANDOM_LENGTH_MIN, _RANDOM_LENGTH_MAX
        from src.track import _RANDOM_MAX_KAPPA

        for seed in (42, 1000, 1005):
            trk = random_track(seed)
            assert _RANDOM_LENGTH_MIN <= trk.total_length <= _RANDOM_LENGTH_MAX
            assert float(np.max(np.abs(trk.signed_curvature))) <= _RANDOM_MAX_KAPPA + 1e-9
            norms = np.hypot(trk.tangents[:, 0], trk.tangents[:, 1])
            np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_random_track_reproducible(self):
        from src.track import random_track

        a = random_track(1000)
        b = random_track(1000)
        np.testing.assert_array_equal(a.points, b.points)


# ---------------------------------------------------------------------------
# Frenet conversion tests
# ---------------------------------------------------------------------------

class TestFrenetOnCircle:
    """Frenet conversion tests using a circle track."""

    RADIUS = 50.0

    @pytest.fixture(scope="class")
    def circle_track(self):
        return make_circle_track(radius=self.RADIUS)

    def test_lateral_offset_left(self, circle_track):
        """A point 2 m to the left of the centreline should give d ≈ +2.

        'Left' is the CCW direction for a CCW circle, i.e. towards the
        centre.  At the point (R, 0) the tangent points in the +y direction,
        so 'left' is the -x direction, placing the offset point at
        (R - 2, 0).
        """
        R = self.RADIUS
        # Centreline point at (R, 0); tangent ≈ (0, 1)
        # Normal (left) ≈ (-1, 0) → offset point at (R-2, 0)
        pos_left = np.array([R - 2.0, 0.0])
        _, d, _ = to_frenet(circle_track, pos_left, heading=np.pi / 2)
        assert abs(d - 2.0) < 0.15, (
            f"Expected d ≈ +2.0 for a point 2 m left, got d = {d:.4f}"
        )

    def test_lateral_offset_right(self, circle_track):
        """A point 2 m to the right should give d ≈ −2."""
        R = self.RADIUS
        pos_right = np.array([R + 2.0, 0.0])
        _, d, _ = to_frenet(circle_track, pos_right, heading=np.pi / 2)
        assert abs(d + 2.0) < 0.15, (
            f"Expected d ≈ -2.0 for a point 2 m right, got d = {d:.4f}"
        )

    def test_zero_heading_error_on_centreline(self, circle_track):
        """A car aligned with the tangent on the centreline has theta_e ≈ 0."""
        R = self.RADIUS
        pos = np.array([R, 0.0])   # on the centreline, tangent ≈ (0,1)
        # Heading = π/2 (pointing in +y direction)
        _, _, theta_e = to_frenet(circle_track, pos, heading=np.pi / 2)
        assert abs(theta_e) < 0.05, (
            f"Expected theta_e ≈ 0 when aligned with tangent, got {theta_e:.4f}"
        )

    def test_heading_error_positive_90(self, circle_track):
        """Car facing left (+90°) relative to tangent gives theta_e ≈ π/2."""
        R = self.RADIUS
        pos = np.array([R, 0.0])
        heading = np.pi   # pointing in -x direction (90° left of +y tangent)
        _, _, theta_e = to_frenet(circle_track, pos, heading=heading)
        assert abs(theta_e - np.pi / 2) < 0.05, (
            f"Expected theta_e ≈ π/2, got {theta_e:.4f}"
        )

    def test_theta_e_wrapping_near_pi(self, circle_track):
        """theta_e must land in [−π, π] even when the raw difference is near ±π."""
        R = self.RADIUS
        pos = np.array([R, 0.0])
        # Tangent ≈ (0, 1) → tangent_angle ≈ π/2.
        # Give a heading just past +π relative to tangent: raw diff ≈ +π + ε → should wrap to −π + ε
        eps = 0.01
        heading = np.pi / 2 + np.pi + eps  # raw theta_e = π + eps
        _, _, theta_e = to_frenet(circle_track, pos, heading=heading)
        assert -np.pi <= theta_e <= np.pi, (
            f"theta_e = {theta_e:.4f} is outside [−π, π]"
        )
        # After wrapping π + ε → −π + ε
        assert abs(theta_e - (-np.pi + eps)) < 0.05, (
            f"Expected theta_e ≈ {-np.pi + eps:.4f}, got {theta_e:.4f}"
        )

    def test_theta_e_wrapping_near_neg_pi(self, circle_track):
        """theta_e stays in [−π, π] when raw difference is slightly below −π."""
        R = self.RADIUS
        pos = np.array([R, 0.0])
        eps = 0.01
        heading = np.pi / 2 - np.pi - eps   # raw theta_e = −π − eps
        _, _, theta_e = to_frenet(circle_track, pos, heading=heading)
        assert -np.pi <= theta_e <= np.pi, (
            f"theta_e = {theta_e:.4f} is outside [−π, π]"
        )
        assert abs(theta_e - (np.pi - eps)) < 0.05, (
            f"Expected theta_e ≈ {np.pi - eps:.4f}, got {theta_e:.4f}"
        )

    def test_lookahead_curvatures_shape(self, circle_track):
        """lookahead_curvatures must return an array of the same length as distances."""
        distances = np.array([5, 10, 15, 20, 30, 40, 55, 70, 90, 110], dtype=float)
        kappas = lookahead_curvatures(circle_track, s=0.0, distances=distances)
        assert kappas.shape == (10,), f"Expected shape (10,), got {kappas.shape}"

    def test_lookahead_curvatures_values(self, circle_track):
        """Lookahead curvatures on a circle should all ≈ 1/R (with correct sign)."""
        R = self.RADIUS
        distances = np.array([5, 10, 20, 50, 100], dtype=float)
        kappas = lookahead_curvatures(circle_track, s=0.0, distances=distances)
        expected = 1.0 / R
        assert np.allclose(kappas, expected, atol=0.003), (
            f"Lookahead curvatures should be ≈ {expected:.5f}; got {kappas}"
        )

    def test_lookahead_wrap_around(self, circle_track):
        """Lookahead wrap-around: distances that exceed total length work correctly."""
        R = self.RADIUS
        # Start near the end of the track; a large distance should wrap back
        s_near_end = circle_track.total_length - 5.0
        distances = np.array([20.0, 50.0, 100.0])
        kappas = lookahead_curvatures(circle_track, s=s_near_end, distances=distances)
        expected = 1.0 / R
        assert np.allclose(kappas, expected, atol=0.003), (
            f"Wrap-around lookahead failed; got {kappas}"
        )


# ---------------------------------------------------------------------------
# Warm-start index tests
# ---------------------------------------------------------------------------

class TestClosestPointIndex:
    """Tests for closest_point_index, including warm-start."""

    @pytest.fixture(scope="class")
    def circle_track(self):
        return make_circle_track(radius=50.0)

    def test_full_scan_finds_nearest(self, circle_track):
        """Full scan should find the index whose point is closest to a query."""
        # Query point at (50, 0) — should be very close to one of the samples
        query = np.array([50.0, 0.0])
        idx = closest_point_index(circle_track, query)
        nearest_dist = np.hypot(*(circle_track.points[idx] - query))
        assert nearest_dist < RESAMPLE_DS, (
            f"Full-scan nearest point is {nearest_dist:.4f} m away (> ds={RESAMPLE_DS})"
        )

    def test_warm_start_same_as_full_scan(self, circle_track):
        """Warm-start should return the same index as a full scan."""
        query = np.array([40.0, 30.0])
        idx_full  = closest_point_index(circle_track, query, start_index=None)
        # Warm-start from a nearby known index
        idx_warm  = closest_point_index(circle_track, query, start_index=idx_full)
        assert idx_warm == idx_full, (
            f"Warm-start gave index {idx_warm}, full scan gave {idx_full}"
        )
