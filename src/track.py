import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from scipy.interpolate import CubicSpline

HALF_WIDTH: float = 6.0      # metres, track half-width (fixed by spec)
RESAMPLE_DS: float = 1.0     # metres, arc-length spacing between samples
CURVATURE_SCALE: float = 20.0  # multiplier used when building the state vector

@dataclass
class Track:
    points: np.ndarray
    cum_arc_length: np.ndarray
    tangents: np.ndarray
    signed_curvature: np.ndarray
    half_width: float = HALF_WIDTH
    total_length: float = field(init=False)

    def __post_init__(self) -> None:
        self.total_length = float(self.cum_arc_length[-1]) + RESAMPLE_DS

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fit_periodic_spline(ctrl_pts: np.ndarray) -> CubicSpline:
    # Close the loop by appending the first point
    pts = np.vstack([ctrl_pts, ctrl_pts[0]])
    # Chord-length parameterisation
    diffs = np.diff(pts, axis=0)
    chord = np.hypot(diffs[:, 0], diffs[:, 1])
    t = np.concatenate([[0.0], np.cumsum(chord)])
    return CubicSpline(t, pts, bc_type='periodic')


def _resample_spline(spline: CubicSpline, ds: float = RESAMPLE_DS) -> Tuple[np.ndarray, np.ndarray]:
    t_end = spline.x[-1]

    # Dense pre-sample to estimate total arc length and build a fine table
    n_fine = max(10_000, int(t_end * 100))
    t_fine = np.linspace(0.0, t_end, n_fine, endpoint=False)
    pts_fine = spline(t_fine)
    seg_len = np.hypot(*np.diff(pts_fine, axis=0).T)
    arc_fine = np.concatenate([[0.0], np.cumsum(seg_len)])

    total_arc = arc_fine[-1]
    n_samples = max(4, int(total_arc / ds))
    target_arcs = np.linspace(0.0, total_arc, n_samples, endpoint=False)

    # Invert arc → t via linear interpolation on the fine table
    t_samples = np.interp(target_arcs, arc_fine, t_fine)
    points = spline(t_samples)
    return points, t_samples


def _compute_tangents(points: np.ndarray) -> np.ndarray:
    n = len(points)
    p_prev = points[np.arange(n) - 1]    # wrap: index -1 → last element
    p_next = points[(np.arange(n) + 1) % n]
    diff = p_next - p_prev
    norms = np.hypot(diff[:, 0], diff[:, 1])[:, np.newaxis]
    return diff / norms


def _compute_signed_curvature(points: np.ndarray) -> np.ndarray:
    n = len(points)
    idx = np.arange(n)
    a = points[(idx - 1) % n]   # p_{i-1}
    b = points[idx]              # p_i
    c = points[(idx + 1) % n]   # p_{i+1}

    ab = b - a
    bc = c - b
    ac = c - a

    cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]   # 2D cross product
    len_ab = np.hypot(ab[:, 0], ab[:, 1])
    len_bc = np.hypot(bc[:, 0], bc[:, 1])
    len_ac = np.hypot(ac[:, 0], ac[:, 1])

    denom = len_ab * len_bc * len_ac
    # Avoid division by zero for degenerate (co-linear) triplets
    kappa = np.where(denom > 1e-12, 2.0 * cross / denom, 0.0)
    return kappa

# ---------------------------------------------------------------------------
# Public factory: build a Track from control points
# ---------------------------------------------------------------------------

def build_track(ctrl_pts: np.ndarray, ds: float = RESAMPLE_DS) -> Track:
    ctrl_pts = np.asarray(ctrl_pts, dtype=float)
    spline = _fit_periodic_spline(ctrl_pts)
    points, _ = _resample_spline(spline, ds)

    tangents = _compute_tangents(points)
    signed_curvature = _compute_signed_curvature(points)

    n = len(points)
    cum_arc = np.arange(n, dtype=float) * ds   # uniform by construction

    return Track(
        points=points,
        cum_arc_length=cum_arc,
        tangents=tangents,
        signed_curvature=signed_curvature,
    )


def track() -> Track:
    ctrl_pts = np.array([
        [0.0,    0.0],    # start / finish line
        [250.0,  0.0],    # end of main straight
        [290.0,  15.0],   # chicane entry (left)
        [310.0,  40.0],   # chicane apex (right)
        [330.0,  42.0],   # chicane exit / second straight
        [370.0,  40.0],   # braking zone for hairpin
        [390.0,  75.0],   # hairpin entry
        [380.0, 130.0],   # hairpin apex (tight right)
        [350.0, 155.0],   # hairpin exit
        [270.0, 160.0],   # S-section start
        [210.0, 175.0],   # S apex 1 (left)
        [160.0, 155.0],   # S apex 2 (right)
        [110.0, 170.0],   # S apex 3 (left)
        [60.0,  155.0],   # S-section exit
        [25.0,  125.0],   # slow technical corner
        [-20.0,  90.0],   # long sweeping left
        [-30.0,  50.0],   # return sweep
        [-10.0,  15.0],   # final corner entry
    ], dtype=float)
    return build_track(ctrl_pts)
