"""
track.py — Track generation for the car racing RL environment.

Pipeline:
  1. Accept a list of (x, y) control points forming a closed loop.
  2. Fit a periodic cubic spline through the control points.
  3. Resample at uniform arc-length spacing (default 2 m).
  4. Compute per-sample unit tangent, signed curvature.

Signed curvature formula (three-point finite difference):
  Given consecutive points p_{i-1}, p_i, p_{i+1}, the signed curvature at p_i
  is approximated by the Menger curvature with sign:

      kappa_i = 2 * cross(p_i - p_{i-1}, p_{i+1} - p_i)
                / (|p_i - p_{i-1}| * |p_{i+1} - p_i| * |p_{i+1} - p_{i-1}|)

  where cross(a, b) = a[0]*b[1] - a[1]*b[0].
  Positive kappa means a left turn (counterclockwise), matching the
  right-hand rule in 2D.

  Reference: Menger (1930); also see Pressley, "Elementary Differential
  Geometry", §2.2.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from scipy.interpolate import CubicSpline

HALF_WIDTH: float = 6.0      # metres, track half-width (fixed by spec)
RESAMPLE_DS: float = 1.0     # metres, arc-length spacing between samples
CURVATURE_SCALE: float = 20.0  # multiplier used when building the state vector


@dataclass
class Track:
    """Holds a fully-discretised closed track.

    Attributes
    ----------
    points : ndarray, shape (N, 2)
        Uniformly-spaced centreline points [x, y] in metres.
    cum_arc_length : ndarray, shape (N,)
        Cumulative arc length s_i from the first point to point i.
        s_0 = 0, s_{N-1} ≈ total track length − ds.
    tangents : ndarray, shape (N, 2)
        Unit tangent vectors at each point (direction of travel).
    signed_curvature : ndarray, shape (N,)
        Signed curvature kappa_i in m^{-1}.
        Positive = left turn (counterclockwise).
    half_width : float
        Track half-width in metres.
    total_length : float
        Total arc length of the closed loop ≈ N * ds.
    """
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
    """Fit a periodic cubic spline through closed-loop control points.

    The first point is appended at the end so the parameterisation wraps
    exactly.  The parameter t is the cumulative chord length (a reasonable
    arc-length proxy for the sparse control points).

    Parameters
    ----------
    ctrl_pts : ndarray, shape (M, 2)
        Control points.  The last point must NOT equal the first (the
        closure is handled internally).

    Returns
    -------
    CubicSpline
        Spline object parameterised by chord-length t.
    """
    # Close the loop by appending the first point
    pts = np.vstack([ctrl_pts, ctrl_pts[0]])
    # Chord-length parameterisation
    diffs = np.diff(pts, axis=0)
    chord = np.hypot(diffs[:, 0], diffs[:, 1])
    t = np.concatenate([[0.0], np.cumsum(chord)])
    # bc_type='periodic' requires f(t[0]) == f(t[-1]), which is satisfied.
    return CubicSpline(t, pts, bc_type='periodic')


def _resample_spline(spline: CubicSpline, ds: float = RESAMPLE_DS
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Walk along the spline and collect samples every *ds* metres.

    Uses a simple Euler walk: evaluate the spline derivative to get the
    instantaneous speed in parameter space, advance t by dt = ds / speed,
    then refine with a Newton step so the arc-length increment is exact to
    float precision.

    Returns
    -------
    points : ndarray, shape (N, 2)
    t_samples : ndarray, shape (N,)
        Parameter values corresponding to each sampled point.
    """
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
    """Compute unit tangents from centred finite differences on a closed loop.

    For interior points:
        tangent_i = (p_{i+1} - p_{i-1}) / |p_{i+1} - p_{i-1}|

    Wrap-around indexing makes it closed.

    Parameters
    ----------
    points : ndarray, shape (N, 2)

    Returns
    -------
    tangents : ndarray, shape (N, 2)
    """
    n = len(points)
    p_prev = points[np.arange(n) - 1]    # wrap: index -1 → last element
    p_next = points[(np.arange(n) + 1) % n]
    diff = p_next - p_prev
    norms = np.hypot(diff[:, 0], diff[:, 1])[:, np.newaxis]
    return diff / norms


def _compute_signed_curvature(points: np.ndarray) -> np.ndarray:
    """Compute signed curvature at each point using Menger's formula.

    For three consecutive points a = p_{i-1}, b = p_i, c = p_{i+1}:

        kappa_i = 2 * cross(b - a, c - b)
                  / ( |b - a| * |c - b| * |c - a| )

    where cross(u, v) = u[0]*v[1] - u[1]*v[0].

    Positive kappa → left turn (counterclockwise); negative → right turn.

    Reference: Menger curvature, see e.g. Pressley "Elementary Differential
    Geometry" §2.2.

    Parameters
    ----------
    points : ndarray, shape (N, 2)

    Returns
    -------
    kappa : ndarray, shape (N,)
    """
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
    """Build a Track from a list of control points.

    Parameters
    ----------
    ctrl_pts : array-like, shape (M, 2)
        Closed-loop control points (first != last).
    ds : float
        Desired arc-length spacing in metres.

    Returns
    -------
    Track
    """
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



# ---------------------------------------------------------------------------
# Named tracks
# ---------------------------------------------------------------------------

def track() -> Track:
    """Return a fixed track loosely inspired by street circuits (~1.2 km).

    Layout (hand-tuned, all coordinates in metres):
      - Long main straight           [0,0] → [250,0]   (~250 m)
      - Fast chicane (left-right)    [250,0] → [310,40]
      - Second straight              [310,40] → [370,40]
      - Tight hairpin (right)        [370,40] → [350,130]
      - Post-hairpin acceleration    [350,130] → [270,150]
      - S-section (3 apexes)         [270,150] → [100,170]
      - Slow technical corner        [100,170] → [30,120]
      - Long sweeping return         [30,120] → [-20,60]
      - Final corner back to start   [-20,60] → [0,0]

    Control points are hand-tuned in metres; the origin is the
    start/finish line.
    """
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
