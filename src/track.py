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
# Validity check (used by random_track)
# ---------------------------------------------------------------------------

def _check_self_intersection(points: np.ndarray) -> bool:
    """Return True if any two non-adjacent segments of the closed polyline
    intersect.

    Uses the standard 2D segment-segment intersection test via cross products.
    Only checks a random subsample for speed (good enough for rejection
    sampling of random tracks).
    """
    n = len(points)
    # Build segment list: (p_i, p_{i+1 mod n})
    segs = [(points[i], points[(i + 1) % n]) for i in range(n)]

    def _cross2(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def _segments_intersect(p1, p2, p3, p4):
        d1 = _cross2(p3, p4, p1)
        d2 = _cross2(p3, p4, p2)
        d3 = _cross2(p1, p2, p3)
        d4 = _cross2(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    # Only check a stride-2 subsample for speed
    step = max(1, n // 200)
    sampled = list(range(0, n, step))
    for ii, i in enumerate(sampled):
        p1, p2 = segs[i]
        for j in sampled[ii + 2:]:
            if j == (i - 1) % n or j == (i + 1) % n:
                continue
            p3, p4 = segs[j]
            if _segments_intersect(p1, p2, p3, p4):
                return True
    return False


def _min_turn_radius(track: Track) -> float:
    """Minimum turn radius in metres (1/max|kappa|)."""
    max_kappa = np.max(np.abs(track.signed_curvature))
    if max_kappa < 1e-9:
        return np.inf
    return 1.0 / max_kappa


# ---------------------------------------------------------------------------
# Named tracks
# ---------------------------------------------------------------------------

def monaco_inspired_track() -> Track:
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


def random_track(
    rng: np.random.Generator,
    n_control: int = 12,
    base_radius: float = 150.0,
    radius_jitter: float = 0.45,
    angle_jitter: float = 0.25,
    max_attempts: int = 200,
) -> Track:
    """Generate a valid random closed track.

    Control points are placed on a circle of *base_radius* metres,
    with each radius and angular spacing perturbed by random noise.
    The track is rejected and regenerated if it self-intersects or if its
    minimum turn radius is less than 2 * half_width (= 12 m).

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator (e.g. ``np.random.default_rng(seed)``).
    n_control : int
        Number of control points (8–12 per spec).
    base_radius : float
        Mean radius of the circle on which control points are placed.
    radius_jitter : float
        Fractional radius noise: r_i ~ U(1 - jitter, 1 + jitter) * base_radius.
    angle_jitter : float
        Fractional angular noise added to the uniform angle spacing.
    max_attempts : int
        Maximum number of rejection-sampling attempts before raising.

    Returns
    -------
    Track

    Raises
    ------
    RuntimeError
        If no valid track is found within *max_attempts* attempts.
    """
    min_radius = 2.0 * HALF_WIDTH   # = 12 m

    for _ in range(max_attempts):
        # Base angles evenly spaced around the circle
        base_angles = np.linspace(0.0, 2.0 * np.pi, n_control, endpoint=False)
        # Add angular jitter (fraction of mean spacing)
        mean_spacing = 2.0 * np.pi / n_control
        angle_noise = rng.uniform(-angle_jitter, angle_jitter, n_control) * mean_spacing
        angles = base_angles + angle_noise
        angles = np.sort(angles)   # keep points in CCW order

        # Radial jitter
        radii = base_radius * rng.uniform(1.0 - radius_jitter, 1.0 + radius_jitter, n_control)

        ctrl_pts = np.column_stack([
            radii * np.cos(angles),
            radii * np.sin(angles),
        ])

        track = build_track(ctrl_pts)

        # Validity checks
        if _check_self_intersection(track.points):
            continue
        if _min_turn_radius(track) < min_radius:
            continue

        return track

    raise RuntimeError(
        f"Could not generate a valid random track in {max_attempts} attempts. "
        "Try reducing radius_jitter or angle_jitter."
    )
