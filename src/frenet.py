"""
frenet.py — Cartesian → Frenet (track-relative) coordinate conversion.

Frenet frame on a planar curve:
  s        : arc-length distance along the centreline from the start.
  d        : signed lateral offset from the centreline.
             Positive d = left of the direction of travel (right-hand rule
             in 2D: d > 0 when the point is to the left of the tangent).
  theta_e  : heading error = vehicle heading − track tangent angle,
             wrapped to [−π, π].

All functions operate on a Track object (see track.py).
"""

import numpy as np
from typing import Optional
from src.track import Track


# ---------------------------------------------------------------------------
# Closest-point lookup
# ---------------------------------------------------------------------------

def closest_point_index(
    track: Track,
    position: np.ndarray,
    start_index: Optional[int] = None,
) -> int:
    """Return the index of the centreline point nearest to *position*.

    Parameters
    ----------
    track : Track
    position : array-like, shape (2,)
        Cartesian position [x, y] in metres.
    start_index : int or None
        If given, search starts at this index and scans a window of ±half
        the track length.  Useful for warm-starting from the previous index
        to avoid wrap-around confusion.  If None, a full linear scan is done.

    Returns
    -------
    int
        Index into track.points of the nearest centreline point.
    """
    pos = np.asarray(position, dtype=float)
    n = len(track.points)

    if start_index is None:
        # Full linear scan — O(N)
        deltas = track.points - pos          # (N, 2)
        dists_sq = deltas[:, 0] ** 2 + deltas[:, 1] ** 2
        return int(np.argmin(dists_sq))

    # Warm-start: only scan a window of ±search_half around start_index.
    # Window size = 10 % of track length, at least 30 points.
    search_half = max(30, n // 10)
    indices = np.arange(start_index - search_half,
                        start_index + search_half + 1) % n
    pts_window = track.points[indices]
    deltas = pts_window - pos
    dists_sq = deltas[:, 0] ** 2 + deltas[:, 1] ** 2
    local_best = int(np.argmin(dists_sq))
    return int(indices[local_best])


# ---------------------------------------------------------------------------
# Full Frenet conversion
# ---------------------------------------------------------------------------

def to_frenet(
    track: Track,
    position: np.ndarray,
    heading: float,
    start_index: Optional[int] = None,
) -> tuple[float, float, float]:
    """Convert a Cartesian pose to track-relative (Frenet) coordinates.

    Algorithm
    ---------
    1. Find the nearest centreline point p* with index i*.
    2. s  = track.cum_arc_length[i*]
    3. normal = rotate tangent 90° CCW = (−t_y, t_x)
       d  = dot(position − p*, normal)
       Positive d ↔ position is to the LEFT of the centreline.
    4. theta_e = heading − atan2(tangent_y, tangent_x), wrapped to [−π, π].

    Parameters
    ----------
    track : Track
    position : array-like, shape (2,)
    heading : float
        Vehicle heading in radians (measured CCW from the positive x-axis).
    start_index : int or None
        Warm-start index for closest_point_index (see that function).

    Returns
    -------
    s : float
        Arc-length along the centreline (metres).
    d : float
        Signed lateral offset (metres).  Positive = left.
    theta_e : float
        Heading error (radians) in [−π, π].
    """
    pos = np.asarray(position, dtype=float)
    idx = closest_point_index(track, pos, start_index)

    p_star = track.points[idx]
    tangent = track.tangents[idx]

    s = float(track.cum_arc_length[idx])

    # Normal points 90° CCW from the tangent: n = (−t_y, t_x)
    normal = np.array([-tangent[1], tangent[0]])
    d = float(np.dot(pos - p_star, normal))

    track_angle = float(np.arctan2(tangent[1], tangent[0]))
    theta_e = heading - track_angle
    # Wrap to [−π, π]
    theta_e = float((theta_e + np.pi) % (2.0 * np.pi) - np.pi)

    return s, d, theta_e


# ---------------------------------------------------------------------------
# Lookahead curvature sampling
# ---------------------------------------------------------------------------

def lookahead_curvatures(
    track: Track,
    s: float,
    distances: np.ndarray,
) -> np.ndarray:
    """Sample the signed curvature at arc-length positions s + d_i.

    The arc length wraps around the closed loop
    (modulo track.total_length).

    Parameters
    ----------
    track : Track
    s : float
        Current arc-length position along the centreline (metres).
    distances : array-like, shape (K,)
        Lookahead distances in metres.

    Returns
    -------
    kappas : ndarray, shape (K,)
        Signed curvature values (m^{-1}).
    """
    distances = np.asarray(distances, dtype=float)
    target_s = (s + distances) % track.total_length  # wrap around the loop

    # Convert arc-length targets to indices:
    # cum_arc_length is uniformly spaced, so index ≈ s / ds.
    # Use np.searchsorted for correctness with non-uniform rounding.
    ds = track.total_length / len(track.points)
    indices = np.round(target_s / ds).astype(int) % len(track.points)

    return track.signed_curvature[indices]
