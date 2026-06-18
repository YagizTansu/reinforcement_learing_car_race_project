import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for file saving
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.axes import Axes
from typing import Optional

from src.track import Track, track as default_track


def plot_track(
    track: Track,
    ax: Optional[Axes] = None,
    title: str = "",
    color_center: str = "black",
    color_boundary: str = "gray",
    linewidth: float = 1.5,
) -> Axes:
    if ax is None:
        ax = plt.gca()

    pts = track.points          # (N, 2)
    t = track.tangents          # (N, 2)
    hw = track.half_width

    # Normal: rotate tangent 90° CCW
    normals = np.column_stack([-t[:, 1], t[:, 0]])   # (N, 2)

    left_boundary  = pts + hw * normals   # (N, 2)
    right_boundary = pts - hw * normals   # (N, 2)

    # Closed-loop: append first point to close the polygon
    def _close(arr):
        return np.vstack([arr, arr[0]])

    pts_c  = _close(pts)
    left_c = _close(left_boundary)
    right_c = _close(right_boundary)

    # Filled track band
    band_x = np.concatenate([left_c[:, 0], right_c[::-1, 0]])
    band_y = np.concatenate([left_c[:, 1], right_c[::-1, 1]])
    ax.fill(band_x, band_y, color="lightgray", zorder=0, alpha=0.5)

    # Boundary lines
    ax.plot(left_c[:, 0],  left_c[:, 1],  color=color_boundary,
            linewidth=linewidth, zorder=1)
    ax.plot(right_c[:, 0], right_c[:, 1], color=color_boundary,
            linewidth=linewidth, zorder=1)

    # Centreline (dashed)
    ax.plot(pts_c[:, 0], pts_c[:, 1], color=color_center,
            linewidth=linewidth * 0.8, linestyle="--", zorder=2, alpha=0.7)

    # Start marker
    ax.plot(pts[0, 0], pts[0, 1], "go", markersize=7, zorder=3,
            label="Start")

    if title:
        ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


# ---------------------------------------------------------------------------
# Main: generate preview figure
# ---------------------------------------------------------------------------

def generate_tracks_preview(
    output_path: str = "experiments/figures/tracks_preview.png",
) -> None:
    """Render the fixed track and save to PNG."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle("Track preview", fontsize=14, fontweight="bold")

    trk = default_track()
    plot_track(trk, ax=ax,
               title=f"Fixed track\n({len(trk.points)} pts, {trk.total_length:.0f} m)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    generate_tracks_preview()
