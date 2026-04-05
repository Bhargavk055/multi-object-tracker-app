"""
visualizer.py
-------------
Drawing utilities for bounding boxes, track IDs, and HUD overlay.
"""

from collections import defaultdict
import cv2
import numpy as np
import config

# 20 distinct, high-contrast colours for up to 20 simultaneous tracks.
# Wraps around via modulo for more.
_PALETTE = [
    (230, 25, 75),   (60, 180, 75),   (255, 225, 25),  (0, 130, 200),
    (245, 130, 48),  (145, 30, 180),  (70, 240, 240),  (240, 50, 230),
    (210, 245, 60),  (250, 190, 212), (0, 128, 128),   (220, 190, 255),
    (170, 110, 40),  (255, 250, 200), (128, 0, 0),     (170, 255, 195),
    (128, 128, 0),   (255, 215, 180), (0, 0, 128),     (128, 128, 128),
]

_track_history = defaultdict(list)

def _color_for_id(track_id: int) -> tuple:
    """Return a deterministic BGR colour for a given track ID."""
    rgb = _PALETTE[track_id % len(_PALETTE)]
    return (rgb[2], rgb[1], rgb[0])  # RGB → BGR


def draw_tracks(frame, tracks: list, frame_idx: int = 0) -> np.ndarray:
    """
    Draw bounding boxes + IDs on *frame* (in-place) and return it.

    Parameters
    ----------
    frame : np.ndarray  – BGR image
    tracks : list[dict] – output of Tracker.update()
    frame_idx : int     – current frame number (shown on HUD)
    """
    overlay = frame.copy()

    # Clear history for tracks not seen recently? (optional, but let's keep it simple)
    # The history just naturally ages out if we wanted, but we don't strictly need to clear it for simple tracking.

    for t in tracks:
        x1, y1, x2, y2 = map(int, t["bbox"])
        tid = t["id"]
        conf = t["conf"]
        color = _color_for_id(tid)

        # ── Trajectory ──────────────────────────────────────────────
        if getattr(config, "DRAW_TRAJECTORIES", False):
            # Calculate bottom center
            cx = int((x1 + x2) / 2)
            cy = int(y2)
            _track_history[tid].append((cx, cy))
            
            # Keep only the last N points
            if len(_track_history[tid]) > getattr(config, "TRAJECTORY_LENGTH", 30):
                _track_history[tid].pop(0)

            pts = _track_history[tid]
            thickness = max(1, config.BBOX_THICKNESS - 1)
            for i in range(1, len(pts)):
                cv2.line(overlay, pts[i - 1], pts[i], color, thickness)

        # ── Bounding box ────────────────────────────────────────────
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, config.BBOX_THICKNESS)

        # ── Label background ────────────────────────────────────────
        label = f"ID:{tid}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, config.FONT_THICKNESS
        )
        cv2.rectangle(overlay, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            overlay, label, (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE,
            (255, 255, 255), config.FONT_THICKNESS, cv2.LINE_AA,
        )

    # ── HUD: frame counter + active track count ────────────────────
    hud = f"Frame {frame_idx}  |  Tracks: {len(tracks)}"
    cv2.putText(
        overlay, hud, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
    )

    return overlay
