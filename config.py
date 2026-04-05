"""
config.py
---------
Central configuration for the Multi-Object Detection & Tracking pipeline.
All tuneable hyper-parameters live here so nothing is hard-coded elsewhere.
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
SCREENSHOTS_DIR = os.path.join(OUTPUT_DIR, "screenshots")

# ── Detection (YOLOv8) ──────────────────────────────────────────────────
YOLO_MODEL = "yolov8n.pt"          # Nano model – fast & free to download
CONFIDENCE_THRESHOLD = 0.25        # Minimum detection confidence
IOU_THRESHOLD = 0.45               # NMS IoU threshold
TARGET_CLASSES = [0]               # COCO class 0 = person

# ── Tracking (ByteTrack via Ultralytics built-in tracker) ────────────────
TRACKER_TYPE = "bytetrack"         # "bytetrack" or "botsort"
# ByteTrack hyper-parameters (written to a temp YAML at runtime)
TRACK_HIGH_THRESH = 0.25           # High detection threshold
TRACK_LOW_THRESH = 0.1             # Low detection threshold (second association)
TRACK_NEW_THRESH = 0.3             # Threshold for new track initialisation
TRACK_BUFFER = 60                  # Frames to keep a lost track alive
MATCH_THRESH = 0.8                 # IoU matching threshold

# ── Visualisation ───────────────────────────────────────────────────────
BBOX_THICKNESS = 2
FONT_SCALE = 0.7
FONT_THICKNESS = 2
SCREENSHOT_INTERVAL = 100          # Save a screenshot every N frames
DRAW_TRAJECTORIES = True           # Whether to draw movement tails
TRAJECTORY_LENGTH = 30             # How many past frames to show in the tail

# ── Video output ─────────────────────────────────────────────────────────
OUTPUT_FPS = None                  # None → use source FPS
OUTPUT_CODEC = "mp4v"              # FourCC codec for VideoWriter

# ── Performance ──────────────────────────────────────────────────────────
# Process every Nth frame (1 = every frame, 2 = every 2nd, 3 = every 3rd)
# Skipped frames reuse the previous tracking result → big speed boost.
# Recommended: 2 for CPU, 1 for GPU
PROCESS_EVERY_N_FRAMES = 2

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
