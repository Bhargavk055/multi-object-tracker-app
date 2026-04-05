"""
utils.py
--------
Helper utilities: video download, screenshot capture, logging helpers.
"""

import glob
import os
import re
import shutil
import subprocess
import sys
import cv2
import config


def _extract_video_id(url: str) -> str:
    """
    Extract a short, filesystem-safe ID from a YouTube (or generic) URL.
    Examples:
        https://youtube.com/shorts/V8c6KrpCiUs  →  V8c6KrpCiUs
        https://www.youtube.com/watch?v=abc123   →  abc123
        https://example.com/my_video.mp4         →  my_video
    """
    # YouTube /shorts/ID or /watch?v=ID
    match = re.search(r"(?:shorts/|watch\?v=|youtu\.be/)([\w-]+)", url)
    if match:
        return match.group(1)

    # Generic URL – use last path segment without extension
    basename = url.rstrip("/").split("/")[-1].split("?")[0]
    name = os.path.splitext(basename)[0]
    return name if name else "video"


def clean_previous_outputs():
    """
    Remove all previous output files so every run starts fresh.
    Called automatically at the start of each pipeline run.
    """
    # Clean output video(s)
    for f in glob.glob(os.path.join(config.OUTPUT_DIR, "tracked_*")):
        try:
            os.remove(f)
            print(f"[clean] Removed old output: {os.path.basename(f)}")
        except OSError:
            pass

    # Clean screenshots (delete files inside, keep the folder)
    ss_dir = config.SCREENSHOTS_DIR
    if os.path.isdir(ss_dir):
        for f in glob.glob(os.path.join(ss_dir, "*")):
            try:
                os.remove(f)
            except OSError:
                pass
        print("[clean] Cleared old screenshots")
    os.makedirs(ss_dir, exist_ok=True)


def clean_previous_inputs():
    """Remove all previously downloaded videos from input/."""
    for f in glob.glob(os.path.join(config.INPUT_DIR, "*")):
        if os.path.isfile(f):
            os.remove(f)
            print(f"[clean] Removed old input: {os.path.basename(f)}")


def download_video(url: str) -> str:
    """
    Download a video from YouTube / any public URL using yt-dlp.

    - Extracts the video ID from the URL for a unique filename
    - Deletes any previously downloaded videos first
    - Always downloads fresh (never reuses old files)

    Parameters
    ----------
    url : str – Public video URL

    Returns
    -------
    str – absolute path to the downloaded file
    """
    # Step 1: Clean old downloads so there's no confusion
    clean_previous_inputs()

    # Step 2: Create a unique filename from the video ID
    video_id = _extract_video_id(url)
    output_path = os.path.join(config.INPUT_DIR, f"{video_id}.mp4")

    print(f"[download] URL      : {url}")
    print(f"[download] Video ID : {video_id}")
    print(f"[download] Saving to: {output_path}")

    # Step 3: Download
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--js-runtimes", "node",
        "--force-overwrites",
        "--no-playlist",
        "-f", "best[ext=mp4]/bestvideo[ext=mp4]",
        "-o", output_path,
        url,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print()
        print("[ERROR] ❌ Failed to download video!")
        print("        Possible reasons:")
        print("        1. The URL is invalid or the video does not exist")
        print("        2. The video is private or age-restricted")
        print("        3. No internet connection")
        print()
        print("        Please provide a valid public YouTube URL.")
        print("        Example: python main.py --url \"https://youtube.com/shorts/V8c6KrpCiUs\"")
        sys.exit(1)

    # Step 4: Verify the download
    if not os.path.isfile(output_path):
        print("[ERROR] Download finished but file not found!")
        sys.exit(1)

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[download] Complete! File size: {file_size:.1f} MB")
    return output_path


def save_screenshot(frame, frame_idx: int, out_dir: str = None) -> str:
    """Save a single frame as a PNG screenshot."""
    if out_dir is None:
        out_dir = config.SCREENSHOTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"frame_{frame_idx:06d}.png")
    cv2.imwrite(path, frame)
    return path


def get_video_info(video_path: str) -> dict:
    """Return basic metadata about a video file."""
    cap = cv2.VideoCapture(video_path)
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    info["duration_sec"] = (
        info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
    )
    return info


def print_banner():
    """Pretty-print a startup banner."""
    print("=" * 60)
    print("  Multi-Object Detection & Persistent ID Tracking")
    print("  YOLOv8 + ByteTrack Pipeline")
    print("=" * 60)
