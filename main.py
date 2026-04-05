"""
main.py
-------
Main pipeline script – ties detection, tracking, and visualisation together.

Every run starts fresh:
  1. Old inputs and outputs are cleaned automatically
  2. New video is downloaded (if URL given)
  3. Detection + tracking runs on every frame
  4. Annotated video + screenshots are saved

Usage
-----
    python main.py --source input/video.mp4
    python main.py --url "https://youtube.com/shorts/VIDEO_ID"
    python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
"""

import argparse
import os
import sys
import time

import cv2

import config
from tracker import Tracker
from visualizer import draw_tracks
from utils import (
    download_video,
    save_screenshot,
    get_video_info,
    print_banner,
    clean_previous_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Object Detection & Persistent ID Tracking"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Path to input video file.",
    )
    parser.add_argument(
        "--url", type=str, default=None,
        help="Public video URL (YouTube, etc.). Downloaded via yt-dlp.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path for annotated output video (default: output/tracked_<name>.mp4).",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show live preview window while processing.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Process only the first N frames (useful for quick tests).",
    )
    return parser.parse_args()


def resolve_source(args) -> str:
    """Return the path to the input video, downloading if necessary."""
    if args.source:
        if not os.path.isfile(args.source):
            print(f"[ERROR] File not found: {args.source}")
            sys.exit(1)
        return args.source

    if args.url:
        return download_video(args.url)

    # Fallback: look for any video in input/
    for f in os.listdir(config.INPUT_DIR):
        if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
            return os.path.join(config.INPUT_DIR, f)

    print("[ERROR] No video source provided. Use --source or --url, "
          "or place a video file in the input/ folder.")
    sys.exit(1)


def run_pipeline(source_path: str, output_path: str,
                 show: bool = False, max_frames: int = None):
    """Core processing loop."""

    # ── Video info ───────────────────────────────────────────────────
    info = get_video_info(source_path)
    print()
    print(f"[Pipeline] Input      : {source_path}")
    print(f"[Pipeline] Resolution : {info['width']}×{info['height']}")
    print(f"[Pipeline] FPS        : {info['fps']:.1f}")
    print(f"[Pipeline] Frames     : {info['frame_count']}")
    print(f"[Pipeline] Duration   : {info['duration_sec']:.1f}s")
    print(f"[Pipeline] Output     : {output_path}")
    print()

    # ── Open video reader ────────────────────────────────────────────
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        print("[ERROR] Cannot open video file.")
        sys.exit(1)

    fps = config.OUTPUT_FPS or info["fps"]
    fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_CODEC)
    writer = cv2.VideoWriter(output_path, fourcc, fps,
                             (info["width"], info["height"]))

    # ── Initialise tracker ───────────────────────────────────────────
    tracker = Tracker()

    frame_idx = 0
    processed_count = 0
    last_tracks = []
    track_counts = []
    skip_n = config.PROCESS_EVERY_N_FRAMES
    t_start = time.time()

    if skip_n > 1:
        print(f"[Pipeline] ⚡ Speed mode: processing every {skip_n} frame(s)")
        print()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_idx >= max_frames:
                break

            # ── Track (only on every Nth frame) ──────────────────────
            if frame_idx % skip_n == 0:
                tracks = tracker.update(frame)
                last_tracks = tracks
                processed_count += 1
            else:
                tracks = last_tracks  # reuse previous result

            track_counts.append(len(tracks))

            # ── Visualise ────────────────────────────────────────────
            annotated = draw_tracks(frame, tracks, frame_idx)

            # ── Write output ─────────────────────────────────────────
            writer.write(annotated)

            # ── Screenshots ──────────────────────────────────────────
            if frame_idx % config.SCREENSHOT_INTERVAL == 0:
                ss_path = save_screenshot(annotated, frame_idx)
                print(f"  📸 Screenshot saved: {ss_path}")

            # ── Live preview ─────────────────────────────────────────
            if show:
                cv2.imshow("Tracking", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[Pipeline] Quit requested.")
                    break

            # ── Progress ─────────────────────────────────────────────
            frame_idx += 1
            if frame_idx % 100 == 0:
                elapsed = time.time() - t_start
                fps_proc = frame_idx / elapsed if elapsed > 0 else 0
                pct = (frame_idx / info["frame_count"] * 100
                       if info["frame_count"] > 0 else 0)
                print(f"  Frame {frame_idx}/{info['frame_count']}  "
                      f"({pct:.1f}%)  "
                      f"Tracks: {len(tracks)}  "
                      f"Speed: {fps_proc:.1f} FPS")

    except KeyboardInterrupt:
        print("\n[Pipeline] Interrupted by user.")

    finally:
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print()
    print(f"[Pipeline] Done – {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx / elapsed:.1f} FPS)")
    if skip_n > 1:
        print(f"[Pipeline] YOLO processed: {processed_count}/{frame_idx} frames "
              f"(skipped {frame_idx - processed_count})")
    print(f"[Pipeline] Output saved to: {output_path}")
    print(f"[Pipeline] Screenshots in : {config.SCREENSHOTS_DIR}")

    # ── Save Statistics ──────────────────────────────────────
    if track_counts:
        try:
            import matplotlib.pyplot as plt
            stats_path = os.path.join(config.OUTPUT_DIR, "stats.png")
            plt.figure(figsize=(10, 5))
            plt.plot(track_counts, label="Active Tracks", color="blue")
            plt.xlabel("Frame")
            plt.ylabel("Number of Detected Subjects")
            plt.title("Object Count Over Time")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(stats_path)
            plt.close()
            print(f"[Pipeline] Statistics   : saved to {stats_path}")
        except ImportError:
            print("[Pipeline] Failed to generate stats graph (matplotlib not installed)")


def main():
    print_banner()
    args = parse_args()

    # ── Step 1: Clean old outputs ────────────────────────────────────
    print("\n[Step 1/4] Cleaning previous outputs...")
    clean_previous_outputs()

    # ── Step 2: Get the video ────────────────────────────────────────
    print("\n[Step 2/4] Resolving video source...")
    source_path = resolve_source(args)

    # ── Step 3: Determine output path ────────────────────────────────
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(source_path))[0]
        output_path = os.path.join(config.OUTPUT_DIR, f"tracked_{base_name}.mp4")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ── Step 4: Run the pipeline ─────────────────────────────────────
    print(f"\n[Step 3/4] Processing video...")
    run_pipeline(source_path, output_path,
                 show=args.show, max_frames=args.max_frames)

    print(f"\n[Step 4/4] ✅ Complete!")
    print(f"  → Video : {output_path}")
    print(f"  → Screenshots : {config.SCREENSHOTS_DIR}")


if __name__ == "__main__":
    main()
