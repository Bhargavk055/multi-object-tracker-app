# Multi-Object Detection & Persistent ID Tracking

**Multi-Object Detection and Persistent ID Tracking in Public Sports/Event Footage**

A clean, modular Python pipeline that detects people in sports/event video footage using **YOLOv8** and maintains persistent identity tracking across frames using **ByteTrack**.

### Original Video Source
*Replace the link below with your chosen public sports/event video before submission:*
- **Source Link:** [https://youtube.com/shorts/F3SlklmUHWM](https://youtube.com/shorts/F3SlklmUHWM)

---

## 📸 Demonstration Videos

### 1. Localhost Web Interface Demo
https://github.com/user-attachments/assets/0825bb91-ebbd-4e25-88bc-702be9c2e3e7
*(Screen recording of the easy-to-use Streamlit Web App in action)*

### 2. High-Speed Pipeline Processing Demo
*(The tracking video you dropped originally didn't finish uploading before you clicked Commit! Please drag and drop `tracked_video (2).mp4` exactly right here, and WAIT for the "Uploading..." text to disappear before you hit Commit!)*

---

## Features & Optional Enhancements Implemented
-  **Multi-Object Tracking:** YOLOv8 + ByteTrack for handling occlusions and identity persistence.
-  **Trajectory Visualization:** Keeps a track history and draws movement tails for subjects.
-  **Object Count Over Time:** Automatically generates a line graph (`output/stats.png`) showing the number of detected subjects at each frame.
-  **Frame Skipping:** Configurable performance boost via `PROCESS_EVERY_N_FRAMES` to easily run on CPU.

---

##  Project Structure

```
AI assignment/
├── main.py              # Main pipeline script (entry point)
├── detector.py          # YOLOv8 detection module
├── tracker.py           # ByteTrack tracking module
├── visualizer.py        # Bounding box & ID drawing utilities
├── utils.py             # Video download, screenshots, helpers
├── config.py            # All tuneable hyper-parameters
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── TECHNICAL_REPORT.md  # Detailed technical report
├── input/               # Place input video files here
└── output/              # Annotated output videos + screenshots
    └── screenshots/     # Auto-captured frame screenshots
```

---

##  Installation

### Prerequisites

- **Python 3.9 – 3.11** (recommended: 3.10)
- **pip** (comes with Python)
- A GPU is optional but recommended for faster processing

### Step 1 – Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 2 – Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 detection + built-in ByteTrack |
| `opencv-python` | Video I/O, image processing |
| `numpy` | Array operations |
| `lap` | Linear assignment (used by ByteTrack) |
| `yt-dlp` | (Optional) Download videos from YouTube |

### Step 3 – Verify installation

```bash
python -c "from ultralytics import YOLO; print(' Ready')"
```

---

##  How to Run

### Option A: Easy Web Interface (Streamlit)
To launch the beautiful graphical user interface:
```bash
python -m streamlit run app.py
```
This will automatically open your web browser to `http://localhost:8501`.

### Option B: Command Line (Fastest)

**1. Process a local video:**
```bash
python main.py --source input/your_video.mp4
```

**2. Download directly from YouTube:**
```bash
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

**3. Auto-detect files:**
Place any `.mp4` / `.avi` / `.mkv` file in `input/` and just run:
```bash
python main.py
```

---

##  Output

After processing, you will find:

| Output | Location |
|---|---|
| Annotated video | `output/tracked_<name>.mp4` |
| Screenshots | `output/screenshots/frame_XXXXXX.png` |

Each frame in the output video shows:
- **Coloured bounding boxes** around each detected person
- **Persistent IDs** (e.g. `ID:3`) that stay consistent across frames
- **Confidence scores** for each detection
- **HUD overlay** with frame number and active track count

---

##  Configuration

All tuneable parameters are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `YOLO_MODEL` | `yolov8n.pt` | YOLOv8 variant (n/s/m/l/x) |
| `CONFIDENCE_THRESHOLD` | `0.25` | Min detection confidence |
| `TARGET_CLASSES` | `[0]` | COCO class IDs to detect (0 = person) |
| `TRACK_BUFFER` | `60` | Frames to keep lost tracks alive |
| `SCREENSHOT_INTERVAL` | `100` | Save screenshot every N frames |

---

##  Dependencies

- Python ≥ 3.9
- ultralytics ≥ 8.0.0
- opencv-python ≥ 4.8.0
- numpy ≥ 1.24.0
- lap ≥ 0.4.0
- yt-dlp ≥ 2023.0 (optional, for video download)

---

##  Assumptions

1. Input video contains **people/players** as the primary objects of interest.
2. Video is at a reasonable resolution (480p–1080p) and frame rate (15–60 FPS).
3. The pretrained YOLOv8 model (COCO dataset) provides sufficient accuracy for person detection without fine-tuning.
4. ByteTrack's motion-based tracking is adequate for sports footage where subjects move predictably.

---

## 📄 License

This project uses only **free and open-source** tools:
- YOLOv8: [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
- OpenCV: [Apache-2.0](https://opencv.org/license/)
- ByteTrack: [MIT](https://github.com/ifzhang/ByteTrack/blob/main/LICENSE)

For academic / assignment use only.
