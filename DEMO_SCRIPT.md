# Demo Video Script (3–5 Minutes)

## Multi-Object Detection and Persistent ID Tracking

---

### 🎬 SLIDE 1: Title (0:00 – 0:15)

> **Say:** "Hi, I'm [Your Name]. In this demo, I'll walk you through my Multi-Object Detection and Persistent ID Tracking pipeline built for public sports and event footage."

---

### 🎬 SLIDE 2: Problem Statement (0:15 – 0:45)

> **Say:** "The goal is to detect multiple people — such as players in a sports video — and assign each person a unique, persistent ID that stays consistent across frames. This needs to handle real-world challenges like occlusion, motion blur, and camera movement."

*[Show a raw input video clip — highlight the challenges]*

---

### 🎬 SLIDE 3: Tech Stack & Architecture (0:45 – 1:30)

> **Say:** "My pipeline uses a tracking-by-detection approach with three components:
> 
> First, **YOLOv8** — a state-of-the-art object detector — finds all people in each frame.
> 
> Second, **ByteTrack** — a recent multi-object tracker — associates detections across frames and assigns persistent IDs. Its key innovation is using BOTH high-confidence and low-confidence detections, which is crucial for handling occlusion.
> 
> Third, **OpenCV** handles all video I/O and visualization."

*[Show the architecture diagram: Video → YOLOv8 → ByteTrack → Annotated Output]*

---

### 🎬 SLIDE 4: Code Walkthrough (1:30 – 2:30)

> **Say:** "Let me quickly walk through the code structure."
> 
> "**config.py** holds all tuneable parameters — detection thresholds, tracking buffer, model selection — all in one place."
> 
> "**tracker.py** wraps the Ultralytics model.track() API, which runs YOLOv8 detection and ByteTrack internally. It returns a list of tracks, each with a persistent ID, bounding box, and confidence score."
> 
> "**visualizer.py** draws coloured bounding boxes and ID labels. Each ID gets a deterministic colour so the same person always has the same colour."
> 
> "**main.py** orchestrates the pipeline — reads frames, tracks objects, draws annotations, and writes the output video."

*[Show each file briefly on screen]*

---

### 🎬 SLIDE 5: Live Demo (2:30 – 3:30)

> **Say:** "Now let me run the pipeline on a real sports video."

*[Run the command on screen:]*
```
python main.py --source input/video.mp4 --show
```

> **Point out:**
> - "You can see each player gets a unique coloured bounding box with their ID."
> - "Notice how ID 3 stays consistent even as the player moves across the frame."
> - "The HUD shows the frame count and number of active tracks."
> - "When a player goes behind another, ByteTrack's two-stage association maintains the correct ID."

---

### 🎬 SLIDE 6: Edge Case Handling (3:30 – 4:15)

> **Say:** "Let me highlight how the system handles edge cases:
> 
> **Occlusion:** ByteTrack uses low-confidence detections in its second association stage, so partially visible players keep their IDs.
> 
> **Fast motion:** The Kalman filter predicts positions, and the track buffer keeps IDs alive for up to 2 seconds even if detection fails temporarily.
> 
> **Scale changes:** YOLOv8's Feature Pyramid Network detects objects at multiple scales, handling players near and far from the camera."

*[Show specific frames demonstrating these scenarios]*

---

### 🎬 SLIDE 7: Results & Limitations (4:15 – 4:45)

> **Say:** "The pipeline processes video at about 20-30 FPS on CPU and over 100 FPS on GPU. It successfully tracks 10-20+ simultaneous players.
> 
> The main limitation is that after prolonged occlusion — more than about 2 seconds — a person may get a new ID, since ByteTrack uses spatial proximity rather than appearance features. This could be improved by switching to BoT-SORT, which adds Re-ID capabilities."

---

### 🎬 SLIDE 8: Conclusion (4:45 – 5:00)

> **Say:** "To summarize: this is a clean, modular, working pipeline for multi-object detection and tracking using YOLOv8 and ByteTrack. It uses only free, open-source tools with no model training required. Thank you."

---

## 🎤 Tips for Recording

1. **Use OBS Studio** (free) or Windows built-in screen recorder (Win+G).
2. **Resolution:** Record at 1080p.
3. **Speak clearly** and at a moderate pace.
4. **Show your screen** — terminal, code, and output video.
5. **Keep it under 5 minutes** — be concise.
