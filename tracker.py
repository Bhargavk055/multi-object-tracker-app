"""
tracker.py
----------
Tracking module – uses Ultralytics' built-in ByteTrack integration so we
get persistent IDs without any external library beyond `lap`.

The Ultralytics `model.track()` API handles:
  • Kalman-filter based motion prediction
  • Two-stage IoU association (high + low confidence)
  • Track lifecycle management (new / active / lost / removed)
"""

import os
import tempfile
import yaml

from ultralytics import YOLO
import config


def _get_tracker_config() -> str:
    """
    Return the path to the tracker config YAML.
    
    Uses the built-in config shipped with ultralytics to ensure all
    required keys are present across versions.
    """
    import ultralytics
    base = os.path.dirname(ultralytics.__file__)
    builtin = os.path.join(base, "cfg", "trackers", f"{config.TRACKER_TYPE}.yaml")
    if os.path.isfile(builtin):
        return builtin

    # Fallback: write a complete config manually
    cfg = {
        "tracker_type": config.TRACKER_TYPE,
        "track_high_thresh": config.TRACK_HIGH_THRESH,
        "track_low_thresh": config.TRACK_LOW_THRESH,
        "new_track_thresh": config.TRACK_NEW_THRESH,
        "track_buffer": config.TRACK_BUFFER,
        "match_thresh": config.MATCH_THRESH,
        "fuse_score": True,
        "gmc_method": "sparseOptFlow",
        "proximity_thresh": 0.5,
        "appearance_thresh": 0.25,
    }
    path = os.path.join(tempfile.gettempdir(), "tracker_config.yaml")
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return path


class Tracker:
    """
    Wraps Ultralytics model.track() for frame-by-frame tracking.
    Internally uses ByteTrack (default) or BoT-SORT.
    """

    def __init__(self, model_path: str = config.YOLO_MODEL):
        self.model = YOLO(model_path)
        self.tracker_cfg = _get_tracker_config()
        print(f"[Tracker] Ready  – algorithm: {config.TRACKER_TYPE}")

    def update(self, frame):
        """
        Detect + track in one call.

        Returns
        -------
        tracks : list[dict]
            Each dict has keys:
              id   – persistent track ID (int)
              bbox – [x1, y1, x2, y2] in pixel coords
              conf – detection confidence
              cls  – class index
        """
        results = self.model.track(
            source=frame,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            classes=config.TARGET_CLASSES,
            tracker=self.tracker_cfg,
            persist=True,
            verbose=False,
        )

        tracks = []
        result = results[0]

        if result.boxes is None or result.boxes.id is None:
            return tracks

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for bbox, tid, conf, cls in zip(boxes, ids, confs, classes):
            tracks.append({
                "id": int(tid),
                "bbox": bbox.tolist(),   # [x1, y1, x2, y2]
                "conf": float(conf),
                "cls": int(cls),
            })

        return tracks
