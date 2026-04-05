"""
detector.py
-----------
Detection module – wraps Ultralytics YOLOv8 for person detection.
"""

from ultralytics import YOLO
import config


class Detector:
    """Thin wrapper around YOLOv8 that filters for target classes."""

    def __init__(self, model_path: str = config.YOLO_MODEL):
        """Load the pretrained YOLOv8 model (auto-downloads if needed)."""
        self.model = YOLO(model_path)
        print(f"[Detector] Loaded model: {model_path}")

    def detect(self, frame):
        """
        Run inference on a single BGR frame.

        Returns
        -------
        results : ultralytics.engine.results.Results
            Raw Ultralytics result object (used downstream by the tracker).
        """
        results = self.model.predict(
            source=frame,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            classes=config.TARGET_CLASSES,
            verbose=False,
        )
        return results[0]  # single image → single result
