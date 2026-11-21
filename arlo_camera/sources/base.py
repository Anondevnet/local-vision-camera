"""
Camera Source Abstractions for Arlo Indoor Camera System

Defines the CameraSource base class and status enumeration.
"""

from enum import Enum
from typing import Any, Optional, Dict, Tuple
import numpy as np
import cv2


class CameraSourceStatus(str, Enum):
    """Truthful status of a camera source."""
    AVAILABLE = "AVAILABLE"       # Real hardware detected and streaming
    UNAVAILABLE = "UNAVAILABLE"   # Hardware physically absent
    SIMULATED = "SIMULATED"       # Synthetic/test-pattern generation
    RECORDED = "RECORDED"         # Playback from prerecorded video file


class CameraSource:
    """Abstract base class for camera video sources.

    Subclasses must implement:
    - read(): return dict with 'frame', 'timestamp', 'metadata'
    - read_with_detections(): optionally apply YOLO detection
    - start()/stop(): lifecycle control
    - get_status(): return CameraSourceStatus
    """

    def __init__(
        self,
        source_id: str,
        sensor_type: Any = None,
        status: CameraSourceStatus = CameraSourceStatus.UNAVAILABLE,
    ):
        self.source_id = source_id
        self.sensor_type = sensor_type
        self.status = status
        self._running = False

    def start(self):
        """Start the source (e.g., open file, init camera)."""
        self._running = True

    def stop(self):
        """Stop the source and release resources."""
        self._running = False

    def read(self) -> dict:
        """Return next frame.

        Returns dict with keys:
        - 'frame': numpy array (H, W, 3) BGR format, or None
        - 'timestamp': float seconds
        - 'metadata': dict with source identification
        """
        raise NotImplementedError("Subclasses must implement read()")

    def read_with_detections(self, yolo_model=None, confidence_threshold: float = 0.45) -> dict:
        """Read frame and apply YOLO-style detection if model provided.

        Returns dict with 'frame' and 'detections' list.
        """
        result = self.read()
        if result.get("frame") is None or yolo_model is None:
            result["detections"] = []
        return result

    def get_status(self) -> CameraSourceStatus:
        return self.status

    def set_status(self, status: CameraSourceStatus):
        self.status = status