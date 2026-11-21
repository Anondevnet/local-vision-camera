"""
Camera Source Abstractions for Arlo Indoor Camera System

Provides modular CameraSource backends:
- Simulated (test-pattern/synthetic frames)
- Prerecorded video files
- Future: Live Arlo RTSP/HTTP stream integration
"""

from .base import CameraSource, CameraSourceStatus
from .simulated import SimulatedCameraSource
from .prerecorded import PrerecordedCameraSource

__all__ = [
    "CameraSource",
    "CameraSourceStatus",
    "SimulatedCameraSource",
    "PrerecordedCameraSource",
]