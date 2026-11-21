"""
Audio Sensor Base Definitions for Arlo Indoor Camera System

Provides shared types and base class used by the audio pipeline.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import time


class SensorType(Enum):
    """Types of sensors in the system."""
    OPTICAL_CAMERA = "OPTICAL_CAMERA"
    MICROPHONE = "MICROPHONE"
    PIR_MOTION = "PIR_MOTION"
    AMBIENT_LIGHT = "AMBIENT_LIGHT"
    NIGHT_VISION_IR = "NIGHT_VISION_IR"
    RADAR = "RADAR"
    LIDAR = "LIDAR"
    THERMAL_LWIR = "THERMAL_LWIR"
    THERMAL_MWIR = "THERMAL_MWIR"
    THERMAL_SWIR = "THERMAL_SWIR"


class SensorState(Enum):
    """Truthful state of a sensor."""
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    SIMULATED = "SIMULATED"
    RECORDED = "RECORDED"
    ERROR = "ERROR"


@dataclass
class SensorReading:
    """A reading from a sensor."""
    sensor_id: str
    sensor_type: SensorType
    state: SensorState
    value: Any
    unit: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)