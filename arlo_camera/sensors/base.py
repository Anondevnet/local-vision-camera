"""
Base Sensor Interface and Truthful State Management
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import time

class SensorState(str, Enum):
    AVAILABLE = "AVAILABLE"       # Real hardware detected and operational
    UNAVAILABLE = "UNAVAILABLE"   # Hardware physically absent or disconnected
    SIMULATED = "SIMULATED"       # Synthetic / mock sensor stream
    RECORDED = "RECORDED"         # Playback from recorded telemetry
    ERROR = "ERROR"               # Hardware error or fault condition

class SensorType(str, Enum):
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

@dataclass
class SensorReading:
    sensor_id: str
    sensor_type: SensorType
    state: SensorState
    value: Any
    unit: str
    confidence: float  # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseSensor:
    def __init__(self, sensor_id: str, sensor_type: SensorType, state: SensorState = SensorState.UNAVAILABLE):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.state = state
        self._last_reading: Optional[SensorReading] = None

    def read(self) -> SensorReading:
        raise NotImplementedError("Subclasses must implement read()")

    def get_status(self) -> Dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type.value,
            "state": self.state.value,
            "last_reading": self._last_reading
        }
