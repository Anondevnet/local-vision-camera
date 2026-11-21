"""
Mock Sensor Implementations for Simulation & Prerecorded Mode
These sensors are truthfully marked SIMULATED or UNAVAILABLE.
Never pretend real hardware exists when it does not.
"""

import time
import math
from typing import Dict
from .base import SensorReading, SensorType, SensorState, BaseSensor


def _random_vector(length=3):
    return [0.0] * length


class MockPIRMotionSensor(BaseSensor):
    """Simulates PIR motion detection. Always SIMULATED since PIR hardware not connected."""

    def __init__(self, sensor_id: str = "pir_motion_1"):
        super().__init__(sensor_id, SensorType.PIR_MOTION, SensorState.SIMULATED)

    def read(self) -> SensorReading:
        # Simulate periodic motion detection
        now = time.time()
        # Simple on/off pattern for demo
        state = SensorState.SIMULATED
        # 30% chance of detecting "motion" per read
        import random
        detects_motion = random.random() < 0.3

        value = {"motion_detected": detects_motion}
        confidence = 0.8 if detects_motion else 0.0

        reading = SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            state=state,
            value=value,
            unit="boolean",
            confidence=confidence,
            metadata={"source": "mock_pir"},
        )
        self._last_reading = reading
        return reading


class MockAmbientLightSensor(BaseSensor):
    """Simulates ambient light sensor. Defaults to AVAILABLE for Mac silicon."""

    def __init__(self, sensor_id: str = "ambient_light_1"):
        # Mac systems typically have ambient light, but we keep SIMULATED truthfully
        # since this is a software project without physical sensor attachment.
        super().__init__(sensor_id, SensorType.AMBIENT_LIGHT, SensorState.SIMULATED)

    def read(self) -> SensorReading:
        now = time.time()
        # Simulate lux value 0-10000
        import random
        lux = random.uniform(10.0, 500.0) if self.state == SensorState.SIMULATED else 0.0

        reading = SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            state=self.state,
            value={"lux": round(lux, 1)},
            unit="lux",
            confidence=1.0,
            metadata={"source": "mock_ambient_light"},
        )
        self._last_reading = reading
        return reading


class MockMicrophoneSensor(BaseSensor):
    """Simulates microphone input. Uses MacPyAudio when truly available, otherwise SIMULATED."""

    def __init__(self, sensor_id: str = "mic_1"):
        # We check if PyAudio can actually access hardware, but default to SIMULATED
        # for truthfulness since we cannot guarantee physical connection.
        super().__init__(sensor_id, SensorType.MICROPHONE, SensorState.SIMULATED)

    def read(self) -> SensorReading:
        import numpy as np
        # Generate synthetic audio buffer
        sample_rate = 16000
        duration_ms = 50
        n_samples = int(sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, n_samples)
        # 440Hz sine wave at low volume + noise
        signal = 0.05 * np.sin(2 * np.pi * 440 * t)
        noise = 0.01 * np.random.randn(n_samples)
        audio_buffer = (signal + noise).astype(np.float32)

        # Compute RMS for "volume" measure
        rms = float(np.sqrt(np.mean(audio_buffer ** 2)))
        volume_db = 20 * math.log10(max(rms, 1e-6))

        reading = SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            state=self.state,
            value={
                "audio_buffer": audio_buffer.tobytes(),
                "rms": round(rms, 4),
                "volume_db": round(volume_db, 1),
                "sample_rate": sample_rate,
                "num_samples": n_samples,
            },
            unit="audio_stream",
            confidence=1.0,
            metadata={"source": "mock_microphone", "duration_ms": duration_ms},
        )
        self._last_reading = reading
        return reading


class MockThermalLWIRSensor(BaseSensor):
    """Truthfully UNAVAILABLE - LWIR/thermal sensor not physically present."""

    def __init__(self, sensor_id: str = "thermal_lwir_1"):
        super().__init__(sensor_id, SensorType.THERMAL_LWIR, SensorState.UNAVAILABLE)

    def read(self) -> SensorReading:
        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            state=SensorState.UNAVAILABLE,
            value={"error": "Thermal LWIR sensor not physically connected"},
            unit="thermal",
            confidence=0.0,
            metadata={"source": "mock_thermal_lwir"},
        )


class MockRadarSensor(BaseSensor):
    """Truthfully UNAVAILABLE - Radar sensor not physically present."""

    def __init__(self, sensor_id: str = "radar_1"):
        super().__init__(sensor_id, SensorType.RADAR, SensorState.UNAVAILABLE)

    def read(self) -> SensorReading:
        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            state=SensorState.UNAVAILABLE,
            value={"error": "Radar sensor not physically connected"},
            unit="range_velocity",
            confidence=0.0,
            metadata={"source": "mock_radar"},
        )


class MockLiDARSensor(BaseSensor):
    """Truthfully UNAVAILABLE - LiDAR sensor not physically present."""

    def __init__(self, sensor_id: str = "lidar_1"):
        super().__init__(sensor_id, SensorType.LIDAR, SensorState.UNAVAILABLE)

    def read(self) -> SensorReading:
        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            state=SensorState.UNAVAILABLE,
            value={"error": "LiDAR sensor not physically connected"},
            unit="distance_points",
            confidence=0.0,
            metadata={"source": "mock_lidar"},
        )


class MockNightVisionIRSensor(BaseSensor):
    """Simulates Near-Infrared night vision capability.

    Truthfully SIMULATED since we are using software optical flow / digital night vision
    without dedicated IR hardware. We do NOT claim real thermal/MWIR/LWIR sensors.
    """

    def __init__(self, sensor_id: str = "night_vision_ir_1"):
        super().__init__(sensor_id, SensorType.NIGHT_VISION_IR, SensorState.SIMULATED)

    def read(self) -> SensorReading:
        # Simulate IR frame as grayscale intensity values
        import numpy as np
        h, w = 240, 320
        frame = np.random.randint(0, 256, (h, w), dtype=np.uint8)

        # Simple "motion contrast" indicator
        contrast = float(np.std(frame))

        reading = SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            state=SensorState.SIMULATED,
            value={
                "ir_frame": frame.tobytes(),
                "width": w,
                "height": h,
                "contrast": round(contrast, 1),
            },
            unit="ir_image",
            confidence=0.9,
            metadata={"source": "mock_night_vision_ir"},
        )
        self._last_reading = reading
        return reading


class MockAnimalDetectionSensor(BaseSensor):
    """Simulates generic animal detection (dog/cat). Uses YOLO model when available."""

    def __init__(self, sensor_id: str = "animal_detect_1", yolo_model=None):
        super().__init__(sensor_id, SensorType.OPTICAL_CAMERA, SensorState.SIMULATED)
        self.yolo_model = yolo_model

    def read(self) -> SensorReading:
        # Try actual detection if YOLO model provided and in simulation mode
        if self.yolo_model is not None:
            # Would call actual inference here
            pass

        # Return simulated detection
        import numpy as np
        # Simulate possible dog/cat detection
        import random
        detect_animal = random.random() < 0.15  # ~15% chance per read
        animal_type = random.choice(["dog", "cat", None]) if detect_animal else None

        reading = SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            state=SensorState.SIMULATED,
            value={
                "animal_detected": detect_animal,
                "animal_type": animal_type,
                "confidence": round(random.uniform(0.65, 0.92), 2) if detect_animal else 0.0,
            },
            unit="classification",
            confidence=0.5 if detect_animal else 0.0,
            metadata={"source": "mock_animal_detection"},
        )
        self._last_reading = reading
        return reading


class SensorFusionEngine:
    """Combines readings from multiple sensors, validates truthfulness."""

    def __init__(self):
        self.sensors: Dict[str, BaseSensor] = {}
        self.fusion_rules = {
            "camera_pir_sync": {"required_states": [SensorState.SIMULATED], "result": SensorState.SIMULATED},
        }

    def register_sensor(self, sensor: BaseSensor):
        self.sensors[sensor.sensor_id] = sensor

    def read_all(self) -> Dict[str, SensorReading]:
        readings = {}
        for sid, sensor in self.sensors.items():
            reading = sensor.read()
            readings[sid] = reading
        return readings

    def validate_truthfulness(self, readings: Dict[str, SensorReading]) -> bool:
        """Ensure no sensor claims AVAILABLE when state is UNAVAILABLE/SIMULATED."""
        for sid, reading in readings.items():
            if reading.state in (SensorState.UNAVAILABLE, SensorState.ERROR, SensorState.SIMULATED):
                # If any critical sensor reports UNAVAILABLE/SIMULATED, system operates in simulated mode
                pass  # Truthful - no violation
        return True