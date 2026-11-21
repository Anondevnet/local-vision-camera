"""
Simulated Camera Source - Test Pattern / Synthetic Frame Generation

Generates synthetic video frames for development, unit testing, and
initial system validation before physical camera integration.

Truthfully identifies as SIMULATED. Never claims real hardware.
"""

import math
import time
import numpy as np
import cv2
from typing import Any
from .base import CameraSource, CameraSourceStatus
from arlo_camera.sensors.base import SensorType, SensorState


class SimulatedCameraSource(CameraSource):
    """Generates synthetic test-pattern video frames.

    Features:
    - Configurable resolution and FPS
    - Moving test patterns (checkerboard, circles, gradient)
    - Synthetic object placement (simulated people/vehicles)
    - Configurable motion/transition control
    """

    def __init__(
        self,
        source_id: str = "simulated_arlo_1",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        model_config=None,
    ):
        super().__init__(
            source_id=source_id,
            sensor_type=SensorType.OPTICAL_CAMERA,
            status=CameraSourceStatus.SIMULATED,
        )
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0
        self.model_config = model_config  # Reference to YOLO config, etc.
        self._running = False

    def start(self):
        self._running = True
        self.frame_count = 0

    def stop(self):
        self._running = False

    def read(self) -> dict:
        """Generate next synthetic frame.

        Returns dict with:
        - 'frame': numpy array (H, W, 3) BGR format
        - 'timestamp': float seconds
        - 'metadata': dict with source identification
        """
        if not self._running:
            self.start()

        self.frame_count += 1

        self.frame_count += 1
        t = self.frame_count / self.fps

        # Create test pattern frame (gradient background)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Vertical gradient
        for y in range(self.height):
            for x in range(self.width):
                frame[y, x] = [
                    int(x * 255 / self.width),
                    int(y * 255 / self.height),
                    int((x + y) * 128 / (self.width + self.height)),
                ]

        # Add moving synthetic "person" bounding box
        box_size = 80
        person_x = int(400 + 300 * math.sin(t * 0.5))  # oscillate
        person_y = int(300 + 200 * math.cos(t * 0.7))
        person_x = max(0, min(person_x, self.width - box_size))
        person_y = max(0, min(person_y, self.height - box_size))

        # Draw person rectangle (BGR color: green)
        cv2_rect = np.array([
            [person_x, person_y],
            [person_x + box_size, person_y],
            [person_x + box_size, person_y + box_size],
            [person_x, person_y + box_size],
        ], dtype=np.int32)
        # Fill with solid color
        cv2_roi = frame[person_y:person_y + box_size, person_x:person_x + box_size]
        cv2_roi[:] = (0, 255, 0)  # BGR green

        # Draw a smaller "head" rectangle
        head_size = 25
        head_x = person_x + box_size // 2 - head_size // 2
        head_y = person_y + 15
        frame[head_y:head_y + head_size, head_x:head_x + head_size] = (0, 0, 255)  # BGR blue

        # Add synthetic vehicle (car) in another area
        vehicle_size = 100
        vehicle_x = int(800 + 200 * math.sin(t * 0.3))
        vehicle_y = int(400 + 150 * math.cos(t * 0.4))
        vehicle_x = max(0, min(vehicle_x, self.width - vehicle_size))
        vehicle_y = max(0, min(vehicle_y, self.height - vehicle_size))
        frame[vehicle_y:vehicle_y + vehicle_size, vehicle_x:vehicle_x + vehicle_size // 2] = (0, 165, 255)  # BGR orange car

        # Add timestamp/text
        cv2_putText = getattr(cv2, "putText", None)
        if cv2_putText:
            cv2_putText(
                frame,
                f"SIMULATED {self.source_id} Frame {self.frame_count}",
                (20, self.height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        timestamp = time.time()

        return {
            "frame": frame,
            "timestamp": timestamp,
            "metadata": {
                "source_id": self.source_id,
                "frame_count": self.frame_count,
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "simulated": True,
                "pattern": "gradient_with_moving_objects",
            },
        }

    def read_with_detections(self, yolo_model=None, confidence_threshold: float = 0.45) -> dict:
        """Read frame and apply YOLO-style detection if model provided.

        Returns dict with frame + detection results.
        """
        result = self.read()
        frame = result["frame"]
        if frame is None or yolo_model is None:
            result["detections"] = []
            return result

        # Convert BGR to RGB for model if needed
        import torch
        if hasattr(yolo_model, "predict"):
            # Ultralytics YOLO model
            try:
                results = yolo_model.predict(source=frame, conf=confidence_threshold, verbose=False)
                detections = []
                for result_obj in results:
                    boxes = result_obj.boxes
                    if boxes is not None:
                        for box in boxes:
                            xyxy = box.xyxy[0].tolist()
                            conf = float(box.conf[0].item())
                            cls_id = int(box.cls[0].item())
                            cls_name = yolo_model.names.get(cls_id, f"class_{cls_id}")
                            detections.append({
                                "bbox": xyxy,  # [x1, y1, x2, y2]
                                "confidence": round(conf, 3),
                                "class_id": cls_id,
                                "class_name": cls_name,
                                "track_id": None,  # Would be assigned by tracker
                            })
                result["detections"] = detections
            except Exception as e:
                # If inference fails, return empty detections but truthful error
                result["detections"] = []
                result["metadata"]["inference_error"] = str(e)
        else:
            # No model provided - return synthetic mock detections
            result["detections"] = self._synthetic_detections(frame)

        return result

    def _synthetic_detections(self, frame) -> list:
        """Generate synthetic detection results when no YOLO model is loaded."""
        import random
        detections = []

        # Sometimes detect a person
        if random.random() < 0.4:
            # Random box in frame
            w, h = self.width, self.height
            box_w = random.randint(50, 150)
            box_h = random.randint(100, 200)
            x1 = random.randint(0, w - box_w)
            y1 = random.randint(0, h - box_h - 100)  # keep mostly bottom
            x2, y2 = x1 + box_w, y1 + box_h

            conf = round(random.uniform(0.65, 0.95), 3)
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": 0,  # person COCO class
                "class_name": "person",
                "track_id": random.randint(1, 999),
            })

        # Sometimes detect a vehicle
        if random.random() < 0.3:
            w, h = self.width, self.height
            box_w = random.randint(80, 200)
            box_h = random.randint(40, 100)
            x1 = random.randint(0, w - box_w)
            y1 = random.randint(0, h // 2 - box_h)
            x2, y2 = x1 + box_w, y1 + box_h
            conf = round(random.uniform(0.7, 0.98), 3)
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": 2,  # car COCO class
                "class_name": "car",
                "track_id": random.randint(1, 999),
            })

        # Sometimes detect a dog/cat
        if random.random() < 0.2:
            w, h = self.width, self.height
            box_w = random.randint(40, 80)
            box_h = random.randint(40, 80)
            x1 = random.randint(0, w - box_w)
            y1 = random.randint(50, h // 2)
            x2, y2 = x1 + box_w, y1 + box_h
            animal = random.choice(["dog", "cat"])
            class_id = 16 if animal == "dog" else 15  # COCO: dog=16, cat=15
            conf = round(random.uniform(0.6, 0.92), 3)
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": class_id,
                "class_name": animal,
                "track_id": random.randint(1, 999),
            })

        return detections

    def get_status(self) -> CameraSourceStatus:
        return self.status

    def set_status(self, status: CameraSourceStatus):
        self.status = status


# Convenience function for quick startup
def create_simulated_source(
    source_id: str = "simulated_arlo_1",
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
) -> SimulatedCameraSource:
    """Create a ready-to-use SimulatedCameraSource instance."""
    return SimulatedCameraSource(
        source_id=source_id,
        width=width,
        height=height,
        fps=fps,
    )