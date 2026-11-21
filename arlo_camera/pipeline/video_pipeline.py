"""
Video Processing Pipeline for Arlo Indoor Camera System

Handles:
- Frame ingestion from CameraSource abstractions (Simulated, Prerecorded, Live)
- YOLO object detection (Ultralytics)
- Centroid tracking across frames
- Zone monitoring (entry/exit, dwell time, loitering)
- Direction of movement analysis
- Bounding box computation and normalization
"""

import time
import numpy as np
import cv2
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

# ---------------------------------------------------------------------------
# COCO class index for YOLO models (Ultralytics/yolov8n.pt)
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 4: airplane,
# 5: bus, 6: train, 7: truck, 8: boat, 9: traffic light,
# 10: fire hydrant, 11: stop sign, 13: bird, 14: cat,
# 15: dog, 16: horse, 17: sheep, 18: cow, 19: elephant,
# 20: bear, 21: zebra, 22: giraffe, 23: backpack,
# 24: umbrella, 31: handbag, 32: tie, 33: suitcase,
# 34: frisbee, 35: skis, 36: snowboard, 37: sports ball,
# 38: kite, 39: baseball bat, 40: baseball glove,
# 41: skateboard, 42: surfboard, 43: tennis racket,
# 44: bottle, 46: cup, 47: fork, 48: knife, 49: spoon,
# 50: bowl, 51: banana, 52: apple, 53: sandwich,
# 54: orange, 55: broccoli, 56: carrot, 57: hot dog,
# 58: pizza, 59: donut, 60: cake, 61: chair,
# 62: couch, 63: potted plant, 64: bed, 65: dining table,
# 66: toilet, 67: tv, 70: laptop, 72: mouse,
# 73: remote, 74: keyboard, 75: cell phone,
# 76: microwave, 77: oven, 78: toaster, 79: sink,
# 81: fridge, 82: book, 83: clock, 84: vase,
# 85: scissors, 86: teddy bear, 87: hair drier,
# 88: toothbrush
# ---------------------------------------------------------------------------

# Tracked classes of interest for Arlo indoor camera
TRACKED_CLASSES = {
    0: "person",    # human
    1: "bicycle",   # cycle
    2: "car",       # vehicle
    3: "motorcycle",
    5: "bus",
    7: "truck",
    15: "cat",      # animal
    16: "dog",      # animal
}


class BoundingBox:
    """Represents a normalized or pixel bounding box with confidence."""

    def __init__(self, x1: float, y1: float, x2: float, y2: float,
                 confidence: float, class_id: int, class_name: str):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.confidence = float(confidence)
        self.class_id = int(class_id)
        self.class_name = class_name

        # Derived properties
        self.width = max(0.0, self.x2 - self.x1)
        self.height = max(0.0, self.y2 - self.y1)
        self.area = self.width * self.height

        # Center point
        self.center_x = (self.x1 + self.x2) / 2.0
        self.center_y = (self.y1 + self.y2) / 2.0

    def iou(self, other: 'BoundingBox') -> float:
        """Compute Intersection-over-Union with another BoundingBox."""
        # Coordinates of intersection rectangle
        x1_i = max(self.x1, other.x1)
        y1_i = max(self.y1, other.y1)
        x2_i = min(self.x2, other.x2)
        y2_i = min(self.y2, other.y2)

        # Intersection area
        inter_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)

        # Union area
        union_area = self.area + other.area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_normalized(self, frame_width: int, frame_height: int) -> 'BoundingBox':
        """Return normalized coordinates (0.0 - 1.0)."""
        return BoundingBox(
            self.x1 / frame_width,
            self.y1 / frame_height,
            self.x2 / frame_width,
            self.y2 / frame_height,
            self.confidence,
            self.class_id,
            self.class_name,
        )


class TrackedObject:
    """Represents a tracked object with ID, history, and state."""

    def __init__(self, obj_id: int, bbox: BoundingBox, class_name: str,
                 class_id: int, confidence: float):
        self.obj_id = obj_id
        self.bbox = bbox
        self.class_name = class_name
        self.class_id = class_id
        self.confidence = confidence

        # Tracking state
        self.age = 0  # frames since last detection
        self.total_frames = 0
        self.disappeared = 0

        # History for trajectory / direction
        self.centroid_history: List[Tuple[float, float]] = []
        self.max_history = 30

        # Zone / dwell state
        self.zone_enter_time: Optional[float] = None
        self.dwell_time = 0.0
        self.in_zone: bool = False

        # Movement state
        self.last_centroid: Optional[Tuple[float, float]] = None
        self.direction: str = "stationary"  # "entering", "exiting", "stationary", "unknown"
        self.last_move_time: float = time.time()

    def update(self, bbox: BoundingBox, timestamp: float):
        """Update tracked object with new detection."""
        self.bbox = bbox
        self.confidence = bbox.confidence
        self.age = 0
        self.total_frames += 1

        # Compute centroid
        cx = (bbox.x1 + bbox.x2) / 2.0
        cy = (bbox.y1 + bbox.y2) / 2.0
        self.centroid_history.append((cx, cy))
        if len(self.centroid_history) > self.max_history:
            self.centroid_history.pop(0)
        self.last_centroid = (cx, cy)

        # Update dwell time if in zone
        # (handled by zone monitor, but we track age here)

        # Determine movement direction
        if self.last_centroid and len(self.centroid_history) >= 3:
            self._compute_direction()

    def _compute_direction(self):
        """Simple direction computation based on centroid trajectory."""
        if len(self.centroid_history) < 3:
            self.direction = "stationary"
            return

        # Compare first vs last centroid in history
        start = self.centroid_history[0]
        end = self.centroid_history[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        # Determine primary direction
        if abs(dx) > abs(dy):
            if dx > 10:
                self.direction = "rightward"
            elif dx < -10:
                self.direction = "leftward"
            else:
                self.direction = "stationary"
        else:
            if dy > 10:
                self.direction = "downward"
            elif dy < -10:
                self.direction = "upward"
            else:
                self.direction = "stationary"

    def get_state(self) -> Dict[str, Any]:
        """Return serializable state for storage/alerts."""
        return {
            "obj_id": self.obj_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox.to_tuple(),
            "center": (self.last_centroid or (0, 0)),
            "direction": self.direction,
            "age": self.age,
            "total_frames": self.total_frames,
            "in_zone": self.in_zone,
            "dwell_time": round(self.dwell_time, 2),
        }


class CentroidTracker:
    """Simple centroid-based object tracker.

    Assigns object IDs and maintains tracking state across frames.
    Uses IOU-based matching for robust re-identification.
    """

    def __init__(self, max_disappeared: int = 30, max_distance: float = 50.0):
        self.next_object_id = 1
        self.objects: Dict[int, TrackedObject] = {}
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance  # pixels for matching

    def register(self, bbox: BoundingBox, tracked_obj: TrackedObject):
        """Register a new tracked object."""
        self.objects[tracked_obj.obj_id] = tracked_obj
        self.disappeared[tracked_obj.obj_id] = 0

    def deregister(self, obj_id: int):
        """Remove a tracked object."""
        del self.objects[obj_id]
        del self.disappeared[obj_id]

    def update(self, bboxes: List[BoundingBox], timestamp: float) -> Dict[int, TrackedObject]:
        """Update tracker with new detections.

        Returns dict of {obj_id: TrackedObject} for all active tracks.
        """
        # If no detections, mark all existing as disappeared
        if len(bboxes) == 0:
            # Update disappeared count for all objects
            for obj_id in self.objects.keys():
                self.disappeared[obj_id] += 1

                if self.disappeared[obj_id] >= self.max_disappeared:
                    self.deregister(obj_id)

            # Return remaining objects (some may just be fading out)
            return {oid: obj for oid, obj in self.objects.items() if self.disappeared[oid] < self.max_disappeared}

        # Keep track of which objects have been matched
        used_object_ids = set()

        # If we have fewer objects than detections, update existing
        if len(self.objects) == 0:
            # All detections are new
            for i, bbox in enumerate(bboxes):
                obj = TrackedObject(
                    obj_id=self.next_object_id + i,
                    bbox=bbox,
                    class_name=self._class_name_from_bbox(bbox),
                    class_id=bbox.class_id,
                    confidence=bbox.confidence,
                )
                self.next_object_id += i + 1
                self.objects[obj.obj_id] = obj
                self.disappeared[obj.obj_id] = 0
            return self.objects

        # Compute distance matrix between existing objects and new detections
        object_ids = list(self.objects.keys())
        n_objects = len(object_ids)
        n_detections = len(bboxes)

        D = np.zeros((n_objects, n_detections), dtype=float)

        for i, obj_id in enumerate(object_ids):
            existing_obj = self.objects[obj_id]
            existing_cx = (existing_obj.bbox.x1 + existing_obj.bbox.x2) / 2.0
            existing_cy = (existing_obj.bbox.y1 + existing_obj.bbox.y2) / 2.0

            for j, bbox in enumerate(bboxes):
                detection_cx = (bbox.x1 + bbox.x2) / 2.0
                detection_cy = (bbox.y1 + bbox.y2) / 2.0

                # Euclidean distance between centroids
                D[i, j] = np.sqrt((detection_cx - existing_cx) ** 2 + (detection_cy - existing_cy) ** 2)

        # Each row's minimum can be used to find best match
        # Apply Hungarian-like assignment via linear scan (simple greedy)
        used_detection_indices = set()

        # Sort rows by minimum distance
        rows = D.argsort(axis=1)

        for row_idx in range(n_objects):
            obj_id = object_ids[row_idx]
            best_col = None
            best_dist = self.max_distance + 1  # default: too far

            for col_idx in rows[row_idx]:
                if col_idx in used_detection_indices:
                    continue
                if D[row_idx, col_idx] < best_dist:
                    best_dist = D[row_idx, col_idx]
                    best_col = col_idx

            if best_col is not None and best_dist <= self.max_distance:
                # Match found
                bbox = bboxes[best_col]
                self.objects[obj_id].update(bbox, timestamp)
                used_detection_indices.add(best_col)
                self.disappeared[obj_id] = 0
            else:
                # No good match; mark as disappeared
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] >= self.max_disappeared:
                    self.deregister(obj_id)

        # Register any detections not yet matched
        for j, bbox in enumerate(bboxes):
            if j not in used_detection_indices:
                obj = TrackedObject(
                    obj_id=self.next_object_id,
                    bbox=bbox,
                    class_name=self._class_name_from_bbox(bbox),
                    class_id=bbox.class_id,
                    confidence=bbox.confidence,
                )
                self.next_object_id += 1
                self.objects[obj.obj_id] = obj
                self.disappeared[obj.obj_id] = 0

        # Clean up deregistered objects
        for obj_id in list(self.objects.keys()):
            if self.disappeared.get(obj_id, 0) >= self.max_disappeared:
                self.deregister(obj_id)

        return self.objects

    def _class_name_from_bbox(self, bbox: BoundingBox) -> str:
        """Guess class name from bbox characteristics (fallback)."""
        area = bbox.area
        if area > 5000:
            return "person"  # rough heuristic
        elif area > 1000:
            return "car"
        return "object"


class ZoneMonitor:
    """Monitors tracked objects within configurable zones.

    Features:
    - Point-in-zone test (polygon)
    - Entry/exit detection
    - Dwell time tracking
    - Loitering detection
    - Tripwire crossing (directional)
    """

    def __init__(self, zones: List[Dict[str, Any]] = None):
        # zones: list of dicts with "name", "points" (normalized coords), "zone_type"
        self.zones = zones or []
        self.zone_polygons: List[Tuple[Tuple[float, float], ...]] = []

        # Build polygon paths from normalized points
        for zone in self.zones:
            points = zone.get("points", [])
            # Convert normalized (x,y) pixel coordinates based on current frame size
            # We'll apply at monitoring time
            self.zone_polygons.append(tuple(points))

        # Loitering config
        self.loiter_threshold_seconds = 8.0

        # Per-object loiter tracking
        self.object_loiter_start: Dict[int, float] = {}

    def _point_in_polygon(self, point: Tuple[float, float],
                          polygon: Tuple[Tuple[float, float], ...]) -> bool:
        """Ray-casting point-in-polygon test."""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y + 1e-10) + p1x
                    if p1x == xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

        return inside

    def _point_in_zone(self, centroid: Tuple[float, float],
                       frame_width: int, frame_height: int,
                       zone: Dict[str, Any]) -> bool:
        """Check if centroid is inside a zone polygon.

        zone: dict with "points" as normalized (0-1) coordinates and "frame_size"
        """
        points = zone.get("points", [])
        zone_type = zone.get("zone_type", "detection")

        # Convert normalized points to pixel coordinates
        px_points = [(int(p[0] * frame_width), int(p[1] * frame_height)) for p in points]
        if len(px_points) < 3:
            return False

        # Ray casting
        px, py = int(centroid[0]), int(centroid[1])
        inside = False
        for i in range(len(px_points)):
            x1, y1 = px_points[i]
            x2, y2 = px_points[(i + 1) % len(px_points)]
            if ((y1 > py) != (y2 > py)):
                xinters = (py - y1) * (x2 - x1) / (y2 - y1 + 1e-10) + x1
                if px < xinters:
                    inside = not inside
        return inside

    def check_zone(self, obj_id: int, centroid: Tuple[float, float],
                   frame_width: int, frame_height: int,
                   zone: Dict[str, Any]) -> Dict[str, Any]:
        """Check object position within a zone and return state changes."""
        zone_name = zone.get("name", "unknown")
        zone_type = zone.get("zone_type", "detection")

        was_in = self.object_loiter_start.get(obj_id) is not None
        is_in = self._point_in_zone(centroid, frame_width, frame_height, zone)

        result = {
            "zone_name": zone_name,
            "zone_type": zone_type,
            "obj_id": obj_id,
            "in_zone": is_in,
            "entered": False,
            "exited": False,
            "dwell_time": 0.0,
            "loitering": False,
        }

        if is_in and not was_in:
            # Object entered zone
            result["entered"] = True
            self.object_loiter_start[obj_id] = time.time()
            result["dwell_time"] = 0.0

        elif not is_in and was_in:
            # Object exited zone
            result["exited"] = True
            start_time = self.object_loiter_start.pop(obj_id, None)
            if start_time is not None:
                self.object_loiter_start.pop(obj_id, None)  # just in case
                dwell = time.time() - start_time
                result["dwell_time"] = round(dwell, 2)

                # Check for loitering
                if dwell >= self.loiter_threshold_seconds:
                    result["loitering"] = True

        elif is_in and was_in:
            # Object still in zone - update dwell time
            start_time = self.object_loiter_start.get(obj_id)
            if start_time is not None:
                dwell = time.time() - start_time
                result["dwell_time"] = round(dwell, 2)
                if dwell >= self.loiter_threshold_seconds and not result.get("loitering", False):
                    result["loitering"] = True

        return result

    def check_all_zones(self, obj_id: int, centroid: Tuple[float, float],
                        frame_width: int, frame_height: int) -> List[Dict[str, Any]]:
        """Check object against all configured zones."""
        results = []
        for zone in self.zones:
            result = self.check_zone(obj_id, centroid, frame_width, frame_height, zone)
            results.append(result)
        return results


class VideoPipeline:
    """Main video processing pipeline integrating camera sources, detection, tracking, and zone monitoring."""

    def __init__(self, config: dict = None, yolo_model=None, camera_source=None):
        self.config = config or {}
        self.yolo_model = yolo_model  # Ultralytics YOLO model instance
        self.camera_source = camera_source  # CameraSource instance

        # Initialize components
        tracker_config = self.config.get("tracker", {})
        self.tracker = CentroidTracker(
            max_disappeared=tracker_config.get("max_disappeared", 30),
            max_distance=tracker_config.get("max_distance", 50.0),
        )

        zones_config = self.config.get("zones", [])
        self.zone_monitor = ZoneMonitor(zones=zones_config)

        # Pipeline state
        self.frame_count = 0
        self.total_processed = 0
        self.running = False
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0

        # Statistics
        self.stats = {
            "total_objects_detected": 0,
            "total_tracked_objects": 0,
            "zone_violations": 0,
            "loitering_events": 0,
        }

    def set_camera_source(self, camera_source):
        """Set or change the camera source."""
        self.camera_source = camera_source

    def set_yolo_model(self, model):
        """Set the YOLO object detection model."""
        self.yolo_model = model

    def process_frame(self) -> Optional[Dict[str, Any]]:
        """Process next frame from camera source.

        Returns dict with frame, detections, tracks, and zone monitoring results,
        or None if no frame available.
        """
        if not self.running or self.camera_source is None:
            return None

        # Read frame from camera source
        read_result = self.camera_source.read()
        if read_result is None or read_result.get("frame") is None:
            return None

        frame = read_result["frame"]
        timestamp = read_result.get("timestamp", time.time())
        frame_height, frame_width = frame.shape[:2]

        self.frame_count += 1
        self.fps_counter += 1

        # Apply YOLO detection if model available
        detections: List[BoundingBox] = []
        if self.yolo_model is not None:
            try:
                # Convert BGR frame (OpenCV) to RGB if model expects RGB
                # Ultralytics YOLO typically accepts BGR numpy arrays
                results = self.yolo_model.predict(
                    source=frame,
                    conf=float(self.config.get("confidence_threshold", 0.45)),
                    iou=float(self.config.get("iou_threshold", 0.45)),
                    verbose=False,
                )

                for result_obj in results:
                    boxes = result_obj.boxes
                    if boxes is not None:
                        for box in boxes:
                            xyxy = box.xyxy[0].tolist()
                            conf = float(box.conf[0].item())
                            cls_id = int(box.cls[0].item())
                            cls_name = self.yolo_model.names.get(cls_id, f"class_{cls_id}")

                            # Only keep tracked classes if configured
                            if cls_id not in self.config.get("tracked_classes", list(TRACKED_CLASSES.keys())):
                                continue

                            bbox = BoundingBox(
                                x1=xyxy[0], y1=xyxy[1],
                                x2=xyxy[2], y2=xyxy[3],
                                confidence=conf,
                                class_id=cls_id,
                                class_name=cls_name,
                            )
                            detections.append(bbox)

            except Exception as e:
                # Log but don't crash; return empty detections
                print(f"[WARNING] YOLO inference error: {e}")

        # If no YOLO model, use synthetic detections from the camera source
        if not detections and self.camera_source is not None:
            # Try to get synthetic detections
            if hasattr(self.camera_source, "read_with_detections"):
                det_result = self.camera_source.read_with_detections(
                    yolo_model=None,
                    confidence_threshold=float(self.config.get("confidence_threshold", 0.45)),
                )
                if det_result and det_result.get("detections"):
                    for d in det_result["detections"]:
                        bbox = BoundingBox(
                            x1=d["bbox"][0], y1=d["bbox"][1],
                            x2=d["bbox"][2], y2=d["bbox"][3],
                            confidence=d["confidence"],
                            class_id=d["class_id"],
                            class_name=d["class_name"],
                        )
                        detections.append(bbox)

        # Update tracker
        tracked_objects = self.tracker.update(detections, timestamp)

        # Run zone monitoring on each tracked object
        zone_results = []
        for obj_id, tracked_obj in tracked_objects.items():
            # Update tracked object state
            tracked_obj.update(tracked_obj.bbox, timestamp)  # bbox already set

            # Get centroid for zone checks
            cx = (tracked_obj.bbox.x1 + tracked_obj.bbox.x2) / 2.0
            cy = (tracked_obj.bbox.y1 + tracked_obj.bbox.y2) / 2.0

            # Check all zones
            zone_checks = self.zone_monitor.check_all_zones(
                obj_id, (cx, cy), frame_width, frame_height
            )
            zone_results.extend(zone_checks)

            # Update stats
            self.stats["total_tracked_objects"] = len(tracked_objects)

        # Compute FPS
        if self.fps_counter % 30 == 1:  # every ~1 second at 30fps
            elapsed = time.time() - self.fps_start_time
            if elapsed > 0:
                self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = time.time()

        # Compile pipeline result
        result = {
            "frame": frame,
            "timestamp": timestamp,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "frame_count": self.frame_count,
            "current_fps": round(self.current_fps, 1),
            "detections": [d.to_tuple() for d in detections],  # simplify for serialization
            "tracked_objects": {oid: tobj.get_state() for oid, tobj in tracked_objects.items()},
            "zone_events": zone_results,
            "camera_source_status": self.camera_source.get_status(),
            "simulated": getattr(self.camera_source, "status", None) == "SIMULATED"
            if hasattr(self.camera_source, "get_status")
            else False,
        }

        # Update global stats
        self.stats["total_objects_detected"] += len(detections)

        # Check for zone violations / loitering events
        for zr in zone_results:
            if zr.get("loitering"):
                self.stats["loitering_events"] += 1
            if zr.get("entered") or zr.get("exited"):
                self.stats["zone_violations"] += 1

        return result

    def process_video_loop(self, max_frames: int = None):
        """Run continuous video processing loop.

        Processes frames until max_frames reached or source ends.
        Yields pipeline result dict for each frame.
        """
        self.running = True
        frame_num = 0

        while self.running:
            result = self.process_frame()
            if result is None:
                # Source ended or error
                break

            yield result
            frame_num += 1

            if max_frames is not None and frame_num >= max_frames:
                break

        self.running = False

    def get_statistics(self) -> Dict[str, Any]:
        """Return pipeline statistics."""
        return {
            **self.stats,
            "frame_count": self.frame_count,
            "current_fps": round(self.current_fps, 1),
            "tracked_objects_active": len(self.tracker.objects),
        }


# Convenience function
def create_pipeline(config: dict = None, yolo_model=None, camera_source=None) -> VideoPipeline:
    """Factory function to create a VideoPipeline instance."""
    return VideoPipeline(
        config=config,
        yolo_model=yolo_model,
        camera_source=camera_source,
    )