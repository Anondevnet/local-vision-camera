"""
System Configuration for Arlo Indoor Camera System
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EVENTS_DIR = DATA_DIR / "events"
FACES_DIR = DATA_DIR / "faces"
PRERECORDED_DIR = DATA_DIR / "prerecorded"

# Ensure directories exist
for d in [DATA_DIR, EVENTS_DIR, FACES_DIR, PRERECORDED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

@dataclass
class ZoneConfig:
    name: str
    points: List[Tuple[float, float]]  # Normalized (x, y) coordinates [0.0, 1.0]
    zone_type: str = "detection"       # "detection", "dwell", "tripwire", "privacy"
    loiter_threshold_seconds: float = 10.0

@dataclass
class SystemConfig:
    # Camera settings
    frame_width: int = 1280
    frame_height: int = 720
    target_fps: int = 30
    
    # Device / Acceleration
    device: str = "mps"  # "mps" for Apple Silicon, "cpu", "cuda"
    
    # Object Detection
    yolo_model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.45
    
    # Target tracking classes (COCO IDs)
    # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck, 15: cat, 16: dog
    tracked_classes: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 5, 7, 15, 16])
    
    # Face Recognition
    face_confidence_threshold: float = 0.75
    face_unknown_label: str = "Unknown Person"
    
    # OCR / License Plate
    ocr_min_confidence: float = 0.60
    
    # Dwell & Loitering
    loitering_time_seconds: float = 8.0
    
    # Zones & Privacy
    zones: List[ZoneConfig] = field(default_factory=lambda: [
        ZoneConfig(
            name="Main Entrance Zone",
            points=[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)],
            zone_type="detection",
            loiter_threshold_seconds=5.0
        ),
        ZoneConfig(
            name="Privacy Mask - Window / Street",
            points=[(0.0, 0.0), (0.3, 0.0), (0.3, 0.2), (0.0, 0.2)],
            zone_type="privacy"
        )
    ])
    
    # Database / Storage
    db_path: str = str(EVENTS_DIR / "events.db")
    event_retention_days: int = 30
    
    # Audio Processing
    sample_rate: int = 16000
    chunk_size: int = 1024
    audio_enabled: bool = True
    transcription_enabled: bool = False  # Disabled by default per privacy requirements

default_config = SystemConfig()
