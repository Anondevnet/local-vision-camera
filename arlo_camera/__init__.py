"""
Arlo Indoor Camera Software System
Local-First, Privacy-Preserving AI Smart Camera Engine
"""

__version__ = "1.0.0"

# Core modules (always available)
from arlo_camera.config import default_config
from arlo_camera.sensors.mock_sensors import *
from arlo_camera.sources.simulated import create_simulated_source
from arlo_camera.sources.prerecorded import create_prerecorded_source
from arlo_camera.pipeline.video_pipeline import create_pipeline
from arlo_camera.audio.pipeline import create_audio_pipeline
from arlo_camera.storage.event_store import create_event_store
from arlo_camera.alert_engine import create_alert_engine
from arlo_camera.privacy import create_privacy_mask_engine

# Optional/preparation modules (importable but not auto-loaded)
try:
    from arlo_camera.live_camera import LiveCameraStatus, LiveCameraCapability, LiveArloCameraSource, check_live_camera_readiness
except ImportError:
    # Live camera preparation not available - acceptable for phase prior to hardware integration
    pass
