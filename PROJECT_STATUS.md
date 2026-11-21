# PROJECT_STATUS.md - Arlo Indoor Camera System

## Current Phase
Phase 11: VMC2070 Integration Research & Status Update

## System Overview
Local-first, privacy-preserving, AI-driven Smart Camera System for Arlo Indoor Camera hardware and multi-source video/audio processing. Optimized for Apple Silicon M4 with 16 GB unified memory.

## Hardware & Sensor Status Matrix
| Component / Sensor | Real Hardware Present | Truthful Status | Operating Mode | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Optical Camera | NO (VMC2070 model identified) | SEE BELOW | SEE BELOW | Arlo Essential Indoor Camera (3rd Gen) |
| Microphone Array | YES (Apple Silicon Mac) | AVAILABLE | Active / Simulation | Mac built-in mic / audio stream |
| Motion / PIR Sensor | NO | SIMULATED | Software Synthetic PIR | Simulated state |
| Thermal Sensor (LWIR/SWIR) | NO | UNAVAILABLE | Truthful Null State | Hardware physically absent |
| Radar Sensor | NO | UNAVAILABLE | Truthful Null State | Hardware physically absent |
| LiDAR Sensor | NO | UNAVAILABLE | Truthful Null State | Hardware physically absent |

## VMC2070 Integration Status
| Element | Status | Details |
| :--- | :--- | :--- |
| **Model** | IDENTIFIED | VMC2070 = Arlo Essential Indoor Camera (3rd Gen) |
| **RTSP Support** | UNSUPPORTED | Arlo explicitly does NOT support RTSP for VMC2070 |
| **ONVIF Support** | UNSUPPORTED | Arlo cameras have no ONVIF support |
| **Local LAN Streaming** | UNSUPPORTED | No documented local LAN/RTSP/ONVIF endpoints |
| **Cloud Dependency** | MANDATORY | All video streams to Arlo Secure cloud; Arlo Secure subscription required |
| **SmartHub/Base Station** | NOT COMPATIBLE | Arlo Essential Indoor Camera (3rd Gen) is not compatible with Arlo SmartHubs or Base Stations |
| **Direct Wi-Fi Connection** | YES | Camera connects directly to Wi-Fi router (2.4 GHz / 5 GHz) |
| **Local Live View** | NOT ACHIEVABLE | No verified local frames obtainable; no documented local interface |
| **Arlo Secure App** | REQUIRED | Video accessible only through Arlo Secure app/web portal |

## Completed Work
- [x] Initial engineering audit performed.
- [x] Workspace verified: `/Users/anderson/arlo-indoor-camera`.
- [x] Local runtime capabilities confirmed: Python 3.11.9, Apple Silicon M4, PyTorch, Ultralytics YOLO, OpenCV, AVFoundation, Librosa, EasyOCR.
- [x] Created `PROJECT_STATUS.md` tracking architecture, truthfulness matrix, and test status.
- [x] Package scaffolding: `arlo_camera/__init__.py`, `config.py`.
- [x] Sensor module: `arlo_camera/sensors/` with `base.py`, `mock_sensors.py` (PIR, mic, thermal, radar, lidar, night-vision, animal detection, fusion engine).
- [x] Camera sources: `arlo_camera/sources/` with `base.py`, `simulated.py` (test-pattern gradient frames with auto-start), `prerecorded.py` (video file playback).
- [x] Video processing pipeline: `arlo_camera/pipeline/video_pipeline.py` with YOLO detection, CentroidTracker, ZoneMonitor.
- [x] Audio event processing: `arlo_camera/audio/pipeline.py` with speech detection, glass-break, siren, animal classification, truthful state management.
- [x] Event storage: `arlo_camera/storage/event_store.py` SQLite-based local-only event persistence.
- [x] Alert engine: `arlo_camera/alert_engine.py` rule-based triggering, cooldown, local notification callbacks.
- [x] Privacy masking: `arlo_camera/privacy.py` PrivacyMaskEngine with color/blur/pixelate modes, zone configurations.
- [x] Phase 10: Live camera integration preparation with truthful status reporting.
- [x] Phase 11: VMC2070 integration research and status update.

## Current Work
- Verifying that simulated and prerecorded sources remain completely unaffected by VMC2070 status.
- Documenting exact hardware requirements for any future Arlo camera integration.
- Maintaining truthful status reporting across all camera source types.

## Tests Performed & Results
- Simulated camera source generates valid 1280x720 frames with test patterns (auto-starts on read()).
- Prerecorded source can load and play MP4 files from data/prerecorded/.
- Mock sensors produce truthful state readings (SIMULATED/UNAVAILABLE).
- Sensor fusion engine validates truthfulness across readings.
- YOLO object detection runs on simulated/prerecorded frames with bounding box output.
- Centroid tracker assigns stable track IDs across frames.
- Zone monitoring detects entry/exit, dwell time, and loitering events.
- Audio pipeline detects speech, glass-break, siren, and animal events with confidence scores.
- EventStore SQLite persists events locally; prune_old_events works correctly.
- Alert engine triggers zone entry/exit/loitering/glass_break/siren alerts with cooldown.
- PrivacyMaskEngine applies color/blur/pixelate masks; annotate_frame draws zone outlines.
- **Phase 10 baseline**: All verification tests pass (8 core modules + live camera stub).
- **Phase 11 findings**: VMC2070 integration research completed with 11/11 verification tests passing.

## Known Limitations
- Physical Arlo camera is absent; operating on synthetic/test-pattern and prerecorded video/audio streams.
- Radar, LiDAR, Thermal, MWIR, LWIR, SWIR sensors are marked `UNAVAILABLE` as hardware is physically absent on device.
- Face recognition and OCR capabilities dependent on model availability; currently simulation-only.
- Audio classification uses simple heuristics (energy/FFT); not a replacement for dedicated ML models.
- Dashboard is local-only; no remote access or cloud integration.
- **VMC2070 specific**: No documented local streaming interface; RTSP not supported; ONVIF not supported; cloud-only video access; no local LAN endpoints.

## VMC2070 Integration Research (Phase 11) - FINDINGS

### Exact Model: VMC2070 (Arlo Essential Indoor Camera 3rd Gen)
- Official model number: VMC2070
- Resolution: 1920 × 1080 HD
- Video codec: H.264
- Wi-Fi: 2.4 GHz / 5 GHz
- Connection: Direct to Wi-Fi router (not SmartHub/Base Station)
- SmartHub compatibility: NOT compatible

### Documented Video Interface
- **NONE** - Arlo Essential Indoor Camera (3rd Gen) has NO documented local video-streaming interface
- Video streams exclusively to Arlo's cloud servers
- Accessible only through Arlo Secure app (iOS/Android) or web portal
- Continuous Video Recording (CVR) optional, but VMC2070 incompatible with SmartHubs

### Documented Local Interface
- **NONE** - No local LAN, RTSP, ONVIF, or HTTP endpoints documented for VMC2070
- Camera communicates with Arlo's cloud using proprietary protocols
- No open ports for direct video access (ports 443 and 80 required for cloud connectivity)
- No RTSP URL, ONVIF endpoint, or local MJPEG stream available

### RTSP Status: UNSUPPORTED
- Arlo explicitly does NOT support RTSP for VMC2070
- "Arlo cameras do NOT support RTSP" (Arlo official docs, VisioForge)
- "No official local streaming option exists" (CamTuber RTSP guide)
- Previous RTSP beta through SmartHub (2019-2021) was removed in 2021 firmware update
- "There is currently no way to access Arlo camera streams via RTSP"
- "This feature was removed from SmartHub in 2021"

### ONVIF Status: UNSUPPORTED
- Arlo cameras have no ONVIF support
- "Arlo cameras have no RTSP, ONVIF, or local streaming endpoints" (VisioForge)
- ONVIF Device Manager cannot discover VMC2070 endpoints

### Cloud Dependency Status: MANDATORY
- All video routes through Arlo's cloud servers
- Arlo Secure subscription required for video access
- "Arlo's business model is subscription-based, and local streaming conflicts with this approach"
- Battery-powered design not suited for continuous local streaming

### Whether Local Live Frames Were Actually Obtained: NO
- No verified local frames obtained for VMC2070
- No documented or supported local interface exists
- Community workarounds exist but are unofficial, unstable, and require network manipulation/authentication bypass
- These do not constitute a "documented or supported interface"

### Files Changed (Phase 11)
- `arl_camera/live_camera.py` (REWORKED - VMC2070-specific statuses, integration reality, check_vmc2070_readiness())
- `PROJECT_STATUS.md` (UPDATED with Phase 11 findings, VMC2070 status matrix, documented interface status)
- `arl_camera/__init__.py` (no changes needed - already had try/except for live_camera)

### Exact Tests/Results
- All 11 verification tests passed:
  - Simulated source: frame shape (720, 1280, 3) uint8
  - Prerecorded source: MP4 playback functional
  - Mock sensors: PIR=SIMULATED, thermal/radar/lidar=UNAVAILABLE
  - LiveArloCameraSource VMC2070: ARLO_VMC2070_NOT_CONFIGURED
  - All capability flags False for VMC2070
  - check_vmc2070_readiness() truthfully reports: rtsp=False, onvif=False, local_live=False, cloud_only=True, live_local_frames_obtained=False, documented_local_interface=False
  - Simulated and prerecorded sources: completely unaffected

### Remaining Limitations
- No local live view possible for VMC2070 without Arlo cloud
- RTSP, ONVIF, and local LAN endpoints not available
- Arlo Secure subscription required for video access
- Community workarounds: unofficial, unstable, not supported
- Camera connects to Wi-Fi but video is cloud-mandatory
- No RTSP URL, ONVIF endpoint, or local port can be assumed or invented

### Recommended Next Step
- VMC2070 remains in ARLO_VMC2070_NOT_CONFIGURED / ARLO_VMC2070_CLOUD_ONLY state
- No local live view integration until Arlo officially documents a local streaming interface
- If/when Arlo documents local support, update LiveArloCameraSource.status to ARLO_VMC2070_LIVE_LOCAL and update capabilities
- Continue simulated/prerecorded development unaffected
- Pipeline and alert engine should check camera source status and disable local live view features for VMC2070, informing user of cloud-only access

## Truthfulness Compliance Statement
The system continues to truthfully report:
- **Simulated camera** = `SIMULATED` (always produces valid frame via `SimulatedCameraSource`)
- **Prerecorded camera** = `RECORDED` (plays back from video files via `PrerecordedCameraSource`)
- **VMC2070 camera** = `ARLO_VMC2070_NOT_CONFIGURED` / `ARLO_VMC2070_CLOUD_ONLY` (no local stream; video cloud-mandatory)
- **Unavailable sensors** (thermal, radar, LiDAR) = `UNAVAILABLE` (physically absent)

No hardware capabilities are claimed until actual camera connection has been tested successfully through a documented or supported interface.

## Phase History
- **Phase 1-4**: Initial audit, architectural setup, sensor abstractions, video pipeline
- **Phase 5-6**: Audio event processing, event storage, alert engine
- **Phase 7**: Privacy masking, dashboard UI
- **Phase 8**: Comprehensive automated testing and verification
- **Phase 9**: Final documentation and audit pass
- **Phase 10**: Real Camera Integration Research & Interface Preparation (stub added, no protocols implemented)
- **Phase 11**: VMC2070 Integration Research & Status Update (VMC2070 reality documented; no local streaming; cloud-only; all existing sources unaffected)

## Next Actions
- No immediate actions; VMC2070 status remains ARLO_VMC2070_NOT_CONFIGURED / CLOUD_ONLY
- Wait for Arlo to officially document a local streaming interface for VMC2070
- If/when documented, update LiveArloCameraSource and pipeline integration
- Continue simulated/prerecorded development unaffected
- Monitor Arlo official documentation for any future local streaming support announcements