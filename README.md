# Local Vision Camera

A local-first computer vision system for zone tracking and on-device alerts. Frames come from a simulated test-pattern source or from a prerecorded video file. Detections can be produced by an injected YOLO model (Ultralytics) or, when no model is loaded, by synthetic detections from the source. A Flask dashboard binds those components to a localhost UI.

**Status:** developed and verified on simulated and prerecorded sources. It has not been proven against a live hardware camera stream in this repository. There is no automated test suite in the tree; earlier verification was run interactively against the modules listed below.

## Architecture

```
Camera source (simulated or prerecorded)
        │
        ▼
Video pipeline ── YOLO detect ── centroid tracker ── zone monitor
        │
        ├─ privacy masks (color / blur / pixelate)
        ├─ alert engine (entry, exit, loitering, audio events, cooldowns)
        ├─ SQLite event store (local only)
        └─ localhost Flask dashboard (127.0.0.1)
```

Optional audio and mock sensor modules report truthful states: simulated when the software is faking a sensor, and unavailable when the hardware is not present.

| Path | Role |
|---|---|
| `arlo_camera/sources/` | Simulated test-pattern frames and prerecorded file playback |
| `arlo_camera/pipeline/` | Detection, `CentroidTracker`, `ZoneMonitor` |
| `arlo_camera/alert_engine.py` | Rule-based triggers and cooldowns |
| `arlo_camera/privacy.py` | Zone privacy masks and annotations |
| `arlo_camera/storage/` | Local SQLite event persistence |
| `arlo_camera/audio/` | Heuristic audio events (energy / FFT), not a trained classifier |
| `arlo_camera/sensors/` | Mock and unavailable sensor states |
| `arlo_camera/dashboard/` | Local-only Flask app |
| `arlo_camera/live_camera.py` | Readiness and capability reporting for a live source (no live local stream is claimed) |

The importable package name is `arlo_camera` for historical reasons. This project is not affiliated with Arlo Technologies.

## Stack

Python 3, OpenCV, NumPy, and Flask. Optional: PyTorch and Ultralytics YOLO (`yolov8n.pt` is the configured default model name, and weights are not shipped), plus PyAudio for microphone input. The dashboard listens on `127.0.0.1` only.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place optional prerecorded clips under `data/prerecorded/`. Event databases and captured media stay local and are gitignored.

Minimal simulated loop:

```python
from arlo_camera import (
    create_simulated_source,
    create_pipeline,
    create_alert_engine,
    create_event_store,
)
from arlo_camera.dashboard import create_dashboard_app

source = create_simulated_source()
pipeline = create_pipeline(camera_source=source)  # pass yolo_model=... to enable detection
event_store = create_event_store()
alerts = create_alert_engine()
app = create_dashboard_app(pipeline, alerts, event_store)
app.run(host="127.0.0.1", port=5000, debug=False)
```

Or, after wiring components the same way, `python -m arlo_camera.dashboard` starts the localhost dashboard.

To use YOLO, load a model yourself and pass it into `create_pipeline(..., yolo_model=model)`. This repository does not vendor model weights.

## What was tested

On simulated 1280×720 test-pattern frames and on prerecorded MP4 playback, the following modules were exercised locally:

- Frame acquisition from simulated and prerecorded sources
- YOLO inference on those frames when a model was provided
- Stable track IDs from the centroid tracker
- Zone entry, exit, dwell, and loitering events
- Privacy masks (color, blur, pixelate)
- Local SQLite persistence and retention pruning
- Alert cooldowns for zone and heuristic audio events
- Mock sensors reporting `SIMULATED` or `UNAVAILABLE` rather than pretending hardware exists

There is no `tests/` directory in this publication. Results above are from the interactive verification recorded in `PROJECT_STATUS.md`.

## Status and limitations

- Tested on simulated and prerecorded sources only. Do not assume a live camera works until you have verified frames from a documented, supported interface you are allowed to use.
- A consumer cloud-only Wi-Fi camera was researched and has no documented local RTSP, ONVIF, or LAN video endpoint in this project. Live local capture from that class of device is not supported here.
- Radar, LiDAR, and thermal sensors are marked unavailable; they are not present.
- Face recognition and OCR are simulation-only and depend on models that are not bundled.
- Audio classification uses simple energy and FFT heuristics, not a dedicated ML model.
- The dashboard is local-only. There is no remote access or cloud upload path in this codebase.
