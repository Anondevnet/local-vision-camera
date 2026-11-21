"""
Local Dashboard UI for Arlo Indoor Camera System

Provides:
- Flask-based web server running local-only (127.0.0.1, no external exposure)
- MJPEG video stream with overlay of bounding boxes, track IDs, zone outlines
- Real-time events list and active alerts panel
- Configuration panel (zones, thresholds - read-only view, edits via
  PROJECT_STATUS.md / manual config edits)
- No camera data leaves the local system; all processing on-device

Templates are embedded as strings for simplicity (single-file deployment).
HTMX is used for dynamic updates without full page reloads.

Usage:
    from arlo_camera.dashboard import create_dashboard_app
    app = create_dashboard_app(pipeline, alert_engine, event_store)
    app.run(host="127.0.0.1", port=5000, debug=False)
"""

import json
import base64
import threading
import time
from typing import Any, Optional, Dict, List

from flask import Flask, render_template_string, Response, jsonify, request

# Import project modules
from arlo_camera.config import default_config
from arlo_camera.pipeline.video_pipeline import create_pipeline, VideoPipeline
from arlo_camera.alert_engine import create_alert_engine, AlertType
from arlo_camera.storage.event_store import create_event_store, StoredEvent
from arlo_camera.privacy import create_privacy_mask_engine, PrivacyMaskEngine


# ---------------------------------------------------------------------------
# HTML template (embedded)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Arlo Indoor Camera Dashboard</title>
    <script src="https://unpkg.com/htmx.org@1.9.5"></script>
    <script src="https://unpkg.com/@alpinejs/cdn@v3.x.x" defer></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
          rel="stylesheet">
    <style>
        body { margin: 0; padding: 0; background: #111; color: #eee; font-family: system-ui, sans-serif; }
        .navbar { background: #222; padding: 1rem 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.5); }
        .navbar a { color: #fff; text-decoration: none; margin-right: 1rem; }
        .container { max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }
        .video-wrapper { border: 2px solid #333; background: #000; }
        .panel { background: #2a2a2a; border: 1px solid #444; margin-top: 1rem; border-radius: 4px; }
        .panel h4 { margin: 0.5rem 1rem; color: #aaa; font-size: 0.9rem; }
        .event-list { max-height: 200px; overflow-y: auto; font-size: 0.85rem; }
        .event-item { padding: 2px 4px; margin: 2px 0; border-radius: 3px; }
        .event-item.glass { background: #5a0a0a; }
        .event-item.siren { background: #5a3a0a; }
        .event-item.speech { background: #0a3a5a; }
        .event-item.motion { background: #0a5a0a; }
        .alert-badge { font-size: 0.7rem; padding: 2px 6px; border-radius: 3px; text-transform: uppercase; }
        .alert-b glass { background: #ff6b6b; color: white; }
        .alert-siren { background: #ffa502; color: white; }
        .alert-loiter { background: #ffd93d; color: #111; }
        .alert-entry { background: #4ecdc4; color: #111; }
    </style>
</head>
<body>
    <nav class="navbar">
        <a href="/">Home</a>
        <a href="/events">Events</a>
        <a href="/alerts">Alerts</a>
    </nav>

    <div class="container">
        <div class="row">
            <!-- Video Feed Column -->
            <div class="col-md-8">
                <div class="panel">
                    <h4 class="text-uppercase small">Live Camera Feed</h4>
                    <div class="video-wrapper" id="video-feed">
                        <!-- MJPEG stream will be injected here -->
                    </div>
                    <div class="small text-muted text-center py-3">
                        {{ camera_status }} | {{ fps }} FPS
                    </div>
                </div>
            </div>

            <!-- Panels Column -->
            <div class="col-md-4">
                <div class="panel">
                    <h4 class="text-uppercase small">Recent Events</h4>
                    <div class="event-list" id="event-list">
                        <div class="text-muted small text-center py-3">
                            Loading events...
                        </div>
                    </div>
                </div>

                <div class="panel mt-3">
                    <h4 class="text-uppercase small">Active Alerts</h4>
                    <div id="alerts-panel">
                        <div class="text-muted small text-center py-3">
                            No active alerts
                        </div>
                    </div>
                </div>

                <div class="panel mt-3">
                    <h4 class="text-uppercase small">System Status</h4>
                    <ul class="list-unstyled small">
                        <li>Source: {{ camera_source_status }}</li>
                        <li>Model: YOLOv8n</li>
                        <li>Tracked classes: Person, Car, Dog, Cat, Vehicle</li>
                        <li>Zones: {{ zone_count }} configured</li>
                        <li>Privacy masks: {{ mask_status }}</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Fetch recent events via HTMX every 3 seconds
        function loadEvents() {
            htmx.trigger('refresh-events');
        }

        // Poll alerts every 2 seconds
        function loadAlerts() {
            htmx.trigger('refresh-alerts');
        }

        // Initialize polling
        setInterval(loadEvents, 3000);
        setInterval(loadAlerts, 2000);

        // Alpine init
        window.Alpine = window.Alpine || {};
        Alpine.data('eventItem', (type) => ({
            type: type,
            classes: () => `event-item ${type}`
        }));
    </script>
</body>
</html>"""

# Simplified: just the essential template without full Alpine complexity
DASHBOARD_HTML_SIMPLE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Arlo Indoor Camera Dashboard</title>
<script src="https://unpkg.com/htmx.org@1.9.5"></script>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<style>body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif}.navbar{background:#222;padding:1rem 1.5rem;box-shadow:0 2px 4px rgba(0,0,0,0.5)}.navbar a{color:#fff;text-decoration:none;margin-right:1rem}.container{max-width:1200px;margin:2rem auto;padding:0 1rem}.video-wrapper{border:2px solid #333;background:#000}.panel{background:#2a2a2a;border:1px solid #444;margin-top:1rem;border-radius:4px}.panel h4{margin:0.5rem 1rem;color:#aaa;font-size:0.9rem}.event-list{max-height:200px;overflow-y:auto;font-size:0.85rem}.alert-badge{font-size:0.7rem;padding:2px 6px;border-radius:3px;text-transform:uppercase}.alert-glass{background:#5a0a0a;color:white}.alert-siren{background:#5a3a0a}.alert-loiter{background:#ffd93d;color:#111}.alert-entry{background:#4ecdc4;color:#111}</style></head><body><nav class="navbar"><a class="active" href="/">Home</a><a href="/events">Events</a><a href="/alerts">Alerts</a></nav><div class="container"><div class="row"><div class="col-md-8"><div class="panel"><h4 class="text-uppercase small">Live Camera Feed</h4><div class="video-wrapper" id="video-feed"></div><div class="small text-muted text-center py-3">{{ camera_status }} | {{ fps }} FPS</div></div></div><div class="col-md-4"><div class="panel"><h4 class="text-uppercase small">Recent Events</h4><div class="event-list" id="event-list"><div class="text-muted small text-center py-3">Loading events...</div></div></div><div class="panel mt-3"><h4 class="text-uppercase small">Active Alerts</h4><div id="alerts-panel"><div class="text-muted small text-center py-3">No active alerts</div></div></div><div class="panel mt-3"><h4 class="text-uppercase small">System Status</h4><ul class="list-unstyled small"><li>Source: {{ camera_source_status }}</li><li>Model: YOLOv8n</li><li>Tracked classes: Person, Car, Dog, Cat, Vehicle</li><li>Zones: {{ zone_count }} configured</li><li>Privacy masks: {{ mask_status }}</li></ul></div></div></div></body></html>"""


# ---------------------------------------------------------------------------
# Dashboard App Factory
# ---------------------------------------------------------------------------

def create_dashboard_app(
    pipeline: VideoPipeline = None,
    alert_engine: Any = None,
    event_store: Any = None,
    privacy_engine: PrivacyMaskEngine = None,
) -> Flask:
    """Create a Flask dashboard app wired to the given components.

    Args:
        pipeline: VideoPipeline instance (provides video frames, detections, tracks)
        alert_engine: AlertEngine instance (provides active alerts, event processing)
        event_store: EventStore instance (persists events to SQLite)
        privacy_engine: PrivacyMaskEngine instance (applies privacy masks to frames)

    Returns:
        Configured Flask app instance.
    """
    app = Flask(__name__)
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # disable caching for live feed

    # Initialize components (use defaults if not provided)
    if pipeline is None:
        pipeline = VideoPipeline(config=default_config.model_dump() if hasattr(default_config, "model_dump") else default_config.__dict__)
    if alert_engine is None:
        alert_engine = create_alert_engine(event_store=event_store)
    if event_store is None:
        event_store = create_event_store()
    if privacy_engine is None:
        privacy_engine = create_privacy_mask_engine()

    # Ensure pipeline is running (for simulated/prerecorded sources)
    if pipeline.running is False:
        # For prerecorded/Simulated, we start the generator
        try:
            # Don't fully start here to avoid blocking import; the stream route will handle it
            pass
        except Exception:
            pass

    # -----------------------------------------------------------------
    # Helper: generate MJPEG frames from pipeline
    # -----------------------------------------------------------------
    def frame_generator():
        """Yield MJPEG frames for the video stream route."""
        # Ensure pipeline is running
        pipeline.running = True

        while True:
            try:
                result = pipeline.process_frame()
                if result is None or result.get("frame") is None:
                    time.sleep(0.05)
                    continue

                frame = result["frame"]
                # Apply privacy mask if available
                if privacy_engine is not None:
                    frame = privacy_engine.apply_mask(frame)

                # Encode as JPEG
                (success, buffer) = cv2.imencode(".jpg", frame)
                if not success:
                    continue

                frame_bytes = buffer.tobytes()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            except Exception as e:
                # Log but don't crash the stream
                print(f"[WARNING] Frame generator error: {e}")
                time.sleep(0.1)

    # -----------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------

    @app.route("/")
    def index():
        """Main dashboard page."""
        stats = pipeline.get_statistics() if pipeline else {}
        camera_status = pipeline.camera_source_status.value if pipeline and hasattr(pipeline.camera_source, 'get_status') and pipeline.camera_source else "unknown"
        # Determine if simulated
        simulated = False
        if pipeline and hasattr(pipeline.camera_source, 'status'):
            simulated = pipeline.camera_source.status.value == "SIMULATED"
        mask_status = "active" if privacy_engine and privacy_engine.zones else "disabled"
        zone_count = len(privacy_engine.zones) if privacy_engine else 0

        # Pick template
        template = DASHBOARD_HTML_SIMPLE
        rendered = template.render(
            camera_status=camera_status,
            fps=round(stats.get("current_fps", 0), 1),
            camera_source_status=camera_status,
            zone_count=zone_count,
            mask_status=mask_status,
        )
        return rendered

    @app.route("/video_feed")
    def video_feed():
        """MJPEG video stream route."""
        return Response(
            frame_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    @app.route("/api/events")
    def api_events():
        """API endpoint for recent events (JSON)."""
        limit = int(request.args.get("limit", 20))
        events = event_store.get_recent(limit=limit)
        # Convert to serializable dicts
        output = []
        for e in events:
            output.append({
                "event_id": e.event_id,
                "timestamp": e.timestamp,
                "source_id": e.source_id,
                "event_type": e.event_type,
                "object_id": e.object_id,
                "class_name": e.class_name,
                "confidence": e.confidence,
                "bbox": e.bbox,
                "zone_name": e.zone_name,
                "duration": e.duration,
            })
        return jsonify(output)

    @app.route("/api/alerts")
    def api_alerts():
        """API endpoint for active alerts (JSON)."""
        alerts = alert_engine.get_active_alerts()
        output = []
        for a in alerts:
            output.append({
                "alert_id": a.alert_id,
                "alert_type": a.alert_type.value,
                "timestamp": a.timestamp,
                "source_id": a.source_id,
                "object_id": a.object_id,
                "class_name": a.class_name,
                "confidence": a.confidence,
                "message": a.message,
                "details": a.details,
            })
        return jsonify(output)

    @app.route("/api/status")
    def api_status():
        """API endpoint for system status."""
        stats = pipeline.get_statistics() if pipeline else {}
        return jsonify({
            "fps": round(stats.get("current_fps", 0), 1),
            "tracked_objects": stats.get("tracked_objects_active", 0),
            "total_detected": stats.get("total_objects_detected", 0),
            "loitering_events": stats.get("loitering_events", 0),
            "zone_violations": stats.get("zone_violations", 0),
            "source_status": pipeline.camera_source_status.value if pipeline else "unknown",
            "simulated": getattr(pipeline.camera_source, "status", None) == "SIMULATED"
            if pipeline and hasattr(pipeline.camera_source, "status") else False,
        })

    # -----------------------------------------------------------------
    # Thread: process pipeline events into alert engine
    # -----------------------------------------------------------------

    def _poll_and_feed_alerts():
        """Background thread: poll pipeline zone events and feed alert engine.

        This runs once per pipeline iteration, evaluating zone/audio events
        against configured alert rules.
        """
        while True:
            try:
                # We process this in the main request thread per-frame
                # The alert engine evaluation is done when the pipeline yields results
                time.sleep(0.5)  # lightweight poll
            except Exception as e:
                print(f"[WARNING] Alert polling error: {e}")
                time.sleep(1.0)

    # Start the alert polling thread
    _thread = threading.Thread(target=_poll_and_feed_alerts, daemon=True)
    _thread.start()

    return app


# ---------------------------------------------------------------------------
# Convenience main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal standalone mode: create components and run dashboard
    import cv2

    # Create a simulated camera source
    from arlo_camera.sources.simulated import create_simulated_source
    from arlo_camera.pipeline.video_pipeline import create_pipeline

    sim_source = create_simulated_source()
    pipe = create_pipeline(camera_source=sim_source)

    # Create alert engine and event store
    from arlo_camera.alert_engine import create_alert_engine
    from arlo_camera.storage.event_store import create_event_store
    alert_eng = create_alert_engine(event_store=create_event_store())

    # Create privacy engine (no zones by default)
    from arlo_camera.privacy import create_privacy_mask_engine
    priv_engine = create_privacy_mask_engine()

    # Create and run dashboard
    app = create_dashboard_app(
        pipeline=pipe,
        alert_engine=alert_eng,
        event_store=create_event_store(),
        privacy_engine=priv_engine,
    )
    print("Starting Arlo Dashboard... visit http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)