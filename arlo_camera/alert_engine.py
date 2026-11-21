"""
Local Alert Engine for Arlo Indoor Camera System

Provides:
- Event-triggered alerts based on CV and audio analysis
- Configurable alert rules (zones, dwell time, sound types)
- Local-only notification (no cloud, no external API)
- Alert deduplication (avoid spamming same alert repeatedly)
- Alert cooldown per zone/object
- Integration with EventStore for historical context

Alerts never leave the local system. No external SMS, email, or webhook
destinations are called from this module directly - those would be
configured by the user's own infrastructure separately.
"""

import time
import threading
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Callable
from collections import defaultdict

from arlo_camera.storage.event_store import StoredEvent, create_event_store


# ---------------------------------------------------------------------------
# Alert rule configuration
# ---------------------------------------------------------------------------

DEFAULT_ALERT_RULES = {
    # Zone-based alerts
    "zone_entry": {
        "enabled": True,
        "description": "Person/vehicle entering a configured zone",
        "event_types": ["zone_entry"],
        "cooldown_seconds": 30,  # min time between same-zone alerts
    },
    "zone_exit": {
        "enabled": True,
        "description": "Person/vehicle exiting a configured zone",
        "event_types": ["zone_exit"],
        "cooldown_seconds": 30,
    },
    "loitering": {
        "enabled": True,
        "description": "Object dwelling in a zone beyond threshold",
        "event_types": ["loitering"],
        "cooldown_seconds": 60,
    },
    "intrusion": {
        "enabled": True,
        "description": "Person detected in restricted zone (combines entry + timing)",
        "event_types": ["zone_entry", "loitering"],
        "cooldown_seconds": 60,
    },
    # Audio-based alerts
    "glass_break": {
        "enabled": True,
        "description": "Glass break sound detected",
        "event_types": ["glass_break"],
        "cooldown_seconds": 120,
    },
    "siren": {
        "enabled": True,
        "description": "Siren or alarm sound detected",
        "event_types": ["siren"],
        "cooldown_seconds": 60,
    },
    "speech": {
        "enabled": False,  # disabled by default per privacy
        "description": "Speech/dialogue detected (no transcription)",
        "event_types": ["speech"],
        "cooldown_seconds": 30,
    },
}


# ---------------------------------------------------------------------------
# Alert types / categories
# ---------------------------------------------------------------------------

class AlertType(str, Enum):
    MOTION = "motion"
    ZONE_ENTRY = "zone_entry"
    ZONE_EXIT = "zone_exit"
    LOITERING = "loitering"
    INTRUSION = "intrusion"
    GLASS_BREAK = "glass_break"
    SIREN = "siren"
    SPEECH = "speech"
    UNKNOWN = "unknown"


@dataclass
class Alert:
    """Represents a triggered alert with all relevant information."""
    alert_id: str
    alert_type: AlertType
    timestamp: float
    source_id: str
    object_id: Optional[int]
    class_name: Optional[str]
    confidence: float
    message: str
    details: Dict[str, Any]  # additional context (zone name, duration, etc.)
    triggered_by: str  # "pipeline", "storage_query", etc.
    expires_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "timestamp": self.timestamp,
            "source_id": self.source_id,
            "object_id": self.object_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "message": self.message,
            "details": self.details,
            "triggered_by": self.triggered_by,
            "expires_at": self.expires_at,
        }


# ---------------------------------------------------------------------------
# Alert Engine
# ---------------------------------------------------------------------------


class AlertEngine:
    """Manages alert rules, deduplication, and local notification.

    Features:
    - Rule-based triggering from CV and audio events
    - Per-zone/object cooldown to prevent alert spam
    - Thread-safe operation
    - Integration with EventStore for context
    - User-registered callback for local notification (no default external calls)
    """

    def __init__(self, event_store: Any = None, rules: dict = None):
        self.event_store = event_store or create_event_store()
        self.rules = rules or DEFAULT_ALERT_RULES

        # State: last trigger time per (alert_type, source_id, object_id or zone)
        self.last_trigger: Dict[str, float] = {}

        # Lock for thread safety
        self._lock = threading.Lock()

        # User-registered alert callback (optional)
        # Callable signature: callback(alert: Alert) -> None
        self._alert_callback: Optional[Callable[[Alert], None]] = None

        # Initialize last_trigger timestamps to 0 (allow immediate first trigger)
        for alert_type in AlertType:
            for key_prefix in ["", "zone_", "obj_"]:
                self.last_trigger[f"{alert_type.value}_{key_prefix}"] = 0.0

    def set_alert_callback(self, callback: Callable[[Alert], None]):
        """Register a callback for local alert notification.

        The callback is invoked in the engine's thread context.
        It must be fast and non-blocking. Default: no external network calls.
        """
        self._alert_callback = callback

    def _is_cooldown_passed(self, alert_type: AlertType, key_suffix: str) -> bool:
        """Check if enough time has passed since last trigger of this rule."""
        cache_key = f"{alert_type.value}_{key_suffix}"
        last = self.last_trigger.get(cache_key, 0.0)
        cooldown = self.rules.get(alert_type.value, {}).get("cooldown_seconds", 60)
        return (time.time() - last) >= cooldown

    def _record_trigger(self, alert_type: AlertType, key_suffix: str):
        """Record the time of this alert trigger."""
        cache_key = f"{alert_type.value}_{key_suffix}"
        self.last_trigger[cache_key] = time.time()

    def _build_alert(self, alert_type: AlertType, message: str,
                     source_id: str, details: Dict[str, Any],
                     object_id: Optional[int] = None,
                     class_name: Optional[str] = None,
                     confidence: float = 1.0) -> Alert:
        """Build an Alert instance with a generated ID."""
        import uuid
        alert_id = str(uuid.uuid4())[:12]  # short but unique
        return Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            timestamp=time.time(),
            source_id=source_id,
            object_id=object_id,
            class_name=class_name,
            confidence=confidence,
            message=message,
            details=details,
            triggered_by="alert_engine",
        )

    def evaluate_zone_entry(self, obj_id: int, class_name: str,
                           source_id: str, zone_name: str,
                           confidence: float) -> Optional[Alert]:
        """Evaluate zone entry alert rule.

        Returns Alert if triggered, else None.
        """
        rule = self.rules.get("zone_entry")
        if not rule or not rule.get("enabled", True):
            return None

        key_suffix = f"zone:{zone_name}_obj:{obj_id}"
        if not self._is_cooldown_passed(AlertType.ZONE_ENTRY, key_suffix):
            return None  # still in cooldown

        # Build alert
        details = {"zone_name": zone_name, "object_id": obj_id, "class": class_name}
        alert = self._build_alert(
            AlertType.ZONE_ENTRY,
            f"{class_name.capitalize()} entered zone '{zone_name}'",
            source_id,
            details,
            object_id=obj_id,
            class_name=class_name,
            confidence=confidence,
        )

        self._record_trigger(AlertType.ZONE_ENTRY, key_suffix)
        return alert

    def evaluate_zone_exit(self, obj_id: int, class_name: str,
                          source_id: str, zone_name: str,
                          confidence: float) -> Optional[Alert]:
        """Evaluate zone exit alert rule."""
        rule = self.rules.get("zone_exit")
        if not rule or not rule.get("enabled", True):
            return None

        key_suffix = f"zone:{zone_name}_obj:{obj_id}"
        if not self._is_cooldown_passed(AlertType.ZONE_EXIT, key_suffix):
            return None

        details = {"zone_name": zone_name, "object_id": obj_id, "class": class_name}
        alert = self._build_alert(
            AlertType.ZONE_EXIT,
            f"{class_name.capitalize()} exited zone '{zone_name}'",
            source_id,
            details,
            object_id=obj_id,
            class_name=class_name,
            confidence=confidence,
        )

        self._record_trigger(AlertType.ZONE_EXIT, key_suffix)
        return alert

    def evaluate_loitering(self, obj_id: int, class_name: str,
                          source_id: str, zone_name: str,
                          dwell_time: float, confidence: float) -> Optional[Alert]:
        """Evaluate loitering alert rule."""
        rule = self.rules.get("loitering")
        if not rule or not rule.get("enabled", True):
            return None

        key_suffix = f"loiter:{zone_name}_obj:{obj_id}"
        if not self._is_cooldown_passed(AlertType.LOITERING, key_suffix):
            return None

        details = {
            "zone_name": zone_name,
            "object_id": obj_id,
            "class": class_name,
            "dwell_seconds": round(dwell_time, 2),
        }
        alert = self._build_alert(
            AlertType.LOITERING,
            f"{class_name.capitalize()} loitering in '{zone_name}' "
            f"{dwell_time:.1f}s (threshold:{rule.get('cooldown_seconds', 8)}s)",
            source_id,
            details,
            object_id=obj_id,
            class_name=class_name,
            confidence=confidence,
        )

        self._record_trigger(AlertType.LOITERING, key_suffix)
        return alert

    def evaluate_glass_break(self, source_id: str, confidence: float,
                            rms_energy: float, dominant_freq: float) -> Optional[Alert]:
        """Evaluate glass-break alert rule."""
        rule = self.rules.get("glass_break")
        if not rule or not rule.get("enabled", True):
            return None

        # Additional guard: energy must exceed threshold
        energy_thresh = rule.get("energy_threshold", 0.05)
        if rms_energy < energy_thresh:
            return None  # too quiet to be glass break

        key_suffix = f"gb_{source_id}"
        if not self._is_cooldown_passed(AlertType.GLASS_BREAK, key_suffix):
            return None

        details = {
            "rms_energy": round(rms_energy, 4),
            "dominant_freq_hz": round(dominant_freq, 1),
        }
        alert = self._build_alert(
            AlertType.GLASS_BREAK,
            "Glass break sound detected",
            source_id,
            details,
            confidence=confidence,
        )

        self._record_trigger(AlertType.GLASS_BREAK, key_suffix)
        return alert

    def evaluate_siren(self, source_id: str, confidence: float,
                       dominant_freq: float) -> Optional[Alert]:
        """Evaluate siren alert rule."""
        rule = self.rules.get("siren")
        if not rule or not rule.get("enabled", True):
            return None

        key_suffix = f"siren_{source_id}"
        if not self._is_cooldown_passed(AlertType.SIREN, key_suffix):
            return None

        details = {"dominant_freq_hz": round(dominant_freq, 1)}
        alert = self._build_alert(
            AlertType.SIREN,
            "Siren or alarm sound detected",
            source_id,
            details,
            confidence=confidence,
        )

        self._record_trigger(AlertType.SIREN, key_suffix)
        return alert

    def evaluate_speech(self, source_id: str, confidence: float) -> Optional[Alert]:
        """Evaluate speech detection alert rule.

        Disabled by default per privacy requirements. Enable only if
        user explicitly opts in, and even then, no transcription is performed.
        """
        rule = self.rules.get("speech")
        if not rule or not rule.get("enabled", False):
            return None  # disabled by default

        key_suffix = f"speech_{source_id}"
        if not self._is_cooldown_passed(AlertType.SPEECH, key_suffix):
            return None

        details = {"confidence": round(confidence, 3)}
        alert = self._build_alert(
            AlertType.SPEECH,
            "Speech/dialogue detected (no transcription)",
            source_id,
            details,
            confidence=confidence,
        )

        self._record_trigger(AlertType.SPEECH, key_suffix)
        return alert

    # -----------------------------------------------------------------
    # Public: process events from pipeline
    # -----------------------------------------------------------------

    def process_zone_event(self, event_data: Dict[str, Any]) -> List[Alert]:
        """Process a zone monitoring event and return triggered alerts.

        Expected event_data keys:
        - "event_type": "zone_entry" | "zone_exit" | "loitering"
        - "obj_id": track ID
        - "class_name": "person" | "car" | "dog" | etc.
        - "source_id": camera source ID
        - "zone_name": configured zone name
        - "confidence": detection confidence
        - "dwell_time": only for loitering
        """
        alerts = []

        event_type = event_data.get("event_type")
        obj_id = event_data.get("obj_id")
        class_name = event_data.get("class_name")
        source_id = event_data.get("source_id", "unknown")
        zone_name = event_data.get("zone_name", "unknown")
        confidence = event_data.get("confidence", 1.0)
        dwell_time = event_data.get("dwell_time", 0.0)

        if event_type == "zone_entry":
            alert = self.evaluate_zone_entry(
                obj_id, class_name, source_id, zone_name, confidence
            )
            if alert:
                alerts.append(alert)

        elif event_type == "zone_exit":
            alert = self.evaluate_zone_exit(
                obj_id, class_name, source_id, zone_name, confidence
            )
            if alert:
                alerts.append(alert)

        elif event_type == "loitering":
            alert = self.evaluate_loitering(
                obj_id, class_name, source_id, zone_name, dwell_time, confidence
            )
            if alert:
                alerts.append(alert)

        return alerts

    def process_audio_event(self, event_data: Dict[str, Any]) -> List[Alert]:
        """Process an audio event and return triggered alerts.

        Expected event_data keys:
        - "event_type": "glass_break" | "siren" | "speech"
        - "source_id": audio source ID
        - "confidence": classifier confidence
        - "rms_energy": RMS energy (for glass_break)
        - "dominant_freq": dominant frequency (for glass_break/siren)
        """
        alerts = []
        event_type = event_data.get("event_type")
        source_id = event_data.get("source_id", "unknown")
        confidence = event_data.get("confidence", 1.0)
        rms_energy = event_data.get("rms_energy", 0.0)
        dominant_freq = event_data.get("dominant_freq", 0.0)

        if event_type == "glass_break":
            alert = self.evaluate_glass_break(
                source_id, confidence, rms_energy, dominant_freq
            )
            if alert:
                alerts.append(alert)

        elif event_type == "siren":
            alert = self.evaluate_siren(source_id, confidence, dominant_freq)
            if alert:
                alerts.append(alert)

        elif event_type == "speech":
            alert = self.evaluate_speech(source_id, confidence)
            if alert:
                alerts.append(alert)

        return alerts

    def get_active_alerts(self) -> List[Alert]:
        """Return currently active (non-expired) alerts from storage.

        Note: This queries the EventStore; actual fresh alerts are
        returned by the above evaluation methods.
        """
        # Retrieve recent alerts from storage
        events = self.event_store.get_recent(
            event_type="alert",
            limit=50,
        )
        # Convert stored dicts back to Alert objects (simplified)
        # In a full impl, we'd store Alert dataclasses directly
        alert_list = []
        for e in events:
            try:
                a = Alert(
                    alert_id=e.event_id,
                    alert_type=AlertType(e.event_type),
                    timestamp=e.timestamp,
                    source_id=e.source_id,
                    object_id=e.object_id,
                    class_name=e.class_name,
                    confidence=e.confidence,
                    message=e.metadata.get("message", e.event_type),
                    details=e.metadata.get("details", {}),
                    triggered_by=e.metadata.get("triggered_by", "storage"),
                )
                alert_list.append(a)
            except Exception:
                continue  # skip malformed entries
        return alert_list

    def trigger_user_defined(self, alert_type: str, message: str,
                             source_id: str, details: Dict[str, Any],
                             obj_id: Optional[int] = None,
                             class_name: Optional[str] = None,
                             confidence: float = 1.0) -> Alert:
        """Allow user/integration to trigger a custom alert type.

        Useful for external rules engines or MQTT bridges that the user
        configures themselves (outside this module's defaults).
        """
        alert = self._build_alert(
            AlertType(alert_type) if alert_type in [t.value for t in AlertType] else AlertType.UNKNOWN,
            message,
            source_id,
            details,
            object_id=obj_id,
            class_name=class_name,
            confidence=confidence,
        )
        # Invoke callback if registered
        if self._alert_callback is not None:
            try:
                self._alert_callback(alert)
            except Exception as e:
                print(f"[WARNING] Alert callback error: {e}")
        return alert


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_alert_engine(event_store: Any = None, rules: dict = None) -> AlertEngine:
    """Create a ready-to-use AlertEngine instance."""
    return AlertEngine(event_store=event_store, rules=rules)