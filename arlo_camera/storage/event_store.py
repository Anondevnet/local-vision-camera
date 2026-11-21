"""
Local Event Storage for Arlo Indoor Camera System

Provides:
- SQLite-based event storage (local-only, no cloud upload)
- Event types: motion, zone_violation, loitering, audio_event, etc.
- Per-object tracking data (bounding boxes, track IDs, confidence)
- Configurable retention policies (TTL in days)
- Query API for recent events, zone activity, audio events
- Privacy-aware: stores only metadata, applies privacy masks before saving frames

Data model is intentionally minimal - stores event attributes, not raw
full-resolution frames (frames are processed in-memory; only metadata/
alerts are persisted).
"""

import sqlite3
import json
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator, Tuple
from dataclasses import dataclass, asdict

# Version string for this module
__version__ = "1.0.0"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class StoredEvent:
    """Represents a stored event with all relevant metadata."""
    event_id: str
    timestamp: float
    source_id: str
    event_type: str  # "motion", "zone_violation", "loitering", "audio_event", etc.
    object_id: Optional[int]  # track ID, or None
    class_name: Optional[str]  # "person", "car", "dog", etc.
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    zone_name: Optional[str]  # which zone was triggered, if any
    duration: Optional[float]  # dwell time, loitering seconds, etc.
    metadata: Dict[str, Any]  # additional event-specific data
    recorded_at: float  # when event was stored to DB

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Storage class
# ---------------------------------------------------------------------------

class EventStore:
    """SQLite-based event storage manager.

    All data is stored locally. No network upload is performed.
    Events are retained for CONFIG_EVENT_RETENTION_DAYS (default 30).
    """

    def __init__(self, db_path: str = None, retention_days: int = 30):
        # Resolve path relative to project root
        if db_path is None:
            BASE_DIR = Path(__file__).resolve().parent.parent
            db_path = str(BASE_DIR / "data" / "events" / "events.db")

        self.db_path = db_path
        self.retention_days = retention_days

        # Ensure directory exists
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()

    def _create_tables(self):
        """Create storage tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                source_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                object_id INTEGER,
                class_name TEXT,
                confidence REAL NOT NULL,
                bbox_text TEXT NOT NULL,
                zone_name TEXT,
                duration REAL,
                metadata_text TEXT,
                recorded_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
            CREATE INDEX IF NOT EXISTS idx_events_source ON events(source_id);
        """)
        self._conn.commit()

    def store(self, event: StoredEvent) -> bool:
        """Store a single event into the database.

        Returns True on success.
        """
        try:
            metadata_json = json.dumps(event.metadata) if event.metadata else "{}"
            self._conn.execute(
                """INSERT OR REPLACE INTO events
                   (event_id, timestamp, source_id, event_type, object_id,
                    class_name, confidence, bbox_text, zone_name, duration,
                    metadata_text, recorded_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.event_id,
                    event.timestamp,
                    event.source_id,
                    event.event_type,
                    event.object_id,
                    event.class_name,
                    event.confidence,
                    json.dumps(event.bbox) if isinstance(event.bbox, (list, tuple)) else event.bbox,
                    event.zone_name,
                    event.duration,
                    metadata_json,
                    event.recorded_at,
                ),
            )
            self._conn.commit()
            return True
        except Exception as e:
            print(f"[ERROR] Failed to store event: {e}")
            return False

    def store_batch(self, events: List[StoredEvent]) -> int:
        """Store a batch of events. Returns number successfully stored."""
        stored = 0
        for event in events:
            if self.store(event):
                stored += 1
        return stored

    def get_recent(self,
                   event_type: str = None,
                   source_id: str = None,
                   limit: int = 100,
                   since: float = None) -> List[StoredEvent]:
        """Retrieve recent events, optionally filtered.

        Args:
            event_type: filter by event type (e.g. "motion", "zone_violation")
            source_id: filter by source identifier
            limit: maximum number of events to return
            since: only return events after this timestamp (seconds since epoch)

        Returns:
            List of StoredEvent objects, newest first.
        """
        query = """SELECT event_id, timestamp, source_id, event_type,
                          object_id, class_name, confidence, bbox_text,
                          zone_name, duration, metadata_text, recorded_at
                   FROM events WHERE 1=1"""
        params = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        if source_id:
            query += " AND source_id = ?"
            params.append(source_id)

        if since is not None:
            query += " AND timestamp > ?"
            params.append(since)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(query, params)
        events = []
        for row in cursor:
            event = StoredEvent(
                event_id=row[0],
                timestamp=row[1],
                source_id=row[2],
                event_type=row[3],
                object_id=row[4],
                class_name=row[5],
                confidence=row[6],
                bbox=self._deserialize_bbox(row[7]),
                zone_name=row[8],
                duration=row[9],
                metadata=self._deserialize_metadata(row[10]),
                recorded_at=row[11],
            )
            events.append(event)

        return events

    def get_stats(self) -> Dict[str, Any]:
        """Return storage statistics."""
        cursor = self._conn.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT event_type) as event_types,
                MIN(timestamp) as first_event,
                MAX(timestamp) as last_event
            FROM events
        """)
        row = cursor.fetchone()
        return {
            "total_events": row[0] if row[0] else 0,
            "event_types": row[1] if row[1] else 0,
            "first_event": row[2],
            "last_event": row[3],
            "database_path": self.db_path,
            "retention_days": self.retention_days,
        }

    def prune_old_events(self) -> int:
        """Remove events older than retention_days. Returns number deleted."""
        cutoff = time.time() - (self.retention_days * 24 * 3600)
        cursor = self._conn.execute(
            "DELETE FROM events WHERE timestamp < ?", (cutoff,)
        )
        deleted = cursor.rowcount
        self._conn.commit()
        return deleted

    def _deserialize_bbox(self, bbox_text: str) -> Tuple[float, float, float, float]:
        """Convert stored JSON text back to bbox tuple."""
        try:
            return tuple(json.loads(bbox_text))
        except (json.JSONDecodeError, TypeError):
            return (0.0, 0.0, 0.0, 0.0)

    def _deserialize_metadata(self, metadata_text: str) -> Dict[str, Any]:
        """Convert stored JSON text back to metadata dict."""
        try:
            return json.loads(metadata_text) if metadata_text else {}
        except (json.JSONDecodeError, TypeError):
            return {}


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_event_store(db_path: str = None, retention_days: int = 30) -> EventStore:
    """Create a ready-to-use EventStore instance."""
    return EventStore(db_path=db_path, retention_days=retention_days)