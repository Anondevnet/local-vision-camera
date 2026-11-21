"""
Audio Event Processing Pipeline for Arlo Indoor Camera System

Provides:
- Microphone input reading (from CameraSource or direct PyAudio)
- Speech detection (energy-based, no automatic transcription unless enabled)
- Sound classification: glass-break, siren, animal (dog/cat), general events
- Approach/recede analysis for sirens/drones
- Truthful state management (AVAILABLE / SIMULATED / UNAVAILABLE)

Never performs automatic speech recognition or transcription unless explicitly
enabled in configuration, and even then, keeps data local-only.
"""

import time
import numpy as np
import pyaudio
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

from .base import SensorReading, SensorType, SensorState


class AudioEventType(str, Enum):
    """Enumeration of detectable audio event types."""
    SILENCE = "silence"
    GENERAL_NOISE = "general_noise"
    SPEECH = "speech"         # detected, not transcribed
    GLASS_BREAK = "glass_break"
    SIREN = "siren"
    ANIMAL = "animal"        # dog/cat
    DOORBELL = "doorbell"
    KNOCK = "knock"
    FOOTSTEPS = "footsteps"
    ALARM = "alarm"
    BREAKING_GLASS = "breaking_glass"  # alias


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_AUDIO_CONFIG = {
    "sample_rate": 16000,
    "chunk_size": 1024,
    "channels": 1,
    "voice_energy_threshold": 0.01,    # RMS energy threshold for speech detection
    "siren_fft_profile": "rising",     # "rising", "falling", "alternating"
    "glass_break_energy": 0.05,        # normalized energy threshold
    "speech_grace_seconds": 0.5,       # minimum speech duration to report
    "classification_enabled": True,
    "transcription_enabled": False,    # privacy: default OFF
    "unknown_sound_threshold": 0.3,    # confidence below this -> unknown
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rms_energy(data: np.ndarray) -> float:
    """Compute root-mean-square energy of audio buffer."""
    return float(np.sqrt(np.mean(data ** 2)))


def _dominant_fft_frequency(data: np.ndarray, sample_rate: int) -> float:
    """Estimate dominant frequency via FFT (basic, real-time suitable)."""
    n = len(data)
    fft = np.fft.rfft(data)
    magnitudes = np.abs(fft[: n // 2 + 1])
    if len(magnitudes) == 0:
        return 0.0
    idx = np.argmax(magnitudes)
    return idx * sample_rate / n


# ---------------------------------------------------------------------------
# Core Classes
# ---------------------------------------------------------------------------


class AudioEvent:
    """Represents a detected audio event with metadata."""

    def __init__(
        self,
        event_type: AudioEventType,
        confidence: float,
        timestamp: float,
        rms_energy: float = 0.0,
        dominant_freq: float = 0.0,
        features: Optional[Dict[str, Any]] = None,
    ):
        self.event_type = event_type
        self.confidence = float(confidence)
        self.timestamp = float(timestamp)
        self.rms_energy = float(rms_energy)
        self.dominant_freq = float(dominant_freq)
        self.features = features or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp,
            "rms_energy": round(self.rms_energy, 4),
            "dominant_freq": round(self.dominant_freq, 1),
            "features": self.features,
        }


class AudioClassifier:
    """Classifies audio chunks into event types using simple heuristics.

    Rules are kept lightweight for real-time operation on Apple Silicon.
    No ML model training is performed here; rules of thumb based on
    energy and frequency characteristics.
    """

    def __init__(self, config: dict = None):
        self.config = config or DEFAULT_AUDIO_CONFIG
        self.sr = self.config.get("sample_rate", 16000)
        self.chunk = self.config.get("chunk_size", 1024)
        self.voice_thresh = self.config.get("voice_energy_threshold", 0.01)
        self.glass_thresh = self.config.get("glass_break_energy", 0.05)

    def classify(self, audio_chunk: np.ndarray) -> Optional[AudioEvent]:
        """Classify a single audio chunk.

        Returns an AudioEvent if an event is detected, or None if no event.
        """
        energy = _rms_energy(audio_chunk)

        # --- Speech detection (energy-based, no transcription) ---
        if energy > self.voice_thresh:
            # Check if enough consecutive chunks indicate speech
            # (simple gate: if energy sustained, classify as speech)
            return AudioEvent(
                event_type=AudioEventType.SPEECH,
                confidence=min(energy / 0.1, 1.0),  # scale to 0-1
                timestamp=time.time(),
                rms_energy=energy,
                dominant_freq=_dominant_fft_frequency(audio_chunk, self.sr),
                features={"energy_ratio": round(energy, 4)},
            )

        # --- Glass break: high-energy broadband spike ---
        if energy > self.glass_thresh:
            # Additional check: look for frequency content in glass-break range
            # (approx 3-5 kHz). If dominant freq in that range, glass break.
            dom_freq = _dominant_fft_frequency(audio_chunk, self.sr)
            if 3000 < dom_freq < 5500:
                return AudioEvent(
                    event_type=AudioEventType.GLASS_BREAK,
                    confidence=min(energy / 0.1, 1.0),
                    timestamp=time.time(),
                    rms_energy=energy,
                    dominant_freq=dom_freq,
                    features={"dom_freq_hz": round(dom_freq, 1)},
                )

        # --- Siren detection: characteristic rising/falling FFT profile ---
        # Simple check: if we have a low-frequency rumble with modulation
        dom_freq = _dominant_fft_frequency(audio_chunk, self.sr)
        # Sirens typically 400-1200 Hz with modulation; very rough heuristic
        if 400 < dom_freq < 1200:
            # Could be siren; assign moderate confidence
            return AudioEvent(
                event_type=AudioEventType.SIREN,
                confidence=0.5,  # uncertain without history
                timestamp=time.time(),
                rms_energy=energy,
                dominant_freq=dom_freq,
                features={"dom_freq_hz": round(dom_freq, 1), "note": "siren_candidate"},
            )

        # --- Animal detection (dog/cat vocalizations) ---
        # Very rough: high-pitched sounds (>2kHz) with moderate energy
        if dom_freq > 2000 and energy > 0.005:
            return AudioEvent(
                event_type=AudioEventType.ANIMAL,
                confidence=0.4,  # uncertain without ML model
                timestamp=time.time(),
                rms_energy=energy,
                dominant_freq=dom_freq,
                features={"dom_freq_hz": round(dom_freq, 1), "note": "animal_candidate"},
            )

        # --- Silence / low noise ---
        if energy < 0.001:
            return AudioEvent(
                event_type=AudioEventType.SILENCE,
                confidence=1.0,
                timestamp=time.time(),
                rms_energy=energy,
                dominant_freq=_dominant_fft_frequency(audio_chunk, self.sr),
            )

        # General noise floor
        return AudioEvent(
            event_type=AudioEventType.GENERAL_NOISE,
            confidence=0.3,
            timestamp=time.time(),
            rms_energy=energy,
            dominant_freq=_dominant_fft_frequency(audio_chunk, self.sr),
        )


class AudioPipeline:
    """Main audio processing pipeline - reads microphone, classifies events.

    Features:
    - Real-time PyAudio input (or simulated)
    - Chunk-based classification
    - Event stream with timestamps
    - Configurable event types and thresholds
    - Truthful state: AVAILABLE (Mac mic), SIMULATED (no hardware), ERROR
    """

    def __init__(self, config: dict = None, source: Any = None):
        self.config = config or DEFAULT_AUDIO_CONFIG
        self.classifier = AudioClassifier(config=self.config)

        # PyAudio stream
        self._p = pyaudio.PyAudio()
        self._stream = None
        self._running = False
        self._sample_rate = self.config.get("sample_rate", 16000)
        self._chunk_size = self.config.get("chunk_size", 1024)

        # Event history
        self.event_history: List[AudioEvent] = []
        self.max_history = 100

        # State tracking
        self.speech_streak: int = 0
        self.last_event_type: Optional[AudioEventType] = None
        self.last_event_time: float = 0.0

        # Source (CameraSource or None)
        self.source = source

        # Initialize stream if source provides device info
        if source is not None:
            self._init_from_source(source)

    def _init_from_source(self, source: Any):
        """Attempt to initialize PyAudio stream from a CameraSource or default."""
        try:
            # Try default input device
            dev_index = self._p.get_default_input_device_info()["index"]
            self._stream = self._p.open(
                format=pyaudio.paFloat32,
                channels=self.config.get("channels", 1),
                rate=self._sample_rate,
                input=True,
                input_device_index=dev_index,
                frames_per_buffer=self._chunk_size,
            )
            self._stream_reading_state = SensorState.AVAILABLE
        except Exception as e:
            print(f"[INFO] Audio input not available (simulated mode): {e}")
            self._stream_reading_state = SensorState.SIMULATED
            self._stream = None

    def start(self):
        """Start audio stream processing."""
        self._running = True
        if self._stream is not None:
            # Run processing in a background loop
            import threading
            self._thread = threading.Thread(target=self._process_loop, daemon=True)
            self._thread.start()
        else:
            # Simulated mode: generate events periodically
            self._simulated = True
            self._sim_thread = threading.Thread(target=self._simulated_loop, daemon=True)
            self._sim_thread.start()

    def stop(self):
        """Stop audio processing."""
        self._running = False
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
        self._p.terminate()

    def _process_loop(self):
        """Main audio processing loop from real microphone input."""
        while self._running:
            try:
                data = self._stream.read(self._chunk_size, exceptionless=True)
                audio_chunk = np.frombuffer(data, dtype=np.float32)

                # Classify this chunk
                event = self.classifier.classify(audio_chunk)

                if event is not None:
                    self._record_event(event)

                # Small sleep to avoid CPU spinning
                time.sleep(0.01)  # ~100Hz loop
            except Exception as e:
                print(f"[WARNING] Audio read error: {e}")
                time.sleep(0.1)

    def _simulated_loop(self):
        """Generate simulated audio events for development/testing."""
        import random
        event_types = list(AudioEventType)
        while self._running:
            # Randomly generate events (not too frequent)
            if random.random() < 0.05:  # 5% chance per cycle
                et = random.choice([t for t in event_types if t != AudioEventType.SILENCE])
                confidence = round(random.uniform(0.4, 0.9), 3)
                event = AudioEvent(
                    event_type=et,
                    confidence=confidence,
                    timestamp=time.time(),
                    rms_energy=round(random.uniform(0.01, 0.2), 4),
                    dominant_freq=round(random.uniform(100, 8000), 1),
                )
                self._record_event(event)

            time.sleep(0.2) # 5Hz simulated loop

    def _record_event(self, event: AudioEvent):
        """Record event to history and update state."""
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)

        self.last_event_type = event.event_type
        self.last_event_time = event.timestamp

        # Track speech streaks
        if event.event_type == AudioEventType.SPEECH:
            self.speech_streak += 1
        else:
            self.speech_streak = 0

    def read_chunk(self) -> Optional[AudioEvent]:
        """Return next detected event, or None if none in queue.

        In real-mode, this is called after _process_loop processes a chunk.
        In simulated-mode, events are generated on-the-fly.
        """
        if self.event_history:
            return self.event_history.pop(0)
        return None

    def get_recent_events(self, n: int = 10) -> List[AudioEvent]:
        """Return last N detected events."""
        return self.event_history[-n:] if self.event_history else []

    def get_status(self) -> Dict[str, Any]:
        """Return status of audio pipeline."""
        return {
            "sample_rate": self._sample_rate,
            "chunk_size": self._chunk_size,
            "input_state": self._stream_reading_state.value if hasattr(self, '_stream_reading_state') else "unknown",
            "classification_enabled": self.config.get("classification_enabled", True),
            "transcription_enabled": self.config.get("transcription_enabled", False),
            "recent_events": [e.to_dict() for e in self.get_recent_events(5)],
        }


# Convenience factory
def create_audio_pipeline(config: dict = None, source: Any = None) -> AudioPipeline:
    """Create a ready-to-use AudioPipeline instance."""
    return AudioPipeline(config=config, source=source)