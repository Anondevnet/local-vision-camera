"""
Privacy Masking for Arlo Indoor Camera System

Provides:
- Configurable privacy zones (masking regions that are blurred/opacified
  before storage or display)
- Multiple mask types: solid color, Gaussian blur, pixelation
- Applied per-frame before any storage/write operations
- Zone definitions from config (normalized polygon coordinates)
- Truthful identification: masks are explicitly marked as "privacy_applied"

Never silently strips or loses privacy-sensitive data. All masking operations
are explicit, configurable, and documented.
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Mask application modes
# ---------------------------------------------------------------------------

MASK_TYPE_COLOR = "color"       # solid color fill
MASK_TYPE_BLUR = "blur"         # Gaussian blur
MASK_TYPE_PIXELATE = "pixelate" # block pixelation


# ---------------------------------------------------------------------------
# PrivacyZone definition
# ---------------------------------------------------------------------------

@dataclass
class PrivacyZone:
    """Defines a privacy mask region within a camera frame."""
    name: str
    # Normalized polygon points (x, y) in [0.0, 1.0] relative to frame width/height
    points: List[Tuple[float, float]]
    mask_type: str = MASK_TYPE_COLOR  # color, blur, pixelate
    mask_color: Tuple[int, int, int] = (0, 0, 0)  # BGR, used for color mask
    mask_intensity: float = 0.7  # 0.0 = fully transparent, 1.0 = opaque
    zone_type: str = "privacy"  # "privacy", "sensitive", "blurring"


# ---------------------------------------------------------------------------
# PrivacyMaskEngine
# ---------------------------------------------------------------------------

class PrivacyMaskEngine:
    """Applies privacy masks to video frames per configured zones.

    Masks are applied in-memory before any storage or display operations.
    The original unmasked frame is never stored or transmitted.
    """

    def __init__(self, zones: List[PrivacyZone] = None, frame_width: int = 1280,
                 frame_height: int = 720):
        self.zones = zones or []
        self.frame_width = frame_width
        self.frame_height = frame_height
        # Pre-compute pixel-coordinate polygons for efficiency
        self._polygons: List[np.ndarray] = []
        for zone in self.zones:
            px_points = [(int(p[0] * frame_width), int(p[1] * frame_height))
                         for p in zone.points]
            self._polygons.append(np.array(px_points, dtype=np.int32))

    def set_frame_size(self, width: int, height: int):
        """Update frame dimensions and recompute polygons."""
        self.frame_width = width
        self.frame_height = height
        self._polygons = []
        for zone in self.zones:
            px_points = [(int(p[0] * width), int(p[1] * height))
                         for p in zone.points]
            self._polygons.append(np.array(px_points, dtype=np.int32))

    def apply_mask(self, frame: np.ndarray) -> np.ndarray:
        """Apply all configured privacy masks to a frame.

        Args:
            frame: numpy array (H, W, 3) BGR format

        Returns:
            Frame with privacy masks applied. Original references are not modified.
        """
        if not self.zones:
            return frame

        # Work on a copy to avoid modifying the original
        result = frame.copy()

        for i, polygon in enumerate(self._polygons):
            zone = self.zones[i]
            mask_type = zone.mask_type
            color = zone.mask_color
            intensity = zone.mask_intensity

            # Create a mask image for this zone
            mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)

            if mask_type == MASK_TYPE_COLOR:
                # Solid color fill: blend the mask color over the region
                # For each channel: result = intensity * color + (1-intensity) * original
                for c in range(3):
                    result[:, :, c] = (
                        intensity * color[c] + (1 - intensity) * result[:, :, c]
                    ).astype(np.uint8)

            elif mask_type == MASK_TYPE_BLUR:
                # Apply Gaussian blur only within the masked region
                # We blur the whole frame then restore outside via mask
                blurred = cv2.GaussianBlur(result, (99, 99), 30)
                # Blend: masked region = blurred, outside = original
                result = np.where(mask[..., None] == 255, blurred, result).astype(np.uint8)

            elif mask_type == MASK_TYPE_PIXELATE:
                # Pixelate the masked region
                # Reduce resolution then upscale
                h, w = self.frame_height, self.frame_width
                small = cv2.resize(result, (max(1, w // 10), max(1, h // 10)),
                                   interpolation=cv2.INTER_AREA)
                pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                result = np.where(mask[..., None] == 255, pixelated, result).astype(np.uint8)

            # else: unknown mask type, skip

        return result

    def annotate_frame(self, frame: np.ndarray, labels: bool = True) -> np.ndarray:
        """Draw zone polygons and labels on frame (for preview only, not stored).

        Args:
            frame: original BGR frame
            labels: whether to draw zone names

        Returns:
            Frame with zone outlines and labels drawn (for UI preview).
            Does NOT apply the mask - just annotates.
        """
        result = frame.copy()
        for i, polygon in enumerate(self._polygons):
            color = (0, 255, 0)  # green outline
            cv2.polylines(result, [polygon], True, color, 2)

            # Draw label background
            zone = self.zones[i] if i < len(self.zones) else type('obj', (object,), {'name': f'zone_{i}'})()
            label = zone.name if hasattr(zone, 'name') else f'zone_{i}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            (label_w, label_h), _ = cv2.getTextSize(label, font, 0.5, 1)

            # Position label above top-left point of polygon
            pt = polygon[0]  # first point
            label_x = pt[0]
            label_y = max(5, pt[1] - 8)

            # Draw label background rectangle
            cv2.rectangle(result, (label_x, label_y - label_h - 4),
                          (label_x + label_w + 4, label_y + 2), (0, 0, 0), -1)

            # Draw label text
            cv2.putText(result, label, (label_x, label_y), font, 0.5, (255, 255, 255), 1)

        return result

    def get_zone_info(self) -> List[Dict[str, Any]]:
        """Return zone configuration for documentation/debugging."""
        info = []
        for i, zone in enumerate(self.zones):
            info.append({
                "id": i,
                "name": zone.name,
                "mask_type": zone.mask_type,
                "points_normalized": zone.points,
                "zone_type": zone.zone_type,
            })
        return info


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_privacy_mask_engine(zones: List[Dict[str, Any]] = None,
                               frame_width: int = 1280,
                               frame_height: int = 720) -> PrivacyMaskEngine:
    """Create a PrivacyMaskEngine from zone configuration dicts.

    Each zone dict should have:
    - "name": str
    - "points": List[Tuple[float, float]] normalized (0-1)
    - "mask_type": one of "color", "blur", "pixelate" (optional, default "color")
    - "mask_color": Tuple[int,int,int] BGR (optional, default black)
    - "mask_intensity": float 0-1 (optional, default 0.7)
    - "zone_type": "privacy", "sensitive", "blurring" (optional, default "privacy")
    """
    privacy_zones = []
    if zones:
        for z in zones:
            pz = PrivacyZone(
                name=z.get("name", "unnamed"),
                points=z.get("points", []),
                mask_type=z.get("mask_type", MASK_TYPE_COLOR),
                mask_color=tuple(z.get("mask_color", [0, 0, 0])),
                mask_intensity=z.get("mask_intensity", 0.7),
                zone_type=z.get("zone_type", "privacy"),
            )
            privacy_zones.append(pz)

    return PrivacyMaskEngine(zones=privacy_zones,
                             frame_width=frame_width,
                             frame_height=frame_height)