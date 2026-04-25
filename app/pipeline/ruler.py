"""
Ruler detection and pixel-to-cm calibration.
Detects the ruler (yellow/white strip on the left edge) and calculates px/cm ratio.
"""

import cv2
import numpy as np


def detect_ruler(image: np.ndarray) -> dict:
    """
    Detect the ruler on the left side of the image.
    Returns:
      {
        "px_per_cm": float,
        "ruler_bbox": (x, y, w, h),
        "detected": bool
      }
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # The ruler is typically yellow/white — detect bright region on left edge
    # Yellow range
    lower_yellow = np.array([15, 80, 120])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # White range (for white rulers)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    ruler_mask = cv2.bitwise_or(yellow_mask, white_mask)

    # Focus on left 15% of image (ruler is usually on the left)
    h, w = ruler_mask.shape
    left_region = ruler_mask[:, :int(w * 0.15)]

    # Find contours in left region
    contours, _ = cv2.findContours(left_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"px_per_cm": 0, "ruler_bbox": (0, 0, 0, 0), "detected": False}

    # Take the longest contour (the ruler body)
    largest = max(contours, key=cv2.contourArea)
    x, y, rw, rh = cv2.boundingRect(largest)

    # The ruler is vertical: height should be much larger than width
    if rh < rw * 3:
        # Try all contours combined
        all_points = np.vstack(contours)
        x, y, rw, rh = cv2.boundingRect(all_points)

    if rh < 100:
        return {"px_per_cm": 0, "ruler_bbox": (x, y, rw, rh), "detected": False}

    # Estimate px_per_cm from ruler length
    # Common rulers: 1m = 100cm, 50cm, etc.
    # We try to detect cm ticks via edge detection on the ruler strip
    ruler_strip = left_region[y:y + rh, x:x + rw]
    px_per_cm = _estimate_px_per_cm_from_ticks(ruler_strip, rh)

    return {
        "px_per_cm": px_per_cm,
        "ruler_bbox": (x, y, rw, rh),
        "ruler_length_px": rh,
        "detected": True
    }


def _estimate_px_per_cm_from_ticks(ruler_strip: np.ndarray, ruler_height: int) -> float:
    """
    Estimate px/cm by detecting periodic tick marks on the ruler.
    Falls back to assuming the ruler is 100cm if detection fails.
    """
    if ruler_strip.size == 0:
        return 0

    # Project to 1D (sum across width)
    profile = np.sum(ruler_strip, axis=1).astype(float)
    if profile.max() == 0:
        return 0

    profile = profile / profile.max()

    # Find peaks (tick marks) — they appear as dips in brightness
    from scipy.signal import find_peaks
    # Invert so ticks become peaks
    inverted = 1.0 - profile
    peaks, properties = find_peaks(inverted, distance=5, height=0.2)

    if len(peaks) >= 3:
        # Median distance between consecutive ticks
        diffs = np.diff(peaks)
        median_tick_px = np.median(diffs)
        # Each tick is 1cm
        if median_tick_px > 3:
            return median_tick_px

    # Fallback: assume a standard 1m (100cm) ruler
    # If the ruler strip is the full ruler, ruler_height ≈ 100cm
    # From the image, the ruler appears to be ~100cm
    px_per_cm_fallback = ruler_height / 100.0
    return px_per_cm_fallback


def px_to_cm(length_px: float, px_per_cm: float) -> float:
    """Convert pixel length to centimeters."""
    if px_per_cm <= 0:
        return 0
    return length_px / px_per_cm
