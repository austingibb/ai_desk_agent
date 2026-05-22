"""Lightweight scene change detection using Pillow + numpy phase correlation."""

import math
import time
import numpy as np
from PIL import Image, ImageFilter


def _phase_correlate(prev_arr, curr_arr):
    """Compute the global pixel shift between two grayscale numpy arrays.

    Returns (dy, dx) — how many pixels curr is shifted relative to prev.
    """
    f_prev = np.fft.fft2(prev_arr)
    f_curr = np.fft.fft2(curr_arr)
    cross = f_curr * np.conj(f_prev)
    magnitude = np.abs(cross)
    magnitude[magnitude == 0] = 1
    corr = np.fft.ifft2(cross / magnitude).real
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    dy, dx = peak
    # Wrap: if shift > half the image, it's negative
    if dy > corr.shape[0] // 2:
        dy -= corr.shape[0]
    if dx > corr.shape[1] // 2:
        dx -= corr.shape[1]
    return int(dy), int(dx)


def _overlap_crop(prev_arr, curr_arr, dy, dx):
    """Crop both arrays to their overlapping region after a (dy, dx) shift.

    Returns (prev_cropped, curr_cropped) with matching content aligned.
    """
    h, w = prev_arr.shape

    # Vertical: if curr shifted down by dy, the overlap is rows dy..h in curr
    # and rows 0..h-dy in prev (and vice versa for negative)
    if dy >= 0:
        prev_y = slice(0, h - dy)
        curr_y = slice(dy, h)
    else:
        prev_y = slice(-dy, h)
        curr_y = slice(0, h + dy)

    if dx >= 0:
        prev_x = slice(0, w - dx)
        curr_x = slice(dx, w)
    else:
        prev_x = slice(-dx, w)
        curr_x = slice(0, w + dx)

    return prev_arr[prev_y, prev_x], curr_arr[curr_y, curr_x]


class SceneChangeDetector:
    """Compare consecutive camera frames to detect meaningful scene changes.

    Phase correlation detects camera shift (pan/vibration) and crops to the
    overlapping region before comparing. This prevents a camera bump from
    triggering a false positive.

    If the shift is large (>10% of frame), treats it as a real change — something
    significant moved the camera.

    Uses two metrics on the aligned overlap:
    - RMS of pixel differences (catches global changes like lights on/off)
    - Percentage of significantly changed pixels (catches localized motion)

    Reference image only updates on detected change, so gradual drift
    (slow daylight shift, cloud shadows) doesn't accumulate into missed detections.
    """

    def __init__(self, rms_threshold=10.5, pct_threshold=0.04,
                 max_stale_seconds=1800, compare_size=(160, 120),
                 max_shift_pct=0.10):
        self.rms_threshold = rms_threshold
        self.pct_threshold = pct_threshold
        self.max_stale_seconds = max_stale_seconds
        self.compare_size = compare_size
        self.max_shift_pct = max_shift_pct
        self.prev_arr = None
        self.last_changed_time = 0.0

    def _prepare(self, img: Image.Image) -> np.ndarray:
        small = (img
                 .resize(self.compare_size)
                 .convert("L")
                 .filter(ImageFilter.GaussianBlur(radius=2)))
        return np.array(small, dtype=np.float32)

    def check(self, img: Image.Image) -> dict:
        """Check whether the scene has meaningfully changed.

        Returns dict with keys:
            changed: bool
            rms: float — RMS of pixel differences (after shift correction)
            pct_changed: float — fraction of pixels with significant change
            shift: (dy, dx) — detected camera shift in pixels
            reason: str — why the decision was made
        """
        curr_arr = self._prepare(img)
        now = time.time()

        if self.prev_arr is None:
            self.prev_arr = curr_arr
            self.last_changed_time = now
            return {"changed": True, "rms": 0.0, "pct_changed": 0.0,
                    "shift": (0, 0), "reason": "first_frame"}

        # Detect camera shift
        dy, dx = _phase_correlate(self.prev_arr, curr_arr)
        h, w = self.compare_size[1], self.compare_size[0]

        # Large shift = something big happened, treat as change
        if abs(dy) > h * self.max_shift_pct or abs(dx) > w * self.max_shift_pct:
            self.prev_arr = curr_arr
            self.last_changed_time = now
            return {"changed": True, "rms": 0.0, "pct_changed": 0.0,
                    "shift": (dy, dx), "reason": "large_shift"}

        # Crop to overlapping region and compare
        prev_crop, curr_crop = _overlap_crop(self.prev_arr, curr_arr, dy, dx)
        diff_arr = np.abs(curr_crop - prev_crop)

        rms = float(np.sqrt(np.mean(diff_arr ** 2)))
        total_pixels = diff_arr.size
        pct_changed = float(np.count_nonzero(diff_arr > 30) / total_pixels)

        stale = (now - self.last_changed_time) > self.max_stale_seconds
        changed = rms > self.rms_threshold or pct_changed > self.pct_threshold

        if changed:
            reason = "scene_changed"
        elif stale:
            changed = True
            reason = "stale"
        else:
            reason = "no_change"

        if changed:
            self.prev_arr = curr_arr
            self.last_changed_time = now

        return {"changed": changed, "rms": rms, "pct_changed": pct_changed,
                "shift": (dy, dx), "reason": reason}
