---
name: camera-setup
description: Open and warm up a cv2.VideoCapture reliably, probe for real (non-black) frames before starting the main loop, and handle macOS index enumeration quirks. Use when a VideoCapture call succeeds but returns black/stale frames, when switching between built-in and USB webcams, or when the first ~5 seconds of a pipeline produce zero face detections.
---

# Camera Setup

OpenCV's `cv2.VideoCapture` on macOS (and some Linux configs) is flaky in
specific, reproducible ways. This skill encodes the patterns that make camera
init deterministic.

## The three common failure modes

1. **`isOpened()` returns True but frames are black** — camera is still
   initialising internally. Reads return a zero frame (`frame.mean() < 10`).
2. **Index points at a virtual camera** — macOS reports "Insta360 Virtual
   Camera" at index 0 even when no Insta360 hardware is present. Your code opens
   it, gets black frames, and reports "no face detected" forever.
3. **Stale first frame** — the first `cap.read()` after opening sometimes
   returns a cached frame from a previous session, not a live one.

## The warmup pattern

```python
import cv2, time, face_recognition

def open_camera(index: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    # macOS needs time after open() before the first usable frame
    time.sleep(0.5)
    # Probe: look for a real (non-black) frame and ideally a face on it
    for _ in range(30):
        ok, frame = cap.read()
        if ok and frame is not None and frame.mean() > 10:
            return cap
    cap.release()
    return None
```

For face-specific pipelines, take the probe further: only return once
`face_recognition.face_locations(small_frame)` succeeds at least once. That
guarantees the model is loaded AND the user is in frame before the timed loop
starts.

## Index probing (pick the right camera)

```python
def probe_indices(max_index: int = 4):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        time.sleep(0.5)
        best_mean = 0
        res = None
        for _ in range(10):
            ok, f = cap.read()
            if ok and f is not None and f.mean() > best_mean:
                best_mean = f.mean()
                res = f.shape[:2]
        cap.release()
        yield i, res, best_mean
```

**Rule:** pick the index where `best_mean > 50` (real lighting) AND the
resolution matches a known physical camera. Skip any `best_mean < 10` — those
are black-returning virtual cameras.

Cross-check with `system_profiler SPCameraDataType` on macOS to confirm which
physical camera you're talking to.

## macOS-specific quirks to remember

- **Virtual cameras occupy index 0.** If you have Insta360, Snap Camera, OBS
  Virtual Camera installed, they'll register at a low index even when inactive.
- **Plugging in a USB webcam shifts indices.** Osmo plugged in might become
  index 0; MacBook camera shifts to 2. Don't hardcode — probe at startup.
- **DJI Osmo Pocket 3** defaults to portrait (1080×1920) over USB webcam.
  Either rotate the gimbal or handle portrait orientation in your pipeline.
- **`rgb=(0,0,0)` can be a no-op** on some downstream actuators that share
  encoding paths; unrelated to VideoCapture but worth a mental bookmark.

## Pre-flight for live demos

Wire an "am I really seeing frames?" probe into the startup sequence and
print wall-clock milestones. Invisible 5-second camera init while the audience
watches a blank terminal is the worst possible way to open a talk.

## Related

- `frame-skip-policy` — how to balance per-frame work against inference cost
- In the `face-recognition-calibration` plugin: `face-recognition-persistence` — absorb detection dropouts that survive camera warmup
