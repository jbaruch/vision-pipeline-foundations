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
    # macOS needs 1s+ after open() before real frames arrive.
    # 0.5s is NOT enough for USB webcams (DJI Osmo, Logitech) — they
    # go dormant between processes and need a full re-handshake.
    time.sleep(1.0)
    # Read-until-non-black: some cameras return 5-20 dark frames before
    # the sensor settles. Threshold mean > 30 (not 10 — dim rooms are
    # 30-50; black frames are < 10).
    for _ in range(30):
        ok, frame = cap.read()
        if ok and frame is not None and frame.mean() > 30:
            return cap
        time.sleep(0.1)   # pace the probe — hammering cap.read() can stall
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

## USB webcams need the macOS event loop (DJI Osmo, Logitech, etc.)

Some USB UVC cameras — confirmed on DJI Osmo Pocket 3 — return **dark frames
(mean ≈ 25) even though `isOpened()` is True** unless the macOS event loop is
pumped. OpenCV's `cap.read()` in a headless script will silently get zeros.

The fix: create a named window and call `cv2.imshow` + `cv2.waitKey` on each
frame. This pumps the NSApplication event loop that macOS needs to deliver
video buffers from the UVC driver.

```python
cap = cv2.VideoCapture(index)
cv2.namedWindow('_cam', cv2.WINDOW_NORMAL)
time.sleep(1.0)
for _ in range(30):
    ok, f = cap.read()
    if ok and f is not None:
        cv2.imshow('_cam', f)
        cv2.waitKey(30)
        if f.mean() > 30:
            break
    time.sleep(0.1)
```

**Do NOT call `cap.set(CAP_PROP_FRAME_WIDTH/HEIGHT)` on DJI Osmo.** The
resolution request triggers a UVC re-negotiation that resets the Osmo's
digital zoom and exposure to defaults. Let the camera use whatever resolution
it was already set to. If you need a specific resolution, set it physically
on the Osmo's touchscreen before the script runs.

## Subprocess camera sharing doesn't work

If your pipeline spawns sub-agents or worker processes that each open their
own `VideoCapture`, only ONE process at a time can hold the camera on macOS.
Others will get dark frames.

**Pattern:** have the parent process own the camera, capture frames to disk
or shared memory, and let sub-agents read from there. Or run the full
vision pipeline in the parent and only delegate non-camera work to subs.

## Pre-flight for live demos

Wire an "am I really seeing frames?" probe into the startup sequence and
print wall-clock milestones. Invisible 5-second camera init while the audience
watches a blank terminal is the worst possible way to open a talk.

## Reference implementation

[`scripts/camera_setup.py`](../../scripts/camera_setup.py) — `open_camera(index)`, `probe_indices()`, and `capture_to_disk()`. Import directly instead of re-implementing.

## Related

- `frame-skip-policy` — how to balance per-frame work against inference cost
- In the `face-recognition-calibration` plugin: `face-recognition-persistence` — absorb detection dropouts that survive camera warmup
