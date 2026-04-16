# Conference Demo Camera Script

## Problem/Feature Description

You are preparing a live coding conference talk in which you demonstrate a real-time face detection pipeline on a MacBook. The demo opens a webcam, waits until the camera is reliably delivering frames, then prints a running count of faces detected per second to the terminal for 30 seconds before exiting cleanly.

Past rehearsals went poorly: the first several seconds showed "0 faces detected" even though the presenter was clearly in front of the camera, and the audience saw a frozen blank terminal before anything appeared. The root cause turned out to be the MacBook's DJI Osmo USB webcam not being ready immediately after `cv2.VideoCapture()` was called, and the pipeline starting inference before usable frames arrived.

You have a helper library already available at `scripts/camera_setup.py` (relative to this file) that you may use. The script needs to work on macOS with a USB webcam connected.

## Output Specification

Produce a single Python script named `demo_camera.py` that:
1. Selects an appropriate camera (do not hardcode index 0).
2. Waits until the camera is reliably delivering real frames before starting inference.
3. Counts detected faces on each processed frame using `face_recognition.face_locations()`.
4. Prints a wall-clock timestamp and face-count line to stdout whenever the count changes, and once per second at minimum.
5. Runs for 30 seconds then exits, releasing all resources.
6. Prints at least one diagnostic line during camera initialisation so that startup progress is visible in the terminal.

The script must be runnable with `python demo_camera.py`. Include brief inline comments explaining the warmup strategy.

## Input Files

The following file is provided as an input. Extract it before beginning.

=============== FILE: scripts/camera_setup.py ===============
from __future__ import annotations

import time
from pathlib import Path

import cv2


MIN_FRAME_MEAN = 30
WARMUP_SLEEP = 1.0
PROBE_READS = 30
PROBE_INTERVAL = 0.1


def open_camera(
    index: int,
    *,
    headless: bool = False,
    window_name: str = "_cam",
) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None

    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    time.sleep(WARMUP_SLEEP)

    for _ in range(PROBE_READS):
        ok, frame = cap.read()
        if ok and frame is not None:
            if not headless:
                cv2.imshow(window_name, frame)
                cv2.waitKey(30)
            if frame.mean() > MIN_FRAME_MEAN:
                return cap
        time.sleep(PROBE_INTERVAL)

    cap.release()
    if not headless:
        cv2.destroyWindow(window_name)
    return None


def probe_indices(
    max_index: int = 4,
    *,
    check_face: bool = False,
) -> list[dict]:
    results = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        time.sleep(0.5)
        best_mean = 0.0
        resolution = None
        has_face = False
        for _ in range(10):
            ok, f = cap.read()
            if ok and f is not None:
                if f.mean() > best_mean:
                    best_mean = f.mean()
                    resolution = (f.shape[1], f.shape[0])
                if check_face and f.mean() > MIN_FRAME_MEAN and not has_face:
                    try:
                        import face_recognition
                        small = cv2.resize(f, (0, 0), fx=0.25, fy=0.25)
                        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                        if face_recognition.face_locations(rgb):
                            has_face = True
                    except ImportError:
                        pass
        cap.release()
        results.append({
            "index": i,
            "resolution": resolution,
            "mean": round(best_mean, 1),
            "has_face": has_face,
            "usable": best_mean > 50,
        })
    return results


def capture_to_disk(
    index: int,
    output_dir: str | Path,
    n_frames: int = 8,
    *,
    window_name: str = "_capture",
) -> int:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = open_camera(index, window_name=window_name)
    if cap is None:
        return 0
    saved = 0
    for _ in range(n_frames + 20):
        ok, f = cap.read()
        if ok and f is not None and f.mean() > MIN_FRAME_MEAN:
            cv2.imwrite(str(out / f"frame_{saved:03d}.jpg"), f)
            saved += 1
            if saved >= n_frames:
                break
        cv2.waitKey(30)
    cap.release()
    cv2.destroyAllWindows()
    return saved
