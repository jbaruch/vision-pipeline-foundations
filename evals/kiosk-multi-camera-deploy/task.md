# Multi-Worker Face Encoding Pipeline

## Problem/Feature Description

A security startup is building a pipeline that captures frames from a reception-area webcam and farms face-encoding work out to a pool of worker processes to maximise throughput on an M-series MacBook. The original engineer assumed multiple workers could each open the same physical webcam directly. In testing on macOS, all but one worker returned dark frames — a problem that only showed up at demo time.

You need to redesign the pipeline so that the camera is opened and probed correctly once in the parent process, frames are captured to disk, and the worker pool reads from disk rather than from the camera directly. The main concern is that worker processes cannot access the camera directly on macOS. The pipeline must also correctly select which camera index to use at startup, since the machine has virtual cameras (OBS, Snap Camera) installed alongside the real webcam.

You have a helper library at `scripts/camera_setup.py` (relative to this file) that you may use.

## Output Specification

Produce a Python script `encoding_pipeline.py` that:
1. Probes available camera indices and selects the best physical camera (skipping virtual cameras with dark frames).
2. Uses the parent process to capture exactly 6 frames from that camera to a local directory `captured_frames/`.
3. Spawns 2 worker processes (using `multiprocessing`) that each read the captured frames from disk and simulate face encoding (stub with a function that returns a dummy list of floats and prints the filename it processed).
4. Writes a report file `pipeline_report.json` listing: chosen camera index, frames captured, frames processed by workers.
5. Cleans up the `captured_frames/` directory after the workers finish.

Include a brief inline comment explaining why the camera is not opened in the worker processes.

The script must be runnable with `python encoding_pipeline.py`. Handle the case where no usable camera is found by writing `pipeline_report.json` with `"camera_found": false` and exiting cleanly.

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
