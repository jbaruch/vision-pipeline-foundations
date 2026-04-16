"""
Camera setup helpers for macOS + USB webcams (DJI Osmo, Logitech, etc.).

Hard rules encoded:
- Wait 1.0s after VideoCapture() before first real read.
- namedWindow + imshow + waitKey(30) pumps the macOS event loop (required
  for USB UVC cameras to deliver frames).
- Probe until frame.mean() > 30 before returning.
- Do NOT cap.set(FRAME_WIDTH/HEIGHT) on DJI Osmo — resets zoom/exposure.
- Only ONE process can hold a camera on macOS — parent owns, subs read from disk.
"""

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
    """Open a cv2.VideoCapture with macOS-safe warmup.

    Parameters
    ----------
    index : camera index (probe first — indices shift when USB devices plug/unplug)
    headless : if True, skip namedWindow/imshow. Only works for built-in cameras;
               USB webcams on macOS will return dark frames without the event loop.
    window_name : name of the cv2 window (ignored if headless)

    Returns
    -------
    An open, warmed-up VideoCapture, or None if no usable frames arrived.
    """
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
    """Probe all camera indices and return info about each.

    Returns list of dicts with keys: index, resolution, mean, has_face.
    """
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
                    resolution = (f.shape[1], f.shape[0])  # (width, height)
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
    """Capture n_frames to disk for sub-agents to read.

    Parent owns the camera. Sub-agents import frames from output_dir instead
    of opening their own VideoCapture (macOS blocks concurrent access).

    Returns number of frames actually saved.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = open_camera(index, window_name=window_name)
    if cap is None:
        return 0
    saved = 0
    for _ in range(n_frames + 20):  # overshoot to skip warmup duds
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


if __name__ == "__main__":
    print("Probing cameras...")
    for info in probe_indices(check_face=True):
        tag = "FACE" if info["has_face"] else ("usable" if info["usable"] else "dark")
        print(f"  index {info['index']}: {info['resolution']}  mean={info['mean']}  {tag}")
