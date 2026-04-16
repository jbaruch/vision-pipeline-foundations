# Real-Time Attendance and Mood Monitor

## Problem/Feature Description

A HR-tech startup is building a real-time kiosk that monitors a meeting room and tracks two things continuously: who is present (face recognition against a known-employee database) and the general emotional tone of the room (emotion classification using a ViT model). The kiosk runs on an M-series MacBook with a 30 fps USB webcam.

Early prototypes ran face recognition and emotion classification on every frame. The camera loop dropped to under 5 fps and the UI became unresponsive. The team needs a redesigned capture loop that keeps the camera reading at full rate while running the two expensive inference steps at sensible, lower frequencies. The emotion model is notably slower than the face encoder.

You do not need to implement actual face recognition or emotion models — stub them with functions that sleep to simulate realistic inference latency. Document the chosen inference frequencies as named constants at the top of the script.

## Output Specification

Produce a single Python script `attendance_monitor.py` that:
1. Implements a camera capture loop that reads frames at full rate.
2. Runs a simulated face recognition step and a simulated emotion classification step at different, reduced rates relative to the frame counter — using named constants for the skip intervals.
3. Always downscales the frame before passing it to either inference stub.
4. Only runs emotion classification when a face was previously detected in the current session.
5. Simulates running for 5 seconds (using a frame counter or wall-clock limit) then prints a summary: total frames read, face-recognition calls made, emotion calls made.
6. Writes the summary to a file named `monitor_summary.json`.

The script must be runnable with `python attendance_monitor.py` (no real camera required — use `cv2.VideoCapture(0)` and handle the case where it cannot be opened by using a synthetic black frame of shape 480×640×3).
