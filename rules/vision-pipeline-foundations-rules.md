# Vision Pipeline Foundations Rules

Hard constraints for OpenCV + dlib CV pipelines on macOS.

## Camera setup (→ `camera-setup` skill)
- **USB webcams on macOS need the event loop.** `cv2.namedWindow` + `cv2.imshow` + `cv2.waitKey(30)` on every frame. Without this, USB cameras (DJI Osmo, Logitech) return dark frames silently.
- **Wait 1.0 s after `cv2.VideoCapture()`** before first real read. 0.5 s is NOT enough for USB webcams.
- **Probe until `frame.mean() > 30`** before starting inference. Some cameras return 5–20 dark warmup frames.
- **NEVER `cap.set(FRAME_WIDTH/HEIGHT)` on DJI Osmo.** Resets digital zoom and exposure to defaults.
- **NEVER share a camera across subprocesses on macOS.** Only one process can hold a VideoCapture. Parent owns camera, sub-agents read from disk/shared state.
- **Virtual cameras sit on low indices.** Insta360, Snap Camera, OBS Virtual Camera — skip any index where `frame.mean() < 10`.
- **Probe, don't hardcode.** Plugging in a USB webcam reshuffles indices.
- **macOS camera app list:** `system_profiler SPCameraDataType`.

## Frame-skip policy (→ `frame-skip-policy` skill)
- Face recognition: **every 3rd frame** (≈10 Hz inference at 30 fps capture).
- Emotion classification: **every 10th frame** (≈3 Hz, emotions change slowly).
- Always downscale before heavy inference (`cv2.resize(frame, (0,0), fx=0.25, fy=0.25)`).

Full context in `skills/camera-setup/SKILL.md` and `skills/frame-skip-policy/SKILL.md`.
