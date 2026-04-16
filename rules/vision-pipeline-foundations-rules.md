# Vision Pipeline Foundations Rules

Hygiene for OpenCV + dlib CV pipelines.

## Camera setup (→ `camera-setup` skill)
- **`cv2.VideoCapture(i)` returning OK + black frames is normal on macOS** — wait 0.5 s after open and probe for real frames before entering the main loop.
- **Virtual cameras sit on low indices.** Insta360, Snap Camera, OBS Virtual Camera. Skip any index where `frame.mean() < 10`.
- **Probe, don't hardcode.** Plugging in a USB webcam reshuffles indices.
- **macOS camera app list:** `system_profiler SPCameraDataType`.

## Frame-skip policy (→ `frame-skip-policy` skill)
- Face recognition: **every 3rd frame** (≈10 Hz inference at 30 fps capture).
- Emotion classification: **every 10th frame** (≈3 Hz, emotions change slowly).
- Always downscale before heavy inference (`cv2.resize(frame, (0,0), fx=0.25, fy=0.25)`).

Full context in `skills/camera-setup/SKILL.md` and `skills/frame-skip-policy/SKILL.md`.
