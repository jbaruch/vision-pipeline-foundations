# vision-pipeline-foundations

A [Tessl](https://tessl.io) plugin of foundation-layer patterns for any
OpenCV + dlib CV pipeline. Not about a specific detector or classifier —
about the hygiene you need below them.

## Skills

| Skill | Purpose |
|---|---|
| `camera-setup` | Reliable `cv2.VideoCapture` init with warmup, real-frame probe, macOS index-enumeration quirks. |
| `frame-skip-policy` | Run expensive inference (dlib, ViT) at 1-of-N capture rate. Keeps the camera loop responsive. |

## Install

```bash
tessl install jbaruch/vision-pipeline-foundations
```

## Why this matters

Nearly every "the pipeline isn't working" bug in a demo-facing CV app
traces back to one of:

1. The camera is opened but returning black frames (warmup).
2. You opened index 0 which is a virtual camera, not the real one (probe).
3. Inference is running on every frame and stalling the loop (frame-skip).

Encode these into a plugin so every new pipeline starts from the right
place.

## Composes with

- `face-recognition-calibration` — producer-side CV model tuning, enrollment, persistence
- `iot-actuator-patterns` — downstream actuator control with debounce + quantization

## License

MIT — see `LICENSE`.
