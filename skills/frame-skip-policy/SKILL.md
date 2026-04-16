---
name: frame-skip-policy
description: Run expensive per-frame inference (face recognition, emotion classification, ViT embeddings) at a fraction of the capture rate so the camera loop stays responsive. Use when designing a vision pipeline that combines high-rate capture (30fps+) with heavy per-frame work (dlib, ViT, DeepFace) and you don't need every frame to be inferred.
---

# Frame-Skip Policy

A webcam captures at 30 fps. A dlib ResNet face-encoding pass costs ~80ms.
A ViT emotion classifier costs ~100–200ms. Running either on every frame
stalls the camera loop to 5–10 fps and trashes the UX.

The answer is **frame skipping** — run the heavy inference on 1-of-N frames
and keep the lighter work (read, resize) on every frame.

## The pattern

```python
FACE_RECOGNITION_EVERY = 3   # every 3rd frame → ~10 Hz inference at 30 fps capture
EMOTION_EVERY = 10           # every 10th frame → 3 Hz — emotion changes slowly anyway

frames = 0
while running:
    ok, frame = cap.read()
    if not ok: break
    frames += 1

    # Always downscale (cheap)
    small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Heavy: face recognition — every 3rd frame
    if frames % FACE_RECOGNITION_EVERY == 0:
        locs = face_recognition.face_locations(small)
        encs = face_recognition.face_encodings(small, locs)
        # ...

    # Heavier: emotion classification — every 10th frame, and only when face present
    if frames % EMOTION_EVERY == 0 and last_face_crop is not None:
        emotion = classify_emotion(last_face_crop)
        # ...
```

## Picking N

- **Face detection + recognition (dlib HOG + ResNet):** N=3 → ~10 Hz. Combined
  with a 0.25× downscale this typically runs real-time on an M-series Mac.
- **Emotion classification (ViT):** N=10 → 3 Hz. Emotions don't change faster
  than that; users won't notice the lower rate.
- **Object detection (YOLO):** depends on model size — N=2 for YOLOv8n,
  higher for larger variants.

## When frame skip breaks

If the subject is moving fast AND inference is skipped, you get phantom
trails in the output (face detected at position A for 3 frames, then
position D next inference). Solutions:

- **Interpolate state** between inferences (linear on bounding boxes).
- **Drop the skip** during high-motion frames (diff between consecutive
  small frames; skip N lower when motion is high).
- **Just accept the artifact** — most pipelines don't need sub-100ms tracking.

## Composition with other patterns

- Pair with `face-recognition-persistence` (producer-side stability) —
  frame-skip naturally drops individual miss rates because 1 "detect" every
  3 frames still feeds 10 decisions per second, easier to stabilise.
- Pair with `debounce-controller` (actuator-side) — the producer's effective
  rate is now 10 Hz, debounce tick at 0.4s sees ~4 samples per tick, plenty
  of data for the stability filter to work with.

## Related

- `camera-setup` — make sure you're actually reading frames before tuning skip
- In the `face-recognition-calibration` plugin: `face-recognition-persistence`
- In the `iot-actuator-patterns` plugin: `debounce-controller`
