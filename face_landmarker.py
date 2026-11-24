from pathlib import Path
from time import monotonic_ns

import csv
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


model_path = Path(__file__).with_name("face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_buffer=model_path.read_bytes()),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)

window = "Face Landmarker"
cheek_region = [
    [117, 100, 205, 187],
    [346, 329, 425, 411],
]
samples = []
with vision.FaceLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        raise SystemExit("Не вдалось відкрити камеру")

    cv2.namedWindow(window)
    start = monotonic_ns()
    previous_timestamp = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Не вдалось отримати кадр")
            break

        timestamp = max((monotonic_ns() - start) // 1_000_000, previous_timestamp + 1)
        previous_timestamp = timestamp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(image, timestamp)

        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        for face in result.face_landmarks:
            for region in cheek_region:
                points = []

                for index in region:
                    point = face[index]
                    x = int(point.x * width)
                    y = int(point.y * height)
                    points.append([x, y])

                polygon = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [polygon], 255)
                cv2.polylines(frame, [polygon], True, (0, 255, 0), 2)

        if cv2.countNonZero(mask) > 0:
            r, g, b = cv2.mean(rgb, mask=mask)[:3]
            samples.append ([timestamp / 1000, r, g, b])
            text = f"R: {r:.1f}  G: {g:.1f}  B: {b:.1f}"
        else:
            text = "No face"

        cv2.putText(
            frame,
            text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        elapsed_seconds = timestamp / 1000
        timer_text = f"{elapsed_seconds:05.1f} s"

        cv2.rectangle(frame, (10, 45), (220, 100), (0, 0, 0), -1)

        cv2.putText(
            frame,
            timer_text,
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            2,
        )
        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

output_path = Path(__file__).with_name("rgb_adaptive.csv")

with output_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Time (s)", "R", "G", "B"])
    writer.writerows(samples)

print(f"Збережено {len(samples)} зразків у файл {output_path}")