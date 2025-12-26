import cv2
import os
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ==============================
# CONFIG
# ==============================
VIDEO_PATH = r"C:\Users\rachana sharma\exam-hall-anomaly\data\videos\exam scene1.mp4"
OUTPUT_DIR = r"C:\Users\rachana sharma\exam-hall-anomaly\data\meg_output"

CONF_THRESH = 0.4
UNAUTH_OBJECTS = {"cell phone", "book", "laptop"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# MODELS
# ==============================
yolo = YOLO("yolov8n.pt")

tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_iou_distance=0.7
)

# ==============================
# LOGS
# ==============================
detect_log = []
track_log = []

# ==============================
# VIDEO SETUP
# ==============================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError("❌ Cannot open video. Check path.")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    os.path.join(OUTPUT_DIR, "processed_milestone2.mp4"),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

frame_id = 0
print("✅ Processing started...")

# ==============================
# MAIN LOOP
# ==============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    annotated = frame.copy()

    # -------- YOLO DETECTION --------
    results = yolo(frame, conf=CONF_THRESH)[0]

    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = results.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detect_log.append({
            "frame": frame_id,
            "label": label,
            "confidence": conf,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2
        })

        # Track ONLY persons
        if label == "person":
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

        # Unauthorized objects
        if label in UNAUTH_OBJECTS:
            cv2.putText(
                annotated,
                f"Unauthorized: {label}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    # -------- DEEPSORT TRACKING --------
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"ID {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        track_log.append({
            "frame": frame_id,
            "track_id": track_id,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2
        })

    out.write(annotated)
    cv2.imshow("Milestone 2 - YOLOv8 + DeepSORT", annotated)

    frame_id += 1
    print(f"Frame {frame_id}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# SAVE LOGS
# ==============================
pd.DataFrame(detect_log).to_csv(
    os.path.join(OUTPUT_DIR, "detect_log.csv"), index=False
)
pd.DataFrame(track_log).to_csv(
    os.path.join(OUTPUT_DIR, "track_log.csv"), index=False
)

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Processing completed successfully.")