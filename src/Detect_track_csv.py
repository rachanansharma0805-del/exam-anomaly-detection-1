import os
import cv2
import time
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ==============================
# OPTIONAL MEDIAPIPE (SAFE)
# ==============================
try:
    import mediapipe as mp
    mp_face = mp.solutions.face_detection
    face_detector = mp_face.FaceDetection(
        model_selection=1,          # full-range model
        min_detection_confidence=0.3
    )
    USE_FACE_BLUR = True
    print("✅ MediaPipe loaded (face blur enabled)")
except Exception as e:
    face_detector = None
    USE_FACE_BLUR = False
    print("MediaPipe not available (face blur disabled)", e)

# ==============================
# CONFIG
# ==============================
VIDEO_PATH = r"C:\Users\rachana sharma\exam-hall-anomaly\data\videos\megs.mp4"
OUTPUT_DIR = r"C:\Users\rachana sharma\exam-hall-anomaly\data\finalised_output"

CONF_DET = 0.4
CONF_TRACK = 0.45

UNAUTH_OBJECTS = {"cell phone", "book", "laptop"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# FORCE CPU (Windows safe)
# ==============================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ==============================
# MODELS
# ==============================
yolo = YOLO("yolov8n.pt")

tracker = DeepSort(
    max_age=300,
    n_init=6,
    max_iou_distance=0.95,
    embedder="mobilenet",
    half=False,
    bgr=True
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
    raise IOError(" Cannot open video")

fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    os.path.join(OUTPUT_DIR, "processed_output.mp4"),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps_video,
    (w, h)
)

frame_id = 0
prev_time = time.time()

print("Processing started...")

# ==============================
# MAIN LOOP
# ==============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    annotated = frame.copy()

    # -------- YOLO DETECTION --------
    results = yolo(frame, conf=CONF_DET)[0]
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
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2
        })

        if label == "person" and conf >= CONF_TRACK:
            bw = x2 - x1
            bh = y2 - y1
            if bw * bh > 1500:
                detections.append(([x1, y1, bw, bh], conf, "person"))

        if label in UNAUTH_OBJECTS:
            cv2.putText(
                annotated,
                f"Unauthorized: {label}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    # -------- DEEPSORT TRACKING --------
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if track.time_since_update > 2:
            continue

        tid = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"ID {tid}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        track_log.append({
            "frame": frame_id,
            "track_id": tid,
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2
        })

    # -------- FACE BLUR (PERSON-AWARE) --------
    if USE_FACE_BLUR and face_detector is not None:
        for track in tracks:
            if track.time_since_update > 2:
                continue

            x1, y1, x2, y2 = map(int, track.to_ltrb())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            person_roi = frame[y1:y2, x1:x2]
            if person_roi.size == 0:
                continue

            rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            faces = face_detector.process(rgb_roi)

            if faces and faces.detections:
                for det in faces.detections:
                    bb = det.location_data.relative_bounding_box

                    fx1 = int(bb.xmin * (x2 - x1))
                    fy1 = int(bb.ymin * (y2 - y1))
                    fx2 = int((bb.xmin + bb.width) * (x2 - x1))
                    fy2 = int((bb.ymin + bb.height) * (y2 - y1))

                    fx1 = max(x1 + fx1, x1)
                    fy1 = max(y1 + fy1, y1)
                    fx2 = min(x1 + fx2, x2)
                    fy2 = min(y1 + fy2, y2)

                    if fx2 <= fx1 or fy2 <= fy1:
                        continue

                    roi = annotated[fy1:fy2, fx1:fx2]
                    if roi.size > 0:
                        annotated[fy1:fy2, fx1:fx2] = cv2.GaussianBlur(
                            roi, (31, 31), 20
                        )

    # -------- FPS DISPLAY --------
    curr_time = time.time()
    fps_display = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(
        annotated,
        f"FPS: {int(fps_display)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    out.write(annotated)
    frame_id += 1

# ==============================
# CLEANUP
# ==============================
cap.release()
out.release()
cv2.destroyAllWindows()

pd.DataFrame(detect_log).to_csv(
    os.path.join(OUTPUT_DIR, "detect_log.csv"),
    index=False
)
pd.DataFrame(track_log).to_csv(
    os.path.join(OUTPUT_DIR, "track_log.csv"),
    index=False
)

print(" **Processing completed successfully**.")