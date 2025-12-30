import os
import cv2
import time
import pandas as pd
from collections import deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


USE_FACE_BLUR = False
USE_POSE = False

try:
    from mediapipe.python.solutions import face_detection as mp_face
    face_detector = mp_face.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )
    USE_FACE_BLUR = True
    print(" MediaPipe Face loaded")
except Exception:
    face_detector = None
    print("MediaPipe Face not available")

try:
    from mediapipe.python.solutions import pose as mp_pose
    pose_estimator = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    USE_POSE = True
    print(" MediaPipe Pose loaded")
except Exception:
    pose_estimator = None
    print(" MediaPipe Pose not available")


VIDEO_PATH = r"C:\Users\rachana sharma\exam-hall-anomaly\data\finalised_output\processed_output.mp4"
OUTPUT_DIR = r"C:\Users\rachana sharma\exam-hall-anomaly\data\week3_pose_update"

CONF_DET = 0.4
CONF_TRACK = 0.45
UNAUTH_OBJECTS = {"cell phone", "book", "laptop"}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

yolo = YOLO("yolov8n.pt")

tracker = DeepSort(
    max_age=300,
    n_init=6,
    max_iou_distance=0.95,
    embedder="mobilenet",
    half=False,
    bgr=True
)

detect_log = []
track_log = []
fps_log = []
pose_log = []

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError("❌ Cannot open video")

fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    os.path.join(OUTPUT_DIR, "processed_output_week3.mp4"),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps_src,
    (W, H)
)


fps_window = deque(maxlen=30)
frame_id = 0
print("▶️ Processing started (Week-3)...")

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    annotated = frame.copy()
    results = yolo(frame, conf=CONF_DET)[0]
    detections = []

    # -------- YOLO DETECTION --------
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
            bw, bh = x2 - x1, y2 - y1
            if bw * bh > 1500:
                detections.append(([x1, y1, bw, bh], conf, "person"))

        if label in UNAUTH_OBJECTS:
            cv2.putText(
                annotated,
                f"Unauthorized: {label}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 2
            )

    # -------- DEEPSORT --------
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if track.time_since_update > 2:
            continue

        pid = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated, f"ID {pid}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )

        track_log.append({
            "frame": frame_id,
            "track_id": pid,
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2
        })

        # -------- POSE ESTIMATION (Week-3) --------
        if USE_POSE:
            person_roi = frame[y1:y2, x1:x2]
            if person_roi.size > 0:
                rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                pose_results = pose_estimator.process(rgb_roi)

                if pose_results.pose_landmarks:
                    for idx, lm in enumerate(pose_results.pose_landmarks.landmark):
                        px = int(lm.x * (x2 - x1)) + x1
                        py = int(lm.y * (y2 - y1)) + y1

                        pose_log.append({
                            "frame": frame_id,
                            "person_id": pid,
                            "landmark_id": idx,
                            "x": px,
                            "y": py,
                            "visibility": lm.visibility
                        })

                        cv2.circle(annotated, (px, py), 2, (255, 0, 0), -1)

    # -------- FACE BLUR --------
    if USE_FACE_BLUR:
        faces = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if faces and faces.detections:
            for det in faces.detections:
                bb = det.location_data.relative_bounding_box
                fx1 = max(int(bb.xmin * W), 0)
                fy1 = max(int(bb.ymin * H), 0)
                fx2 = min(int((bb.xmin + bb.width) * W), W - 1)
                fy2 = min(int((bb.ymin + bb.height) * H), H - 1)

                roi = annotated[fy1:fy2, fx1:fx2]
                if roi.size > 0:
                    annotated[fy1:fy2, fx1:fx2] = cv2.GaussianBlur(roi, (41, 41), 25)

    # -------- FPS --------
    fps = 1 / (time.time() - start_time)
    fps_window.append(fps)
    avg_fps = sum(fps_window) / len(fps_window)

    fps_log.append({"frame": frame_id, "fps": round(avg_fps, 2)})

    cv2.putText(
        annotated, f"FPS: {avg_fps:.2f}",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 0, 255), 2
    )

    out.write(annotated)
    frame_id += 1

# ==============================
# SAVE CSVs
# ==============================
pd.DataFrame(detect_log).to_csv(os.path.join(OUTPUT_DIR, "detect_log.csv"), index=False)
pd.DataFrame(track_log).to_csv(os.path.join(OUTPUT_DIR, "track_log.csv"), index=False)
pd.DataFrame(fps_log).to_csv(os.path.join(OUTPUT_DIR, "fps_log.csv"), index=False)
pd.DataFrame(pose_log).to_csv(os.path.join(OUTPUT_DIR, "pose_keypoints.csv"), index=False)

print(" Pose estimation + CSV exported successfully!")
