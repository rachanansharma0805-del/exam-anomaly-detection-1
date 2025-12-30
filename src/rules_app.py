import os
import cv2
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

VIDEO_PATH = r"C:\Users\rachana sharma\exam-hall-anomaly\data\week3_pose_update\processed_output_week3.mp4"
OUTPUT_DIR = r"C:\Users\rachana sharma\exam-hall-anomaly\data\week4_anomaly_detection"
EVIDENCE_DIR = os.path.join(OUTPUT_DIR, "evidence")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)

FPS_ASSUMED = 30

HEAD_TURN_THRESHOLD = 10      # per minute
HAND_MOVE_THRESHOLD = 25     # per minute
PROXIMITY_THRESHOLD = 120    # pixels
PROXIMITY_FRAMES = 90        # ~3 sec

from mediapipe.python.solutions import pose as mp_pose
pose_estimator = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

yolo = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=300)

pose_angles = defaultdict(lambda: deque(maxlen=FPS_ASSUMED * 60))
hand_moves = defaultdict(int)
head_turns = defaultdict(int)
proximity_counter = defaultdict(int)

anomaly_log = []

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError("Cannot open video")

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    os.path.join(OUTPUT_DIR, "processed_output_week4.mp4"),
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS_ASSUMED,
    (W, H)
)

frame_id = 0
print("▶️ Week-4 processing started (with video output)")

def angle(p1, p2):
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    annotated = frame.copy()

    results = yolo(frame, conf=0.4)[0]
    detections = []

    for box in results.boxes:
        if int(box.cls[0]) == 0:  # person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], 0.9, "person"))

    tracks = tracker.update_tracks(detections, frame=frame)
    centers = {}

    for track in tracks:
        if not track.is_confirmed():
            continue

        pid = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        centers[pid] = ((x1 + x2) // 2, (y1 + y2) // 2)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated, f"ID {pid}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 0), 2
        )

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        pose = pose_estimator.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        if not pose.pose_landmarks:
            continue

        lm = pose.pose_landmarks.landmark

        # ---- Head Turn ----
        l_sh = (lm[11].x, lm[11].y)
        r_sh = (lm[12].x, lm[12].y)
        current_angle = angle(l_sh, r_sh)

        pose_angles[pid].append(current_angle)
        if len(pose_angles[pid]) > 2:
            if abs(pose_angles[pid][-1] - pose_angles[pid][-2]) > 15:
                head_turns[pid] += 1

        # ---- Hand Movement ----
        if abs(lm[15].x - lm[16].x) > 0.15:
            hand_moves[pid] += 1

        # ---- Rule Engine ----
        if head_turns[pid] > HEAD_TURN_THRESHOLD:
            anomaly_log.append({
                "frame": frame_id,
                "person_id": pid,
                "type": "Excessive Head Turn"
            })
            cv2.putText(
                annotated, "HEAD TURN ALERT",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255), 2
            )
            cv2.imwrite(
                os.path.join(EVIDENCE_DIR, f"headturn_{pid}_f{frame_id}.jpg"),
                annotated
            )

        if hand_moves[pid] > HAND_MOVE_THRESHOLD:
            anomaly_log.append({
                "frame": frame_id,
                "person_id": pid,
                "type": "Excessive Hand Movement"
            })

    # ---- Proximity ----
    for id1, c1 in centers.items():
        for id2, c2 in centers.items():
            if id1 >= id2:
                continue
            if math.dist(c1, c2) < PROXIMITY_THRESHOLD:
                proximity_counter[(id1, id2)] += 1
                if proximity_counter[(id1, id2)] > PROXIMITY_FRAMES:
                    anomaly_log.append({
                        "frame": frame_id,
                        "person_id": f"{id1}-{id2}",
                        "type": "Suspicious Proximity"
                    })
                    cv2.imwrite(
                        os.path.join(EVIDENCE_DIR, f"proximity_{id1}_{id2}_f{frame_id}.jpg"),
                        annotated
                    )

    out.write(annotated)
    frame_id += 1

cap.release()
out.release()

pd.DataFrame(anomaly_log).to_csv(
    os.path.join(OUTPUT_DIR, "anomaly_log.csv"),
    index=False
)

print("Integrated Anomalies successfully processed and saved.")
