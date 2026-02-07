# anomaly_rules.property
import os
import cv2
import time
import math
import pandas as pd
from collections import defaultdict, deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from mediapipe.python.solutions import pose as mp_pose

# ==============================
# CONFIG
# ==============================
VIDEO_PATH = r"C:\Users\rachana sharma\exam-hall-anomaly\data\videos\megs.mp4"
OUTPUT_DIR = r"C:\Users\rachana sharma\exam-hall-anomaly\outputs\megs_output"
EVIDENCE_DIR = os.path.join(OUTPUT_DIR, "evidence")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)

FPS_ASSUMED = 30
SECONDS_PER_WINDOW = 30
FRAMES_PER_WINDOW = FPS_ASSUMED * SECONDS_PER_WINDOW

# Thresholds
HEAD_TURN_THRESHOLD = 15
HAND_MOVE_THRESHOLD = 30
PROXIMITY_THRESHOLD = 150
PROXIMITY_FRAMES = 60
HAND_NEAR_FACE_THRESHOLD = 0.08
LOOK_AWAY_THRESHOLD = 25
HANDS_RAISED_THRESHOLD = 0.25
MOVEMENT_THRESHOLD = 100
HEIGHT_RATIO_THRESHOLD = 0.3

# ==============================
# MODELS
# ==============================
pose_estimator = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

yolo = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30, n_init=3)

# ==============================
# STORAGE
# ==============================
pose_angles = defaultdict(lambda: deque(maxlen=FPS_ASSUMED * 5))
head_turn_counts = defaultdict(lambda: deque(maxlen=FRAMES_PER_WINDOW))
hand_move_counts = defaultdict(lambda: deque(maxlen=FRAMES_PER_WINDOW))
speaking_counts = defaultdict(lambda: deque(maxlen=FRAMES_PER_WINDOW))
look_away_counts = defaultdict(lambda: deque(maxlen=FRAMES_PER_WINDOW))
hands_raised_counts = defaultdict(lambda: deque(maxlen=FRAMES_PER_WINDOW))

proximity_counter = defaultdict(int)
movement_history = defaultdict(lambda: deque(maxlen=FPS_ASSUMED * 2))
height_history = defaultdict(lambda: deque(maxlen=FPS_ASSUMED * 2))

current_counts = defaultdict(lambda: {
    "head_turns": 0,
    "hand_moves": 0,
    "speaking": 0,
    "look_away": 0,
    "hands_raised": 0
})

anomaly_log = []
last_evidence_frame = defaultdict(int)

# ==============================
# VIDEO SETUP - TWO SEPARATE VIDEOS
# ==============================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError("Cannot open video")

actual_fps = cap.get(cv2.CAP_PROP_FPS)
FPS = actual_fps if actual_fps > 0 else FPS_ASSUMED

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video: {W}x{H} @ {FPS} FPS")

# 1. VIDEO WITH ANOMALIES (Red boxes)
anomaly_video_path = os.path.join(OUTPUT_DIR, "anomalies_video.mp4")
anomaly_out = cv2.VideoWriter(
    anomaly_video_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (W, H)
)

# 2. VIDEO WITH NORMAL BEHAVIOR (Green boxes only)
normal_video_path = os.path.join(OUTPUT_DIR, "normal_behavior_video.mp4")
normal_out = cv2.VideoWriter(
    normal_video_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (W, H)
)

# 3. FULL VIDEO with everything (optional)
full_video_path = os.path.join(OUTPUT_DIR, "full_annotated_video.mp4")
full_out = cv2.VideoWriter(
    full_video_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (W, H)
)

frame_id = 0
start_time = time.time()
print("▶️ Processing - Creating separate anomaly/normal videos")

# ==============================
# HELPER FUNCTIONS
# ==============================
def angle(p1, p2):
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

def timestamp(frame_id):
    seconds = frame_id / FPS
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def reset_window_counts():
    current_counts.clear()

# ==============================
# MAIN LOOP
# ==============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Reset window counts periodically
    if frame_id % FRAMES_PER_WINDOW == 0:
        reset_window_counts()

    annotated = frame.copy()
    has_anomaly = False
    
    # ----- YOLO DETECTION -----
    results = yolo(frame, conf=0.5, classes=[0])[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

    # ----- TRACKING -----
    tracks = tracker.update_tracks(detections, frame=frame)
    centers = {}
    current_frame_anomalies = []

    # Track movement for invigilator detection
    for track in tracks:
        if not track.is_confirmed():
            continue
        pid = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        centers[pid] = center
        movement_history[pid].append(center)
        height_history[pid].append(y2 - y1)

    # Identify invigilators
    invigilators = set()
    for pid in movement_history:
        if len(movement_history[pid]) < 10:
            continue
        x_disp = max([c[0] for c in movement_history[pid]]) - min([c[0] for c in movement_history[pid]])
        y_disp = max([c[1] for c in movement_history[pid]]) - min([c[1] for c in movement_history[pid]])
        avg_height = sum(height_history[pid]) / len(height_history[pid])
        
        if (x_disp > MOVEMENT_THRESHOLD or y_disp > MOVEMENT_THRESHOLD) and avg_height > H * HEIGHT_RATIO_THRESHOLD:
            invigilators.add(pid)
            # Mark invigilator with orange
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.putText(annotated, f"Invigilator {pid}", (x1, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

    # ----- ANOMALY DETECTION -----
    students_processed = 0
    frame_has_any_anomaly = False
    
    for track in tracks:
        if not track.is_confirmed():
            continue
            
        pid = track.track_id
        if pid in invigilators:
            continue
            
        students_processed += 1
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        pose = pose_estimator.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        person_anomalies = []

        if pose.pose_landmarks:
            lm = pose.pose_landmarks.landmark

            # Head Turn
            l_sh = (lm[11].x, lm[11].y)
            r_sh = (lm[12].x, lm[12].y)
            current_angle = angle(l_sh, r_sh)
            pose_angles[pid].append(current_angle)
            
            if len(pose_angles[pid]) >= 2:
                angle_diff = abs(pose_angles[pid][-1] - pose_angles[pid][-2])
                if angle_diff > 15:
                    head_turn_counts[pid].append(1)
                    current_counts[pid]["head_turns"] += 1
                else:
                    head_turn_counts[pid].append(0)
            
            if sum(head_turn_counts[pid]) > HEAD_TURN_THRESHOLD:
                person_anomalies.append("Excessive Head Turn")

            # Hand Movement
            hand_distance = abs(lm[15].x - lm[16].x)
            if hand_distance > 0.15:
                hand_move_counts[pid].append(1)
                current_counts[pid]["hand_moves"] += 1
            else:
                hand_move_counts[pid].append(0)
            
            if sum(hand_move_counts[pid]) > HAND_MOVE_THRESHOLD:
                person_anomalies.append("Excessive Hand Movement")

            # Speaking (hand near face)
            left_hand = (lm[15].x, lm[15].y)
            right_hand = (lm[16].x, lm[16].y)
            nose = (lm[0].x, lm[0].y)
            
            if distance(left_hand, nose) < HAND_NEAR_FACE_THRESHOLD or distance(right_hand, nose) < HAND_NEAR_FACE_THRESHOLD:
                speaking_counts[pid].append(1)
                current_counts[pid]["speaking"] += 1
            else:
                speaking_counts[pid].append(0)
            
            if sum(speaking_counts[pid]) > 15:
                person_anomalies.append("Speaking")

            # Looking Away
            shoulder_angle = angle(l_sh, r_sh)
            if abs(shoulder_angle) > LOOK_AWAY_THRESHOLD:
                look_away_counts[pid].append(1)
                current_counts[pid]["look_away"] += 1
            else:
                look_away_counts[pid].append(0)
            
            if sum(look_away_counts[pid]) > 15:
                person_anomalies.append("Looking Away")

            # Hands Raised
            if lm[15].y < lm[11].y - HANDS_RAISED_THRESHOLD and lm[16].y < lm[12].y - HANDS_RAISED_THRESHOLD:
                hands_raised_counts[pid].append(1)
                current_counts[pid]["hands_raised"] += 1
            else:
                hands_raised_counts[pid].append(0)
            
            if sum(hands_raised_counts[pid]) > 15:
                person_anomalies.append("Hands Raised")

        # Determine box color
        if person_anomalies:
            box_color = (0, 0, 255)  # RED
            frame_has_any_anomaly = True
            has_anomaly = True
        else:
            box_color = (0, 255, 0)  # GREEN
            
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
        
        # Display ID
        cv2.putText(annotated, f"ID {pid}", (x1, y1 - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # Add anomaly labels (only for anomalies)
        if person_anomalies:
            for i, a_type in enumerate(person_anomalies):
                cv2.putText(annotated, a_type, (x1, y2 + 20 + 20 * i), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                current_frame_anomalies.append({
                    "frame": frame_id,
                    "person_id": pid,
                    "type": a_type,
                    "timestamp": timestamp(frame_id),
                    "timestamp_seconds": frame_id / FPS
                })

    # ----- PROXIMITY DETECTION -----
    for id1, c1 in centers.items():
        for id2, c2 in centers.items():
            if id1 >= id2 or id1 in invigilators or id2 in invigilators:
                continue
            
            if distance(c1, c2) < PROXIMITY_THRESHOLD:
                proximity_counter[(id1, id2)] += 1
                if proximity_counter[(id1, id2)] > PROXIMITY_FRAMES:
                    # Draw line and label
                    cv2.line(annotated, c1, c2, (0, 0, 255), 2)
                    midpoint = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
                    cv2.putText(annotated, "Close Proximity", midpoint, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    current_frame_anomalies.append({
                        "frame": frame_id,
                        "person_id": f"{id1}-{id2}",
                        "type": "Suspicious Proximity",
                        "timestamp": timestamp(frame_id),
                        "timestamp_seconds": frame_id / FPS
                    })
                    frame_has_any_anomaly = True
                    has_anomaly = True
            else:
                proximity_counter[(id1, id2)] = 0

    # ----- SAVE TO APPROPRIATE VIDEOS -----
    # Always save to full video
    full_out.write(annotated)
    
    # Save to anomaly video if ANY anomaly detected in frame
    if frame_has_any_anomaly:
        anomaly_out.write(annotated)
    else:
        # Save to normal video if NO anomalies
        normal_out.write(annotated)

    # ----- LOG ANOMALIES -----
    if current_frame_anomalies:
        anomaly_log.extend(current_frame_anomalies)

    # ----- DISPLAY STATS -----
    elapsed_time = time.time() - start_time
    fps_display = frame_id / elapsed_time if elapsed_time > 0 else 0
    
    # Status indicator in top-left
    status_color = (0, 0, 255) if frame_has_any_anomaly else (0, 255, 0)
    status_text = "ANOMALY" if frame_has_any_anomaly else "NORMAL"
    
    cv2.rectangle(annotated, (5, 5), (150, 130), (30, 30, 30), -1)
    cv2.putText(annotated, f"STATUS: {status_text}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(annotated, f"FPS: {fps_display:.1f}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(annotated, f"Time: {timestamp(frame_id)}", (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(annotated, f"Frame: {frame_id}", (10, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Save evidence frame if anomaly (limited to 1 per second)
    if frame_has_any_anomaly and frame_id % FPS == 0:
        evidence_path = os.path.join(EVIDENCE_DIR, f"anomaly_frame_{frame_id:06d}.jpg")
        cv2.imwrite(evidence_path, annotated)

    frame_id += 1
    
    # Progress indicator
    if frame_id % 100 == 0:
        print(f"Frame {frame_id} - Status: {status_text}")

# ==============================
# CLEANUP
# ==============================
cap.release()
full_out.release()
anomaly_out.release()
normal_out.release()
pose_estimator.close()

# Save logs
if anomaly_log:
    df_anomalies = pd.DataFrame(anomaly_log)
    csv_path = os.path.join(OUTPUT_DIR, "anomaly_log.csv")
    df_anomalies.to_csv(csv_path, index=False)
    
    # Summary
    print(f"\n📊 ANOMALY SUMMARY:")
    print(f"Total anomalies detected: {len(df_anomalies)}")
    print("By type:")
    print(df_anomalies['type'].value_counts())
    print(f"\n✅ Log saved to: {csv_path}")
else:
    print("\n✅ No anomalies detected")

# Verify video files were created
print(f"\n📹 VIDEO OUTPUTS:")
videos = [
    ("Full annotated video", full_video_path),
    ("Anomalies only", anomaly_video_path),
    ("Normal behavior only", normal_video_path)
]

for video_name, video_path in videos:
    if os.path.exists(video_path):
        file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
        print(f"✅ {video_name}: {video_path}")
        print(f"   Size: {file_size:.2f} MB")
    else:
        print(f"❌ {video_name}: NOT CREATED")

print(f"\n🎯 Processing completed in {time.time() - start_time:.1f} seconds")
print(f"Total frames processed: {frame_id}")