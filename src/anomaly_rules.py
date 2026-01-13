import os
import cv2
import time
import math
import pandas as pd
import warnings
from collections import defaultdict, deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from mediapipe.python.solutions import pose as mp_pose

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# ==============================
# CONFIG
# ==============================
VIDEO_PATH = r"C:\Users\rachana sharma\exam-hall-anomaly\data\videos\exam scene1.mp4"
OUTPUT_DIR = r"C:\Users\rachana sharma\exam-hall-anomaly\tests\Test3"
EVIDENCE_DIR = os.path.join(OUTPUT_DIR, "evidence")
CLIPS_DIR = os.path.join(OUTPUT_DIR, "anomaly_clips")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)

FPS_ASSUMED = 30
HEAD_TURN_THRESHOLD = 10
HAND_MOVE_THRESHOLD = 25
PROXIMITY_THRESHOLD = 120
PROXIMITY_FRAMES = 90
HAND_NEAR_FACE_THRESHOLD = 0.1
LOOK_AWAY_THRESHOLD = 20
HANDS_RAISED_THRESHOLD = 0.2
MOVEMENT_THRESHOLD = 80
HEIGHT_RATIO_THRESHOLD = 0.4

# Temporal filtering
MIN_ANOMALY_DURATION = 10
ANOMALY_COOLDOWN = 30

# Face blur
BLUR_FACES = True
BLUR_KERNEL_SIZE = 99
BLUR_SIGMA = 30

# Clip extraction
CLIP_BEFORE_SECONDS = 2
CLIP_AFTER_SECONDS = 2

# ==============================
# MODELS
# ==============================
pose_estimator = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

yolo = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=300)

# ==============================
# STORAGE
# ==============================
pose_angles = defaultdict(lambda: deque(maxlen=FPS_ASSUMED * 60))
hand_moves = defaultdict(int)
head_turns = defaultdict(int)
proximity_counter = defaultdict(int)
movement_history = defaultdict(lambda: deque(maxlen=FPS_ASSUMED*2))
height_history = defaultdict(lambda: deque(maxlen=FPS_ASSUMED*2))
y_history = defaultdict(lambda: deque(maxlen=FPS_ASSUMED*2))
look_away_counter = defaultdict(int)
hands_raised_counter = defaultdict(int)

active_anomalies = defaultdict(lambda: defaultdict(int))
last_logged = defaultdict(lambda: defaultdict(int))

anomaly_log = []

# ==============================
# HELPER FUNCTIONS
# ==============================
def angle(p1, p2):
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def should_log_anomaly(pid, anomaly_type, frame_id):
    if frame_id - last_logged[pid][anomaly_type] < ANOMALY_COOLDOWN:
        return False
    if active_anomalies[pid][anomaly_type] >= MIN_ANOMALY_DURATION:
        last_logged[pid][anomaly_type] = frame_id
        return True
    return False

def blur_face(frame, x1, y1, x2, y2):
    face_height = int((y2 - y1) * 0.4)
    face_y1 = max(0, y1)
    face_y2 = min(frame.shape[0], y1 + face_height)
    x1 = max(0, x1)
    x2 = min(frame.shape[1], x2)
    
    face_roi = frame[face_y1:face_y2, x1:x2]
    if face_roi.size > 0:
        blurred = cv2.GaussianBlur(face_roi, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), BLUR_SIGMA)
        frame[face_y1:face_y2, x1:x2] = blurred
    return frame

def extract_clip(video_path, start_frame, end_frame, output_path, fps):
    """Extract video clip for anomaly evidence"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    
    for _ in range(end_frame - start_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    cap.release()
    out.release()
    return True

# ==============================
# VIDEO SETUP
# ==============================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError("Cannot open video")

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(
    os.path.join(OUTPUT_DIR, "processed_output_week4.mp4"),
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS_ASSUMED,
    (W, H)
)

frame_id = 0
start_time = time.time()
print("▶️  Exam Anomaly Detection Started")
print(f"📹 Video: {os.path.basename(VIDEO_PATH)}")
print(f"📊 Total frames: {total_frames}")
print(f"⏱️  Duration: {total_frames/FPS_ASSUMED:.1f}s")
print("-" * 60)

# ==============================
# MAIN LOOP
# ==============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    annotated = frame.copy()
    results = yolo(frame, conf=0.4, verbose=False)[0]
    detections = []

    for box in results.boxes:
        if int(box.cls[0]) == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], 0.9, "person"))

    tracks = tracker.update_tracks(detections, frame=frame)
    centers = {}

    for track in tracks:
        if not track.is_confirmed():
            continue
        pid = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        center = ((x1 + x2)//2, (y1 + y2)//2)
        centers[pid] = center
        movement_history[pid].append(center)
        height_history[pid].append(y2 - y1)
        y_history[pid].append(center[1])

    invigilators = set()
    for pid in movement_history:
        if len(movement_history[pid]) < movement_history[pid].maxlen:
            continue
        x_disp = max([c[0] for c in movement_history[pid]]) - min([c[0] for c in movement_history[pid]])
        y_disp = max([c[1] for c in movement_history[pid]]) - min([c[1] for c in movement_history[pid]])
        avg_height = sum(height_history[pid])/len(height_history[pid])
        if (x_disp > MOVEMENT_THRESHOLD or y_disp > MOVEMENT_THRESHOLD) and avg_height > H*HEIGHT_RATIO_THRESHOLD:
            invigilators.add(pid)

    current_frame_anomalies = defaultdict(set)

    for track in tracks:
        if not track.is_confirmed():
            continue
        pid = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        
        if BLUR_FACES:
            annotated = blur_face(annotated, x1, y1, x2, y2)
        
        if pid in invigilators:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(annotated, f"Staff {pid}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            continue

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        pose = pose_estimator.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        anomaly_types = []

        if pose.pose_landmarks:
            lm = pose.pose_landmarks.landmark

            l_sh, r_sh = (lm[11].x, lm[11].y), (lm[12].x, lm[12].y)
            current_angle = angle(l_sh, r_sh)
            pose_angles[pid].append(current_angle)
            
            if len(pose_angles[pid]) > 2 and abs(pose_angles[pid][-1]-pose_angles[pid][-2])>15:
                head_turns[pid] += 1
            
            if head_turns[pid] > HEAD_TURN_THRESHOLD:
                current_frame_anomalies[pid].add("Excessive Head Turn")

            if abs(lm[15].x - lm[16].x) > 0.15:
                hand_moves[pid] += 1
            
            if hand_moves[pid] > HAND_MOVE_THRESHOLD:
                current_frame_anomalies[pid].add("Excessive Hand Movement")

            shoulder_angle = angle(l_sh, r_sh)
            if abs(shoulder_angle) > LOOK_AWAY_THRESHOLD:
                look_away_counter[pid] += 1
            
            if look_away_counter[pid] > 10:
                current_frame_anomalies[pid].add("Looking Away")

            if lm[15].y < lm[11].y - HANDS_RAISED_THRESHOLD and lm[16].y < lm[12].y - HANDS_RAISED_THRESHOLD:
                hands_raised_counter[pid] += 1
            
            if hands_raised_counter[pid] > 10:
                current_frame_anomalies[pid].add("Hands Raised")

        for anomaly_type in current_frame_anomalies[pid]:
            active_anomalies[pid][anomaly_type] += 1
            
            if should_log_anomaly(pid, anomaly_type, frame_id):
                anomaly_types.append(anomaly_type)
                anomaly_log.append({
                    "frame": frame_id,
                    "person_id": pid,
                    "type": anomaly_type,
                    "timestamp": f"{frame_id/FPS_ASSUMED:.2f}s",
                    "clip_start": max(0, frame_id - CLIP_BEFORE_SECONDS*FPS_ASSUMED),
                    "clip_end": min(total_frames-1, frame_id + CLIP_AFTER_SECONDS*FPS_ASSUMED)
                })
                cv2.imwrite(os.path.join(EVIDENCE_DIR, f"{anomaly_type.replace(' ','')}_{pid}_f{frame_id}.jpg"), annotated)

        for anomaly_type in list(active_anomalies[pid].keys()):
            if anomaly_type not in current_frame_anomalies[pid]:
                active_anomalies[pid][anomaly_type] = 0

        box_color = (0, 255, 0) if not anomaly_types else (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(annotated, f"ID {pid}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        for i, a_type in enumerate(anomaly_types):
            cv2.putText(annotated, a_type, (x1, y2+20+20*i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    for id1, c1 in centers.items():
        for id2, c2 in centers.items():
            if id1 >= id2 or id1 in invigilators or id2 in invigilators:
                continue
            if distance(c1, c2) < PROXIMITY_THRESHOLD:
                proximity_counter[(id1,id2)] += 1
                if proximity_counter[(id1,id2)] == PROXIMITY_FRAMES:
                    anomaly_log.append({
                        "frame": frame_id,
                        "person_id": f"{id1}-{id2}",
                        "type": "Suspicious Proximity",
                        "timestamp": f"{frame_id/FPS_ASSUMED:.2f}s",
                        "clip_start": max(0, frame_id - CLIP_BEFORE_SECONDS*FPS_ASSUMED),
                        "clip_end": min(total_frames-1, frame_id + CLIP_AFTER_SECONDS*FPS_ASSUMED)
                    })
                    cv2.imwrite(os.path.join(EVIDENCE_DIR, f"proximity_{id1}_{id2}_f{frame_id}.jpg"), annotated)

    elapsed_time = time.time() - start_time
    fps_display = frame_id/elapsed_time if elapsed_time>0 else 0
    cv2.putText(annotated, f"FPS: {fps_display:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    cv2.putText(annotated, f"Time: {frame_id/FPS_ASSUMED:.2f}s", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    out.write(annotated)
    frame_id += 1
    
    if frame_id % 100 == 0:
        progress = (frame_id / total_frames) * 100
        print(f"Progress: {progress:.1f}% | Frame {frame_id}/{total_frames} | Anomalies: {len(anomaly_log)}")

cap.release()
out.release()

# ==============================
# EXTRACT CLIPS
# ==============================
print("\n" + "="*60)
print("📹 Extracting anomaly clips...")
print("="*60)

processed_video = os.path.join(OUTPUT_DIR, "processed_output_week4.mp4")
anomaly_df = pd.DataFrame(anomaly_log)

for idx, row in anomaly_df.iterrows():
    clip_filename = f"{row['type'].replace(' ','')}_{row['person_id']}_f{row['frame']}.mp4"
    clip_path = os.path.join(CLIPS_DIR, clip_filename)
    
    extract_clip(processed_video, int(row['clip_start']), int(row['clip_end']), clip_path, FPS_ASSUMED)
    
    if (idx + 1) % 5 == 0 or idx == len(anomaly_df) - 1:
        print(f"   Extracted {idx + 1}/{len(anomaly_df)} clips")

# ==============================
# SAVE RESULTS
# ==============================
anomaly_df.to_csv(os.path.join(OUTPUT_DIR, "anomaly_log.csv"), index=False)

print("\n" + "="*60)
print("✅ PROCESSING COMPLETE")
print("="*60)
print(f"📊 Statistics:")
print(f"   • Total frames processed: {frame_id}")
print(f"   • Processing time: {time.time() - start_time:.1f}s")
print(f"   • Average FPS: {frame_id/(time.time()-start_time):.2f}")
print(f"   • Anomalies detected: {len(anomaly_log)}")
print(f"\n📁 Output files:")
print(f"   • Video: {os.path.join(OUTPUT_DIR, 'processed_output_week4.mp4')}")
print(f"   • Log: {os.path.join(OUTPUT_DIR, 'anomaly_log.csv')}")
print(f"   • Screenshots: {EVIDENCE_DIR}/ ({len(os.listdir(EVIDENCE_DIR))} files)")
print(f"   • Clips: {CLIPS_DIR}/ ({len(os.listdir(CLIPS_DIR))} files)")
print("="*60)