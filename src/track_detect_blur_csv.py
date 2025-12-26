import os
import cv2
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ---------------------------------
# CONFIGURATION (CHANGE ONLY THIS)
# ---------------------------------
YOLO_MODEL = "yolov8n.pt"
CONF_THRESH = 0.5
CAMERA_IDS = {0: "cam1", 1: "cam2"}  # Map for multi-camera

INPUT_PATH = "data/videos"   # folder or single video
OUTPUT_DIR = "data/week2_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------
# LOAD MODELS
# ---------------------------------
model = YOLO(YOLO_MODEL)
tracker = DeepSort(max_age=30, n_init=3, embedder="mobilenet")

# ---------------------------------
# GLOBAL CSV TRACKERS
# ---------------------------------
det_log = []
track_log = []

# ---------------------------------
# FACE BLUR FUNCTION
# ---------------------------------
def blur_face(frame, x1, y1, x2, y2):
    """Blurs the upper portion of a person bounding box (face area)."""
    face_height = max(int((y2 - y1) * 0.5), 20)
    face_region = frame[y1:y1 + face_height, x1:x2]
    
    if face_region.size > 0:
        ksize = ((face_height // 3) | 1, (x2 - x1) // 3 | 1)
        blurred_face = cv2.GaussianBlur(face_region, ksize, 0)
        frame[y1:y1 + face_height, x1:x2] = blurred_face

# ---------------------------------
# DETECTION + TRACKING
# ---------------------------------
def detect_and_track(frame, frame_count, timestamp, video_name, camera_id):
    """Detects people using YOLO, tracks with DeepSORT, blurs faces, logs data."""
    results = model(frame)[0]
    detections = []

    if results.boxes is not None:
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == 0 and conf > CONF_THRESH:  # person
                blur_face(frame, x1, y1, x2, y2)
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

                # Log detection
                det_log.append({
                    'video_name': video_name,
                    'camera_id': camera_id,
                    'frame': frame_count,
                    'timestamp_rel': timestamp,
                    'timestamp_abs': pd.Timestamp.now(),
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'conf': conf
                })

    # Update tracker with error handling
    try:
        tracks = tracker.update_tracks(detections, frame=frame)
    except Exception as e:
        print(f"Tracker error: {e}")
        tracks = []

    for track in tracks:
        if not track.is_confirmed():
            continue
        l, t, r, b = map(int, track.to_ltrb())
        track_id = track.track_id

        # Draw track box
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (l, t - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Log track
        track_log.append({
            'video_name': video_name,
            'camera_id': camera_id,
            'frame': frame_count,
            'timestamp_rel': timestamp,
            'timestamp_abs': pd.Timestamp.now(),
            'track_id': track_id,
            'x1': l, 'y1': t, 'x2': r, 'y2': b,
            'bbox_area': (r - l) * (b - t)
        })

    return frame

# ---------------------------------
# VIDEO PROCESSING
# ---------------------------------
def process_video(video_path, camera_id=0):
    """Process single video with tracking and logging."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_name = os.path.basename(video_path)
    cam_name = CAMERA_IDS.get(camera_id, f"cam{camera_id}")

    # Output path
    out_name = f"{cam_name}_blurred_{os.path.splitext(video_name)[0]}.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    out_video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_count = 0
    print(f"🎥 Processing {video_name} (Camera {camera_id})...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        frame = detect_and_track(frame, frame_count, timestamp, video_name, camera_id)
        out_video.write(frame)
        
        # Live preview every 30 frames
        if frame_count % 30 == 0:
            cv2.imshow(f"YOLOv8 + DeepSORT - {cam_name}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    print(f"✅ Video saved to {out_path}")

# ---------------------------------
# BATCH VIDEO PROCESSING
# ---------------------------------
def process_video_folder(folder_path):
    """Process all videos in folder with multi-camera support."""
    video_extensions = (".mp4", ".avi", ".mov", ".mkv")
    video_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(video_extensions)]
    
    if not video_files:
        print("No video files found")
        return

    print(f"🎥 Processing {len(video_files)} videos...")
    
    # Process videos with camera ID assignment
    for idx, video_name in enumerate(video_files):
        video_path = os.path.join(folder_path, video_name)
        camera_id = min(idx, 1)  # 0 or 1 for dual-camera
        process_video(video_path, camera_id)

    # Save consolidated CSVs
    if det_log:
        pd.DataFrame(det_log).to_csv(os.path.join(OUTPUT_DIR, 'detections.csv'), index=False)
        print(f"✅ detections.csv saved ({len(det_log)} rows)")
    
    if track_log:
        tracks_df = pd.DataFrame(track_log)
        tracks_df.to_csv(os.path.join(OUTPUT_DIR, 'tracks.csv'), index=False)
        print(f" tracks.csv saved ({len(track_log)} rows)")
        print(" CSVs ready for trajectory analysis!")

# ---------------------------------
# MAIN EXECUTION (VIDEO ONLY)
# ---------------------------------
if __name__ == "__main__":
    if os.path.isdir(INPUT_PATH):
        process_video_folder(INPUT_PATH)
    elif INPUT_PATH.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        process_video(INPUT_PATH, camera_id=0)
    else:
        print("❌ Use video folder or single video only")
