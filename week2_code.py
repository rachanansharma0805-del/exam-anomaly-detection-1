import os
import cv2
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ---------------------------------
# CONFIGURATION
# ---------------------------------
YOLO_MODEL = "yolov8n.pt"
CONF_THRESH = 0.5

INPUT_PATH = "data/videos"   # image, image folder, video, or video folder
OUTPUT_DIR = "data/final_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CAMERA_IDS = {0: "cam1", 1: "cam2"}

# ---------------------------------
# LOAD MODELS
# ---------------------------------
model = YOLO(YOLO_MODEL)
tracker = DeepSort(max_age=30, n_init=3, embedder="mobilenet")

# ---------------------------------
# GLOBAL LOGS
# ---------------------------------
det_log = []
track_log = []

# ---------------------------------
# FACE BLUR FUNCTION
# ---------------------------------
def blur_face(frame, x1, y1, x2, y2):
    face_height = max(int((y2 - y1) * 0.5), 20)
    face_region = frame[y1:y1 + face_height, x1:x2]

    if face_region.size > 0:
        kx = max((x2 - x1) // 3 | 1, 3)
        ky = max(face_height // 3 | 1, 3)
        blurred = cv2.GaussianBlur(face_region, (kx, ky), 0)
        frame[y1:y1 + face_height, x1:x2] = blurred

# ---------------------------------
# IMAGE PROCESSING
# ---------------------------------
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not read image: {image_path}")
        return

    image_name = os.path.basename(image_path)
    print(f"🖼️ Processing image: {image_name}")

    results = model(image)[0]

    if results.boxes is not None:
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == 0 and conf > CONF_THRESH:
                blur_face(image, x1, y1, x2, y2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                det_log.append({
                    "type": "image",
                    "file_name": image_name,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "confidence": conf,
                    "timestamp": pd.Timestamp.now()
                })

    out_path = os.path.join(OUTPUT_DIR, f"blurred_{image_name}")
    cv2.imwrite(out_path, image)
    print(f"✅ Saved: {out_path}")

# ---------------------------------
# IMAGE FOLDER
# ---------------------------------
def process_image_folder(folder_path):
    image_exts = (".jpg", ".jpeg", ".png", ".bmp")
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(image_exts)]

    if not images:
        print("❌ No images found")
        return

    for img in images:
        process_image(os.path.join(folder_path, img))

    if det_log:
        pd.DataFrame(det_log).to_csv(
            os.path.join(OUTPUT_DIR, "image_detections.csv"), index=False
        )
        print("📄 image_detections.csv saved")

# ---------------------------------
# VIDEO DETECTION + TRACKING
# ---------------------------------
def detect_and_track(frame, frame_count, timestamp, video_name, camera_id):
    results = model(frame)[0]
    detections = []

    if results.boxes is not None:
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == 0 and conf > CONF_THRESH:
                blur_face(frame, x1, y1, x2, y2)
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

                det_log.append({
                    "type": "video",
                    "video_name": video_name,
                    "camera_id": camera_id,
                    "frame": frame_count,
                    "timestamp_rel": timestamp,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "confidence": conf
                })

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        l, t, r, b = map(int, track.to_ltrb())
        track_id = track.track_id

        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        track_log.append({
            "video_name": video_name,
            "camera_id": camera_id,
            "frame": frame_count,
            "track_id": track_id,
            "x1": l, "y1": t, "x2": r, "y2": b,
            "bbox_area": (r - l) * (b - t)
        })

    return frame

# ---------------------------------
# VIDEO PROCESSING
# ---------------------------------
def process_video(video_path, camera_id=0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_name = os.path.basename(video_path)
    cam_name = CAMERA_IDS.get(camera_id, f"cam{camera_id}")

    out_name = f"{cam_name}_blurred_{video_name}"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    out = cv2.VideoWriter(out_path,
                           cv2.VideoWriter_fourcc(*"mp4v"),
                           fps, (width, height))

    frame_count = 0
    print(f"🎥 Processing video: {video_name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        frame = detect_and_track(frame, frame_count, timestamp, video_name, camera_id)
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Saved: {out_path}")

# ---------------------------------
# VIDEO FOLDER
# ---------------------------------
def process_video_folder(folder_path):
    video_exts = (".mp4", ".avi", ".mov", ".mkv")
    videos = [f for f in os.listdir(folder_path) if f.lower().endswith(video_exts)]

    if not videos:
        print("❌ No videos found")
        return

    for idx, vid in enumerate(videos):
        process_video(os.path.join(folder_path, vid), camera_id=min(idx, 1))

    if det_log:
        pd.DataFrame(det_log).to_csv(
            os.path.join(OUTPUT_DIR, "video_detections.csv"), index=False
        )

    if track_log:
        pd.DataFrame(track_log).to_csv(
            os.path.join(OUTPUT_DIR, "video_tracks.csv"), index=False
        )

    print("📊 Video CSVs saved")

# ---------------------------------
# MAIN
# ---------------------------------
if __name__ == "__main__":

    if os.path.isdir(INPUT_PATH):
        files = os.listdir(INPUT_PATH)

        if any(f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")) for f in files):
            process_image_folder(INPUT_PATH)

        elif any(f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")) for f in files):
            process_video_folder(INPUT_PATH)

        else:
            print("❌ Unsupported folder contents")

    elif INPUT_PATH.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        process_image(INPUT_PATH)

    elif INPUT_PATH.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        process_video(INPUT_PATH)

    else:
        print("❌ Unsupported input type")