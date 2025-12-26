import os
import cv2
import pandas as pd
import numpy as np
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ==================== CONFIGURATION ====================
YOLO_MODEL = "yolov8n.pt"          # Fastest model for CPU
CONF_THRESH = 0.35                 # Balance between detection and speed
IMG_SIZE = 320                     # Optimal for CPU
INPUT_PATH = "data/videos/malprctice1.mp4"
OUTPUT_DIR = "data/friday_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== INITIALIZATION ====================
print("🔄 Loading models...")
model = YOLO(YOLO_MODEL)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global variables
detections_log = []
tracks_log = []
last_fps_time = time.time()
fps_counter = 0

# ==================== OPTIMIZED FUNCTIONS ====================
def fast_blur(frame, x1, y1, x2, y2):
    """Fast blurring for CPU"""
    h, w = frame.shape[:2]
    
    # Ensure valid coordinates
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return frame
    
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame
    
    # Fast blur using resize
    try:
        small = cv2.resize(roi, (8, 8))
        blurred = cv2.resize(small, (x2 - x1, y2 - y1))
        frame[y1:y2, x1:x2] = blurred
    except:
        # Fallback
        frame[y1:y2, x1:x2] = cv2.blur(roi, (15, 15))
    
    return frame

def detect_faces_fast(frame, x1, y1, x2, y2):
    """Fast face detection within person bounding box"""
    # Extract upper body region (faces are usually here)
    person_height = y2 - y1
    face_search_height = min(person_height // 2, 150)
    
    face_region = frame[y1:y1+face_search_height, x1:x2]
    if face_region.size == 0:
        return []
    
    # Convert to grayscale
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    
    # Equalize histogram for better detection
    gray = cv2.equalizeHist(gray)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30),
        maxSize=(150, 150)
    )
    
    # Adjust coordinates
    return [(x1 + fx, y1 + fy, fw, fh) for (fx, fy, fw, fh) in faces]

def process_frame(frame, frame_id, timestamp, video_name, camera_id, tracker):
    """Process a single frame with detection and tracking"""
    global detections_log, tracks_log, last_fps_time, fps_counter
    
    # Initialize FPS tracking on first frame
    if not hasattr(process_frame, 'last_time'):
        process_frame.last_time = time.time()
        process_frame.frame_count = 0
        process_frame.fps = 0
    
    current_time = time.time()
    process_frame.frame_count += 1
    
    # Calculate FPS every 30 frames
    if process_frame.frame_count % 30 == 0:
        time_diff = current_time - process_frame.last_time
        if time_diff > 0:
            process_frame.fps = 30 / time_diff
        process_frame.last_time = current_time
    
    # Detect persons every 3rd frame (optimization)
    detections = []
    if frame_id % 3 == 0:
        results = model(
            frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRESH,
            classes=[0],
            max_det=20,
            verbose=False,
            device='cpu'
        )[0]
        
        if results.boxes is not None:
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf >= CONF_THRESH:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    width, height = x2 - x1, y2 - y1
                    
                    # Basic size validation
                    if width > 30 and height > 40:
                        detections.append(([x1, y1, width, height], conf, "person"))
    
    # Update tracker
    tracks = []
    if tracker and detections:
        tracked = tracker.update_tracks(detections, frame=frame)
        tracks = [t for t in tracked if t.is_confirmed()]
    
    # Process each tracked person
    current_tracks = []
    for track in tracks:
        l, t, r, b = map(int, track.to_ltrb())
        tid = track.track_id
        
        # Blur the person
        frame = fast_blur(frame, l, t, r, b)
        
        # Detect and blur faces (every 5th frame for speed)
        if frame_id % 5 == 0:
            faces = detect_faces_fast(frame, l, t, r, b)
            for fx, fy, fw, fh in faces:
                frame = fast_blur(frame, fx, fy, fx+fw, fy+fh)
        
        # Draw bounding box and ID
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{tid}", (l, t-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        current_tracks.append(tid)
        
        # Log tracking
        tracks_log.append({
            "video": video_name,
            "camera": camera_id,
            "frame": frame_id,
            "timestamp": timestamp,
            "person_id": tid,
            "x1": l, "y1": t, "x2": r, "y2": b
        })
    
    # Display FPS and info
    cv2.putText(frame, f"FPS: {process_frame.fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_id}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Persons: {len(current_tracks)}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return frame, current_tracks

# ==================== VIDEO PROCESSING ====================
def process_video(video_path, camera_id=0):
    """Process a video file"""
    print(f"\n🎥 Processing: {os.path.basename(video_path)}")
    
    # Reset FPS counter for this video
    if hasattr(process_frame, 'last_time'):
        del process_frame.last_time
        del process_frame.frame_count
        del process_frame.fps
    
    # Initialize tracker
    tracker = DeepSort(
        max_age=30,
        n_init=3,
        embedder="mobilenet",
        half=False,  # CPU doesn't use half precision
        bgr=True,
        max_cosine_distance=0.4,
        nn_budget=50
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Reduce processing resolution for speed
    process_scale = 0.75
    process_w, process_h = int(w * process_scale), int(h * process_scale)
    
    video_name = os.path.basename(video_path)
    out_path = os.path.join(OUTPUT_DIR, f"output_{video_name}")
    
    # Create video writer
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        min(fps, 15),  # Cap at 15 FPS
        (process_w, process_h)
    )
    
    print(f"📊 Original: {w}x{h} @ {fps}fps")
    print(f"⚡ Processing at: {process_w}x{process_h}")
    
    # Processing loop
    frame_id = 0
    processed_count = 0
    start_time = time.time()
    last_print_time = start_time
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        
        # Skip frames to achieve ~10 FPS processing
        if fps > 20 and frame_id % (fps // 10) != 0:
            continue
        
        # Resize frame for faster processing
        frame_resized = cv2.resize(frame, (process_w, process_h))
        
        # Process frame
        processed_frame, tracks = process_frame(
            frame_resized, 
            frame_id, 
            frame_id/fps, 
            video_name, 
            f"cam{camera_id}", 
            tracker
        )
        
        # Write frame
        writer.write(processed_frame)
        processed_count += 1
        
        # Print progress every 5 seconds
        current_time = time.time()
        if current_time - last_print_time >= 5:
            elapsed = current_time - start_time
            processing_fps = processed_count / elapsed if elapsed > 0 else 0
            progress = (frame_id / total_frames) * 100
            
            print(f"⏳ Frame {frame_id}/{total_frames} ({progress:.1f}%) | "
                  f"FPS: {processing_fps:.1f} | "
                  f"Tracks: {len(tracks)}")
            last_print_time = current_time
    
    # Cleanup
    cap.release()
    writer.release()
    
    # Final stats
    total_time = time.time() - start_time
    avg_fps = processed_count / total_time if total_time > 0 else 0
    
    print(f"\n✅ Saved: {out_path}")
    print(f"📊 Final stats:")
    print(f"   Time: {total_time:.1f}s")
    print(f"   FPS: {avg_fps:.1f}")
    print(f"   Processed: {processed_count}/{total_frames} frames")
    print(f"   Tracks logged: {len([t for t in tracks_log if t['video'] == video_name])}")
    
    return avg_fps

# ==================== MAIN FUNCTION ====================
def main():
    """Main entry point"""
    print("🚀 Fast CPU Person Detection & Face Blurring")
    print("=" * 50)
    print(f"Model: {YOLO_MODEL}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Confidence: {CONF_THRESH}")
    print("=" * 50)
    
    # Check input path
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Input path not found: {INPUT_PATH}")
        print("Testing with webcam instead...")
        
        # Webcam test
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return
        
        # Set webcam resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        tracker = DeepSort(max_age=30, n_init=3, embedder="mobilenet", half=False, bgr=True)
        
        frame_id = 0
        fps_history = []
        last_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id += 1
            
            # Process frame
            processed, tracks = process_frame(
                frame, frame_id, frame_id/30, "webcam", "cam0", tracker
            )
            
            # Calculate and display FPS
            current_time = time.time()
            time_diff = current_time - last_time
            if time_diff > 0:
                fps = 1.0 / time_diff
                fps_history.append(fps)
                if len(fps_history) > 10:
                    fps_history.pop(0)
            
            avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
            last_time = current_time
            
            # Display FPS
            cv2.putText(processed, f"FPS: {avg_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Live Detection - Press 'q' to quit", processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n📊 Average webcam FPS: {avg_fps:.1f}")
        return
    
    # Process video(s)
    if os.path.isdir(INPUT_PATH):
        videos = [v for v in os.listdir(INPUT_PATH) 
                 if v.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not videos:
            print("❌ No videos found in directory")
            return
        
        print(f"📁 Found {len(videos)} videos")
        
        results = []
        for i, video in enumerate(videos):
            print(f"\n{'='*40}")
            print(f"Processing {i+1}/{len(videos)}: {video}")
            
            video_path = os.path.join(INPUT_PATH, video)
            fps = process_video(video_path, i % 2)
            results.append((video, fps))
        
        # Summary
        if results:
            print(f"\n{'='*40}")
            print("📊 PROCESSING SUMMARY")
            for video, fps in results:
                print(f"📹 {video}: {fps:.1f} FPS")
            
            avg_fps = sum(r[1] for r in results) / len(results)
            print(f"\n📈 Average FPS: {avg_fps:.1f}")
            
            # Save CSV logs
            if tracks_log:
                df = pd.DataFrame(tracks_log)
                df.to_csv(os.path.join(OUTPUT_DIR, "tracks.csv"), index=False)
                print(f"📄 Saved {len(df)} track entries")
    
    elif os.path.isfile(INPUT_PATH):
        process_video(INPUT_PATH)
        
        # Save CSV log
        if tracks_log:
            df = pd.DataFrame(tracks_log)
            df.to_csv(os.path.join(OUTPUT_DIR, "tracks.csv"), index=False)
            print(f"📄 Saved {len(df)} track entries")
    
    print("\n✅ Processing complete!")

# ==================== RUN ====================
if __name__ == "__main__":
    main()