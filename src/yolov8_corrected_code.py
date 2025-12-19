import cv2
import mediapipe as mp
from ultralytics import YOLO
import os

# -------------------------------
# Create folder for screenshots
# -------------------------------
os.makedirs("docs/tests", exist_ok=True)

# -------------------------------
# MediaPipe Pose setup (lightweight)
# -------------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,  # lightweight for faster processing
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------------------
# Load YOLOv8 model
# -------------------------------
model = YOLO("yolov8n.pt")  # lightweight pretrained model

# -------------------------------
# Open webcam (Windows optimized)
# -------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

# -------------------------------
# Screenshot counter and frame control
# -------------------------------
screenshot_count = 0
frame_count = 0  # used to skip YOLO on alternate frames

# -------------------------------
# Main loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    frame_resized = cv2.resize(frame, (640, 480))
    frame_count += 1

    # -------------------------------
    # YOLO detection (every 2 frames for speed)
    # -------------------------------
    if frame_count % 2 == 0:
        results = model.predict(frame_resized, conf=0.5)
        annotated_frame = results[0].plot()
    else:
        annotated_frame = frame_resized.copy()

    # -------------------------------
    # MediaPipe Pose detection
    # -------------------------------
    rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb)

    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # -------------------------------
    # Display live feed
    # -------------------------------
    cv2.imshow("YOLOv8 + MediaPipe Pose", annotated_frame)

    # -------------------------------
    # Keyboard controls
    # -------------------------------
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Quit
        break
    elif key == ord("s"):  # Save screenshot
        screenshot_count += 1
        filename = f"docs/tests/output_frame_{screenshot_count}.png"
        cv2.imwrite(filename, annotated_frame)
        print(f"✅ Screenshot saved: {filename}")

# -------------------------------
# Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()
