import cv2
import mediapipe as mp
from ultralytics import YOLO

# Load YOLOv8 model (pretrained on COCO dataset)
model = YOLO("yolov8n.pt")  # lightweight model for quick testing

# Initialize Mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 detection
    results = model(frame)
    annotated_frame = results[0].plot()

    # Mediapipe pose detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)

    if pose_results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("YOLOv8 + Mediapipe", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()