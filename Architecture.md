## System Overview
The system follows a modular video analytics pipeline that converts raw exam hall footage into interpretable anomaly alerts.
Each processing block corresponds directly to a module in the codebase.

## Diagram
![Architecture Diagram](docs/Architecture-diagram.png)

## Pipeline Mapping to Diagram -
### 1.Video Input
Recorded exam hall videos are loaded and preprocessed using OpenCV.
### 2.YOLOv8 Person Detection
Each frame is analysed to detect all visible students and generate bounding boxes with confidence scores.
### 3.DeepSORT Tracking
Detected students are assigned persistent Person IDs, enabling identity tracking across consecutive frames.
### 4.MediaPipe Pose Estimation
For each tracked student, body keypoints such as head, shoulders, and hands are extracted.
### 5.Feature Computation
Temporal features like head turns, hand movement patterns, and proximity between students are calculated.
### 6.Rule-Based Anomaly Detection
Heuristic rules analyse the extracted features to flag suspicious behaviour patterns.
### 7.Logging & Dashboard
Detected anomalies are logged and displayed in a Streamlit dashboard for review and verification.

## Project Structure (reference)
exam-hall-anomaly-detection/
```text
├── src/
│   ├── video_input.py
│   ├── detection_yolov8.py
│   ├── tracking_deepsort.py
│   ├── pose_mediapipe.py
│   ├── features.py
│   ├── anomaly_rules.py
│   └── dashboard_app.py
├── data/
│   ├── videos/
│   ├── logs/
│   └── screenshots/
├── docs/
│   ├── diagrams/
│   ├── tests/
│   └── reports/
├── requirements.txt
├── README.md
└── Architecture.md

## Key Design Decisions
### Interpretability first: Rule-based detection allows transparent behaviour analysis.
### Privacy-aware: No biometric or identity recognition is performed.
### Scalable design: Each module can be independently improved or replaced.

