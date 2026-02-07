# Exam Hall Anomaly Detection System

## Project Description
The **Exam Hall Anomaly Detection System** is an AI-based computer vision project designed to automatically monitor examination hall videos and detect suspicious or abnormal student behavior.

The system acts as a **digital invigilator** by analyzing video footage to identify activities such as excessive head movement, unusual posture, and abnormal hand movements using object detection, tracking, and pose estimation techniques.

---

## Objectives
- Detect students in exam hall videos
- Track multiple students across frames
- Analyze posture, head orientation, and movement patterns
- Identify suspicious or abnormal behavior using rule-based logic
- Generate logs and visual evidence for manual review

---

## Features
- Person detection using **YOLOv8**
- Multi-person tracking using **DeepSORT**
- Pose estimation using **MediaPipe**
- Rule-based anomaly detection
- Timestamped anomaly logging (CSV)
- Evidence frame capture for flagged events
- Streamlit dashboard for video upload and result visualization
- Privacy-preserving face blurring

---

## Technologies Used
- Python 3.10+
- OpenCV
- YOLOv8 (Ultralytics)
- DeepSORT
- MediaPipe
- Streamlit
- NumPy
- Pandas
- Git & GitHub

---

## System Architecture Overview
The system follows a modular video analytics pipeline where each component operates independently and passes structured outputs to the next stage.

### Pipeline Flow
1. Video Input  
2. YOLOv8 – Person Detection  
3. DeepSORT – Multi-Object Tracking  
4. MediaPipe – Pose Estimation  
5. Feature Extraction  
6. Rule-Based Anomaly Detection  
7. Logs & Dashboard Visualization  

---

## Anomaly Types Detected
- Repeated head turns (looking away from exam paper)
- Excessive or abnormal hand movements
- Unusual posture or body orientation
- Prolonged suspicious behavior across frames
- Irregular movement patterns compared to baseline behavior

---

## Project Structure
exam-anomaly-detection/
│
├── src/ # Core source code
├── data/ # Input videos and sample data
├── outputs/ # Processed videos, logs, evidence frames
├── docs/ # Architecture diagrams, screenshots, reports
├── requirements.txt
├── README.md
└── Evaluation.md


---

## How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/rachanansharma0805-del/exam-anomaly-detection-1
cd exam-anomaly-detection-1

### 2. Create Virtual Environment
python -m venv env
env\Scripts\activate

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Run the Application
streamlit run app.py

Upload an exam hall video through the dashboard to start processing.

## Project Management

Project tasks were planned and tracked using Trello with a weekly milestone-based approach.

Trello Board:
https://trello.com/invite/b/69593f10d77282585d9e62a1/ATTIf28afd3f91d72ee65c3e4f3ec83d572475D03CF4/exam-anomaly-detection

