# Exam Hall Anomaly Detection System 
## Project Description
The Exam Hall Anomaly Detection System is an AI-based computer vision project designed to automatically monitor examination hall videos and detect suspicious or abnormal student behavior.  
The system acts as a digital invigilator by identifying activities such as excessive head movements, object exchange, or unusual posture using video analysis.

## Objectives
- Detect students in exam hall videos
- Track multiple students across frames
- Analyze posture, head orientation, and movement
- Identify suspicious or abnormal behavior
- Generate logs and visual evidence for review

## Features
- Person detection using YOLOv8  
- Multi-person tracking using DeepSORT  
- Pose estimation using MediaPipe  
- Rule-based anomaly detection  
- CSV logs with timestamps  
- Streamlit dashboard for video upload and analysis  

## Technologies Used
- Python 3.10+
- OpenCV
- YOLOv8 (Ultralytics)
- DeepSORT
- MediaPipe
- Streamlit
- NumPy, Pandas
- Git & GitHub

## Project Management - Project tasks are planned and tracked using Trello with a weekly milestone-based approach.
### TRELLO BOARD FOR PROJECT TRACKING:
[Exam Anomaly Detection Project](https://trello.com/invite/b/69593f10d77282585d9e62a1/ATTIf28afd3f91d72ee65c3e4f3ec83d572475D03CF4/exam-anomaly-detection)

## System Workflow
1. Video Input  
2. YOLOv8 – Person Detection  
3. DeepSORT – Multi-object Tracking  
4. MediaPipe – Pose Estimation  
5. Feature Extraction  
6. Anomaly Detection  
7. Logs & Dashboard Output  

## Project Structure
src/        - core source code  
data/       - input images/videos  
outputs/    - results and logs  
docs/       - diagrams and screenshots 


## How to Run the Project 
## Installation & Setup
### 1. Clone the repository
git clone https://github.com/rachanansharma0805-del/exam-anomaly-detection-1 

### 2. Create virtual environment
- python -m venv env
- env\Scripts\activate

### 3. Install dependencies
- pip install -r requirements.txt
(Installs all packages)

## Week 1 Progress
- Development environment set up successfully 
- YOLOv8 pretrained model tested on sample image/video  
- Person detection verified with bounding boxes  
- Initial observations documented for further improvements

## Week 2 Progress
- Implemented person detection using YOLOv8 on video input  
- Integrated DeepSORT for multi-object tracking and consistent ID assignment  
- Applied face blurring to detected individuals for privacy preservation  
- Generated and stored detection and tracking logs in CSV format  
- Verified outputs through processed videos with bounding boxes and track IDs

## Week 3 Progress

- Performed pose estimation on video input to extract keypoints
-  Applied anomaly rules on pose data to flag suspicious behavior
-  Included more rulesand tested the code for more videos for more accurate anomaly detection
-  Defined confidence thresholds to reduce false positives
-  Logged detected anomalies with timestamps in CSV files
-  Saved sample evidence frames for flagged events
-  Compared flagged events with actual events to evaluate performance
-  Documented results and evaluation in Evaluation.md

## Week 4 Progress

- Built a Streamlit dashboard for video upload and result visualization
- Integrated the complete detection pipeline with the dashboard
- Generated concise summary tables and logged anomalies (CSV)
- Tested the system on unseen videos to verify end-to-end performance
- Created A fully working MVP with an interactive dashboard
- End-to-end video upload → processing → anomaly reporting pipeline
- Detection outputs successfully visualized and logged

## Author
Rachana Sharma 
N Meghana 
AI/ML Internship Project
