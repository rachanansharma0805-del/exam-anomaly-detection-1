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
[Exam Anomaly Detection Project](https://trello.com/b/UfJU17FE/exam-anomaly-detection-project)

## System Workflow 
Video Input
->
YOLOv8 (Person Detection)
->
DeepSORT (Tracking)
->
MediaPipe (Pose Estimation)
->
Feature Extraction
->
Anomaly Detection
->
Logs & Dashboard Output

## Project Structure
src/        - core source code  
data/       - input images/videos  
outputs/    - results and logs  
docs/       - diagrams and screenshots 

## Installation & Setup
### 1. Clone the repository
git clone https://github.com/rachanansharma0805-del/exam-anomaly-detection-1 
 [github_link](https://github.com/rachanansharma0805-del/exam-anomaly-detection-1)

### 2.Create virtual environment
- python -m venv env
- env\Scripts\activate

### 3.Install dependencies
- pip install -r requirements.txt

### 4.Install packages
- py -m pip install --upgrade pip
- py -m pip install opencv-python
- py -m pip install opencv-contrib-python
- py -m pip install mediapipe
- py -m pip install --upgrade pip setuptools wheel
-  py -m pip install "numpy<2"
-  py -m pip install mediapipe
-  py -m pip install torch torchvision torchaudio
-  py -m pip install ultralytics or py -m pip install ultralytics --no-deps and then py -m pip install torch torchvision
-  pip freeze > requirements.txt

## Week 1 Progress
- Development environment set up successfully  
- YOLOv8 pretrained model tested on sample image/video  
- Person detection verified with bounding boxes  
- Initial observations documented for further improvements

## How to Run the Project
Currently under development.  
Initial setup and environment configuration completed.  
Execution steps will be updated as modules are implemented.


