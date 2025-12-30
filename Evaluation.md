# Evaluation Metrics– Week 2
This document explains how the people detection and tracking system was evaluated during **Week 2** of the project.
The main goal of Week 2 was to ensure that:
- People detection works reliably
- Tracking IDs remain reasonably consistent
- The system is stable enough to be extended in Week 3 (pose estimation and anomaly detection)
  Evaluation was performed using **manual inspection** and **CSV log analysis**.

## Evaluation Goals
The evaluation focused on validating the following objectives:
- Accurate detection of people in video frames
- Reasonable tracking ID consistency across frames
- Identification of false detections and missed detections
- Measurement of system performance (FPS)
- Baseline estimation of Precision and Recall

## Evaluation Method
The system was evaluated using:
- Manual visual inspection of video output
- Analysis of generated CSV files:
  - `detect_logs.csv`
  - `track_logs.csv`

The video was paused at selected intervals to compare:
- Actual number of people in the frame
- Number of detections made by the system
- Tracking behavior and ID stability

## Manual Visual Inspection
During evaluation, the following were observed manually:
- Number of people present vs detected
- Missed detections (people not detected)
- False detections (incorrect bounding boxes)
- Tracking ID flickering or resets
- Approximate FPS during execution
This helped verify that detection and tracking were working as expected in real-world video conditions.

## Precision Evaluation
Precision was estimated using the `detect_logs.csv` file.

### Method:
- A frame with **30 people** present was selected.
- Detections with **high confidence** were counted as correct detections.
### Formula:
#### Precision=
          True Positives (TP)/True Positives (TP)+False Positives (FP)
Precision tells us how many of the detected objects were actually correct.
True Positives (TP): Correctly detected people
False Positives (FP): Incorrect detections (detections where no person was present)

### Result:
- Correct detections: **29**
- Total people present: **30**

\[
\text{Precision} = \frac{29}{30} = 0.96
\]

**Interpretation:**  
The high precision indicates that when the system detects a person, it is usually correct, with very few false positives.

## Recall Evaluation
Recall was estimated using the `track_logs.csv` file.

### Method:
- All unique tracking IDs were extracted.
- The number of frames each ID appeared in was calculated.
- A threshold of **30 or more frames** was used to identify valid and stable tracks.

### Formula:
 #### Recall=
      True Positives (TP)/True Positives (TP)+False Negatives (FN)
Recall tells us how many actual people were successfully detected/tracked.
True Positives (TP): People correctly detected and tracked
False Negatives (FN): People present but missed or not consistently tracked

### Result:
- Total unique tracks: **17**
- Valid tracks (≥ 30 frames): **8**

\[
\text{Recall} = \frac{8}{17} = 0.47
\]

**Interpretation:**  
Some people were not tracked consistently across frames, leading to lower recall.

## Key Observations (Week 2)
- Detection accuracy is strong (high precision)
- Tracking consistency needs improvement
- ID flickering affects recall
- Occlusions and movement impact tracking
- CPU-based inference limits real-time performance
  
## Limitations
- No labeled ground-truth dataset was used
- Metrics were estimated manually and from logs
- Results are video- and environment-dependent

## Conclusion
- The detection and tracking pipeline functions correctly
- Logs are generated reliably
- Performance issues and limitations are clearly identified.
