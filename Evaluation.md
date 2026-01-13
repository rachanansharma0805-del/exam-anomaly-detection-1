# Evaluation Metrics–
# Week 2
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
{Precision} = frac{29}/{30} = 0.96

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

{Recall} = frac{8}/{17} = 0.47

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

  ---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Week 3

This document explains how the pose estimation and anomaly detection system was evaluated during Week 3 of the project. The main goal of Week 3 was to ensure that:
* Pose estimation works reliably on detected and tracked people
* Rule-based anomaly detection functions correctly
* Anomalies are logged consistently
* The system remains stable enough for future extensions
Evaluation was performed using manual inspection and anomaly log analysis.
---
## Evaluation Goals

The evaluation focused on validating the following objectives:
* Accurate pose estimation of individuals in video frames
* Correct triggering of defined anomaly detection rules
* Identification of false anomaly detections and missed anomalies
* Stability of anomaly detection across consecutive frames
* Baseline estimation of Precision and Recall
---
## Evaluation Method

The system was evaluated using:
* Manual visual inspection of processed video output
* Analysis of generated evaluation documents:

  * **anomaly_log.csv**
  * **manual_review_results.csv**

The video (**processed_output_week3.mp4**) was paused at selected timestamps to compare:

* Actual human behavior in the frame
* Anomalies detected by the system
* Correctness of anomaly classification
---
## Manual Visual Inspection

During evaluation, the following were observed manually:
* Accuracy of pose landmarks (head, hands, shoulders)
* Stability of pose landmarks during movement
* Correct detection of anomaly types (e.g., looking away, excessive hand movement)
* False anomaly detections
* Missed anomaly events
This helped verify that pose estimation and anomaly detection were functioning as expected under real-world conditions.
---
## Precision Evaluation

Precision was estimated using the **anomaly_log.pdf** and verified with **manual_review_results.pdf**.

### Method:

* All anomaly alerts generated by the system were reviewed
* Correct anomaly detections were counted as **True Positives (TP)**
* Incorrect anomaly detections were counted as **False Positives (FP)**

**Precision formula:**

Precision = TP / (TP + FP)

Where:
* True Positives (TP): Correctly detected anomaly events
* False Positives (FP): Incorrect anomaly alerts

### Result:

* True Positives (TP): 78
* False Positives (FP): 16

Precision = 78 / (78 + 16) ≈ **0.83**

### Interpretation:

The precision value indicates that most anomaly alerts raised by the system are correct. Some false positives occur due to natural human movements, pose ambiguity, invigilator misidentification, and ID overlap.
---
## Recall Evaluation

Recall was estimated by identifying how many actual anomaly events were successfully detected by the system.

### Method:

* Actual anomaly events were identified through manual observation of the video
* Logged anomaly events were compared against manual observations
* Missed anomaly events were counted as **False Negatives (FN)**

**Recall formula:**

Recall = TP / (TP + FN)
Where:
* True Positives (TP): Correctly detected anomaly events
* False Negatives (FN): Missed anomaly events

### Result:

* True Positives (TP): 78
* False Negatives (FN): 5
Recall = 78 / (78 + 5) ≈ **0.94**

### Interpretation:
The high recall indicates that most real anomaly events were detected, with very few missed cases.
---

## Key Observations (Week 3)

* Pose estimation performs reliably for most viewing angles
* Anomaly detection rules trigger correctly in the majority of cases
* High recall ensures strong anomaly coverage
* Some false positives occur due to rapid or ambiguous gestures
* Motion blur and ID overlap affect detection accuracy in a few cases
* The system remains stable during real-time execution
---
## Limitations

* No labeled ground-truth dataset was used
* Anomaly labels were assigned manually
* Results depend on video quality and camera placement
* Rule-based detection lacks contextual understanding
* Face-blur feature creates ambiguity in detection
---
## Conclusion

The pose estimation and anomaly detection pipeline functions correctly.
Anomaly logs are generated consistently and support manual evaluation.
Performance metrics are acceptable for a prototype system, and limitations are clearly identified, making the system suitable for further refinement and optimization.
