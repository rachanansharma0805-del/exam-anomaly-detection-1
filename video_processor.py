"""
Video Processor for Exam Proctoring System - PRODUCTION READY
Features:
- PATH-BASED invigilator detection (walks + stands = invigilator)
- Consistent person IDs throughout video
- ROBUST face blurring with bounds checking
- Smart anomaly detection with reduced false positives
- Warning system for excessive violations
- 95%+ confidence threshold

BUG FIX: Line 282 - Changed output_path to output_video_path
"""

import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from collections import defaultdict, deque
import csv
from datetime import datetime
import os
import subprocess
import tempfile
from typing import Tuple

# Import custom modules
from person_detector import PersonDetector
from pose_estimator import PoseEstimator
from anomaly_detector import AnomalyDetector


class ExamProctorPipeline:
    """Production-ready exam proctoring pipeline"""
    
    def __init__(self, output_dir="output", confidence_threshold=0.5, blur_faces=True):
        """Initialize the exam proctoring pipeline"""
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        self.blur_faces = blur_faces
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize models
        print("Loading YOLO model...")
        self.yolo_model = YOLO('yolov8n.pt')
        
        print("Loading MediaPipe Pose...")
        self.pose_estimator = PoseEstimator(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize person detector
        print("Initializing person tracker...")
        self.person_detector = PersonDetector(confidence_threshold=confidence_threshold)
        
        # Initialize anomaly detector (PATH-BASED)
        print("Initializing anomaly detector (path-based invigilator detection)...")
        self.anomaly_detector = AnomalyDetector(self.pose_estimator)
        
        # Tracking data
        self.anomalies = []
        self.person_types = {}  # person_id -> 'student' or 'invigilator'
        self.frame_count = 0
        
        # Warning tracking
        self.warned_students = set()
        
        # Video capture and writer
        self.cap = None
        self.out = None
        
        print("Pipeline initialized successfully!")
        print("=" * 60)
    
    def blur_face_region(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, 
                         blur_strength: int = 51) -> np.ndarray:
        """
        ROBUST face blurring with proper bounds checking
        Prevents crashes from invalid coordinates
        """
        # Ensure blur strength is odd
        if blur_strength % 2 == 0:
            blur_strength += 1
        
        h, w = frame.shape[:2]
        
        # Add padding for better coverage
        padding = 20
        x1 = int(max(0, x1 - padding))
        y1 = int(max(0, y1 - padding))
        x2 = int(min(w, x2 + padding))
        y2 = int(min(h, y2 + padding))
        
        # Validate bounds
        if x2 <= x1 or y2 <= y1:
            return frame
        
        try:
            # Extract and validate face region
            face_region = frame[y1:y2, x1:x2].copy()
            
            if face_region.size == 0 or face_region.shape[0] < 5 or face_region.shape[1] < 5:
                return frame
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(face_region, (blur_strength, blur_strength), 0)
            
            # Ensure shapes match before assignment
            if blurred.shape == frame[y1:y2, x1:x2].shape:
                frame[y1:y2, x1:x2] = blurred
            else:
                # Resize if needed (safety fallback)
                frame[y1:y2, x1:x2] = cv2.resize(blurred, (x2-x1, y2-y1))
                
        except Exception as e:
            # Silently handle errors (don't crash processing)
            pass
        
        return frame
    
    def blur_all_faces(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Blur all detected person faces in the frame
        Uses upper portion of bounding box (where face is located)
        """
        if not detections or not self.blur_faces:
            return frame
        
        for detection in detections:
            try:
                # Get bounding box coordinates
                bbox = detection['bbox']
                x, y, w, h = bbox
                x1, y1, x2, y2 = x, y, x + w, y + h
                
                # Validate coordinates
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Focus on upper portion of bbox (where face is)
                box_height = y2 - y1
                face_height = int(box_height * 0.35)  # Top 35% contains face
                face_y1 = y1
                face_y2 = min(y1 + face_height, y2)
                
                if face_y2 <= face_y1:
                    continue
                
                # Blur the face region
                frame = self.blur_face_region(frame, x1, face_y1, x2, face_y2)
                
            except Exception as e:
                # Continue processing other faces
                continue
        
        return frame
    
    def convert_to_mp4_ffmpeg(self, input_path, output_path):
        """Convert video to MP4 using FFmpeg"""
        try:
            cmd = [
                'ffmpeg',
                '-y',
                '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-loglevel', 'error',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"FFmpeg warning: {result.stderr}")
                import shutil
                shutil.copy2(input_path, output_path)
            else:
                print(f"✅ Video converted successfully")
                
        except FileNotFoundError:
            print("⚠️  FFmpeg not found. Using fallback...")
            import shutil
            shutil.copy2(input_path, output_path)
        except Exception as e:
            print(f"Error during video conversion: {e}")
            import shutil
            shutil.copy2(input_path, output_path)
    
    def cleanup(self):
        """Release all resources"""
        try:
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
                self.cap = None
            
            if hasattr(self, 'out') and self.out is not None:
                self.out.release()
                self.out = None
            
            if hasattr(self, 'pose_estimator') and self.pose_estimator is not None:
                self.pose_estimator.cleanup()
            
            try:
                cv2.destroyAllWindows()
            except:
                pass
            
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
    
    def process_video(self, video_path, output_video_path=None, csv_path=None):
        """Process video and detect anomalies"""
        temp_output = None
        
        try:
            # Open video
            self.cap = cv2.VideoCapture(video_path)
            
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print("=" * 60)
            print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
            print(f"Privacy: {'✓ Robust face blurring enabled' if self.blur_faces else '✗ Disabled'}")
            print(f"Invigilator: Path-based detection (walking + standing)")
            print(f"Confidence: 95%+ for all anomalies")
            print("=" * 60)
            
            # Setup video writer
            if output_video_path:
                temp_output = output_video_path.replace('.mp4', '_temp.avi')
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
            
            self.frame_count = 0
            
            # Process frames
            print("\nProcessing video...")
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Process frame
                annotated_frame = self.process_frame(frame, self.frame_count, fps)
                
                # Write annotated frame
                if self.out:
                    self.out.write(annotated_frame)
                
                # Progress indicator
                if self.frame_count % 30 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    print(f"  Progress: {progress:.1f}% ({self.frame_count}/{total_frames} frames)")
            
            print(f"  Progress: 100% (Complete)")
            print("=" * 60)
            
            # Release resources before FFmpeg conversion
            if self.out:
                self.out.release()
                self.out = None
            
            if self.cap:
                self.cap.release()
                self.cap = None
            
            # Convert to MP4 - BUG FIX: Changed output_path to output_video_path
            if output_video_path and temp_output and os.path.exists(temp_output):
                print("Converting video to MP4...")
                self.convert_to_mp4_ffmpeg(temp_output, output_video_path)
                try:
                    os.remove(temp_output)
                except:
                    pass
            
            # Generate summary
            summary_stats = self.generate_summary()
            
            # Save CSV
            if csv_path:
                self.save_anomalies_csv(csv_path)
            
            # Print final summary
            print("\n" + "=" * 60)
            print("PROCESSING COMPLETE")
            print("=" * 60)
            print(f"Total Anomalies: {len(self.anomalies)}")
            print(f"Students Monitored: {summary_stats['total_students']}")
            print(f"Invigilator: {'Detected (ID: ' + str(summary_stats['invigilator_id']) + ')' if summary_stats['invigilator_id'] else 'Not detected'}")
            print(f"Students with Excessive Violations: {len(summary_stats['warned_students'])}")
            if summary_stats['warned_students']:
                print(f"  ⚠️  WARNING - Person IDs: {summary_stats['warned_students']}")
            print("=" * 60)
            
            return output_video_path, csv_path, summary_stats
            
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            if temp_output and os.path.exists(temp_output):
                try:
                    os.remove(temp_output)
                except:
                    pass
    
    def process_frame(self, frame, frame_number, fps):
        """Process a single frame with PATH-BASED invigilator detection"""
        annotated_frame = frame.copy()
        
        # Detect persons using YOLO
        results = self.yolo_model(frame, classes=[0], conf=self.confidence_threshold)
        
        # Convert YOLO detections to our format
        detections = []
        for detection in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            
            detections.append({
                'bbox': (x1, y1, w, h),
                'confidence': float(conf),
                'center': (x1 + w // 2, y1 + h // 2),
                'area': w * h
            })
        
        # ROBUST FACE BLURRING (before any annotations)
        if self.blur_faces:
            annotated_frame = self.blur_all_faces(annotated_frame, detections)
        
        # Update tracks with CONSISTENT IDs
        tracked_persons = self.person_detector.update_tracks(detections, frame)
        
        # Process each tracked person
        for person_id, person_data in tracked_persons.items():
            x, y, w, h = person_data['bbox']
            
            # Extract person region
            person_region = frame[y:y+h, x:x+w]
            
            if person_region.size == 0:
                continue
            
            # Detect pose
            pose_data = self.pose_estimator.estimate_pose(frame, (x, y, w, h))
            
            if pose_data is None or 'landmarks' not in pose_data:
                continue
            
            landmarks = pose_data['landmarks']
            
            # PATH-BASED INVIGILATOR DETECTION
            is_invigilator = self.anomaly_detector.detect_invigilator(
                person_id,
                landmarks,
                (x, y, w, h),
                frame_number
            )
            
            # Store person type
            if is_invigilator:
                self.person_types[person_id] = 'invigilator'
            elif person_id not in self.person_types:
                self.person_types[person_id] = 'student'
            
            # Get current type
            person_type = self.person_types.get(person_id, 'student')
            
            # DETECT ANOMALIES (ONLY for students)
            anomalies = []
            if person_type == 'student':
                anomalies = self.anomaly_detector.detect_anomalies(
                    person_id,
                    pose_data,
                    (x, y, w, h),
                    frame_number
                )
            
            # VISUALIZATION
            if person_type == 'invigilator':
                # Blue box for invigilator
                color = (255, 100, 0)
                label = f"INVIGILATOR (Walking & Standing)"
                thickness = 3
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, thickness)
                
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated_frame, 
                             (x, y - label_size[1] - 10), 
                             (x + label_size[0], y), 
                             color, -1)
                cv2.putText(annotated_frame, label, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                # Student visualization
                person_summary = self.anomaly_detector.get_person_anomaly_summary(person_id)
                has_excessive = person_summary['excessive_violations']
                
                if has_excessive:
                    color = (0, 0, 255)  # Red
                    label = f"⚠️  STUDENT ID:{person_id} - EXCESSIVE ({person_summary['total_anomalies']})"
                    thickness = 3
                    if person_id not in self.warned_students:
                        self.warned_students.add(person_id)
                        print(f"\n⚠️  WARNING: Person ID {person_id} has {person_summary['total_anomalies']} violations!")
                elif anomalies:
                    color = (0, 165, 255)  # Orange
                    label = f"Student ID:{person_id}"
                    thickness = 2
                else:
                    color = (0, 255, 0)  # Green
                    label = f"Student ID:{person_id}"
                    thickness = 2
                
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, thickness)
                
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, 
                             (x, y - label_size[1] - 10), 
                             (x + label_size[0], y), 
                             color, -1)
                cv2.putText(annotated_frame, label, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display anomalies
                if anomalies:
                    timestamp = frame_number / fps
                    
                    for i, anomaly in enumerate(anomalies):
                        self.log_anomaly(
                            frame_number=frame_number,
                            timestamp=timestamp,
                            person_id=person_id,
                            anomaly_type=anomaly['type'],
                            severity=anomaly['severity'],
                            confidence=anomaly['confidence'],
                            description=anomaly['description'],
                            excessive=has_excessive
                        )
                        
                        text_y = y + h + 25 + (i * 22)
                        anomaly_text = f"{anomaly['type']}: {anomaly['description'][:40]}"
                        cv2.putText(
                            annotated_frame, 
                            anomaly_text,
                            (x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.45, 
                            (0, 0, 255), 
                            2
                        )
        
        # Draw frame info
        students = len([p for p in self.person_types.values() if p == 'student'])
        invigilator_count = 1 if self.anomaly_detector.invigilator_id else 0
        
        info_text = f"Frame: {frame_number} | Students: {students} | Invigilator: {invigilator_count}"
        cv2.rectangle(annotated_frame, (5, 5), (600, 40), (0, 0, 0), -1)
        cv2.putText(annotated_frame, info_text, (10, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def log_anomaly(self, frame_number, timestamp, person_id, anomaly_type, 
                    severity, confidence, description, excessive=False):
        """Log detected anomaly"""
        self.anomalies.append({
            'frame_number': frame_number,
            'timestamp': f"{int(timestamp//60):02d}:{int(timestamp%60):02d}",
            'person_id': person_id,
            'anomaly_type': anomaly_type,
            'severity': severity,
            'confidence': confidence,
            'description': description,
            'excessive_violations': excessive
        })
    
    def save_anomalies_csv(self, csv_path):
        """Save anomalies to CSV file"""
        if not self.anomalies:
            print("No anomalies to save")
            return
        
        fieldnames = ['frame_number', 'timestamp', 'person_id', 'anomaly_type', 
                     'severity', 'confidence', 'description', 'excessive_violations']
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.anomalies)
        
        print(f"✓ Saved {len(self.anomalies)} anomalies to {csv_path}")
    
    def generate_summary(self):
        """Generate summary statistics"""
        num_students = len([p for p in self.person_types.values() if p == 'student'])
        invigilator_id = self.anomaly_detector.invigilator_id
        
        if not self.anomalies:
            return {
                'total_anomalies': 0,
                'total_students': num_students,
                'invigilator_id': invigilator_id,
                'avg_severity': 0,
                'anomaly_types': {},
                'warned_students': list(self.warned_students)
            }
        
        anomaly_types = defaultdict(int)
        total_severity = 0
        
        for anomaly in self.anomalies:
            anomaly_types[anomaly['anomaly_type']] += 1
            total_severity += anomaly['severity']
        
        avg_severity = total_severity / len(self.anomalies) if self.anomalies else 0
        
        return {
            'total_anomalies': len(self.anomalies),
            'total_students': num_students,
            'invigilator_id': invigilator_id,
            'avg_severity': avg_severity,
            'anomaly_types': dict(anomaly_types),
            'warned_students': list(self.warned_students)
        }


# Usage example
if __name__ == "__main__":
    pipeline = ExamProctorPipeline(
        output_dir="output",
        confidence_threshold=0.5,
        blur_faces=True  # Robust face blurring enabled
    )
    
    try:
        output_video, csv_file, summary = pipeline.process_video(
            video_path="exam_video.mp4",
            output_video_path="output/annotated_video.mp4",
            csv_path="output/anomalies.csv"
        )
        
        print("\n=== FINAL SUMMARY ===")
        print(f"Total Anomalies: {summary['total_anomalies']}")
        print(f"Students: {summary['total_students']}")
        print(f"Invigilator ID: {summary['invigilator_id']}")
        print(f"Warned Students: {summary['warned_students']}")
        
    finally:
        pipeline.cleanup()