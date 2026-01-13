"""
Detection Pipeline Wrapper for Streamlit Integration
Save this as: detector.py
"""

import os
import cv2
import time
import math
import pandas as pd
import warnings
from collections import defaultdict, deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from mediapipe.python.solutions import pose as mp_pose

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")


class ExamAnomalyDetector:
    """
    Exam Hall Anomaly Detection System
    Detects: Head turns, hand movements, proximity, looking away, hands raised
    """
    
    def __init__(self, config=None):
        """Initialize detector with configuration"""
        # Default configuration
        self.config = {
            'fps_assumed': 30,
            'head_turn_threshold': 10,
            'hand_move_threshold': 25,
            'proximity_threshold': 120,
            'proximity_frames': 90,
            'look_away_threshold': 20,
            'hands_raised_threshold': 0.2,
            'movement_threshold': 80,
            'height_ratio_threshold': 0.4,
            'min_anomaly_duration': 10,
            'anomaly_cooldown': 30,
            'blur_faces': True,
            'blur_kernel_size': 99,
            'blur_sigma': 30,
            'clip_before_seconds': 2,
            'clip_after_seconds': 2
        }
        
        if config:
            self.config.update(config)
        
        # Initialize models
        self.pose_estimator = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.yolo = YOLO("yolov8n.pt")
        self.tracker = DeepSort(max_age=300)
        
        # Initialize tracking storage
        self._reset_tracking()
    
    def _reset_tracking(self):
        """Reset all tracking variables"""
        fps = self.config['fps_assumed']
        self.pose_angles = defaultdict(lambda: deque(maxlen=fps * 60))
        self.hand_moves = defaultdict(int)
        self.head_turns = defaultdict(int)
        self.proximity_counter = defaultdict(int)
        self.movement_history = defaultdict(lambda: deque(maxlen=fps*2))
        self.height_history = defaultdict(lambda: deque(maxlen=fps*2))
        self.y_history = defaultdict(lambda: deque(maxlen=fps*2))
        self.look_away_counter = defaultdict(int)
        self.hands_raised_counter = defaultdict(int)
        self.active_anomalies = defaultdict(lambda: defaultdict(int))
        self.last_logged = defaultdict(lambda: defaultdict(int))
        self.anomaly_log = []
    
    def angle(self, p1, p2):
        """Calculate angle between two points"""
        return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    
    def distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    def should_log_anomaly(self, pid, anomaly_type, frame_id):
        """Check if anomaly should be logged based on duration and cooldown"""
        if frame_id - self.last_logged[pid][anomaly_type] < self.config['anomaly_cooldown']:
            return False
        if self.active_anomalies[pid][anomaly_type] >= self.config['min_anomaly_duration']:
            self.last_logged[pid][anomaly_type] = frame_id
            return True
        return False
    
    def blur_face(self, frame, x1, y1, x2, y2):
        """Blur face region for privacy"""
        face_height = int((y2 - y1) * 0.4)
        face_y1 = max(0, y1)
        face_y2 = min(frame.shape[0], y1 + face_height)
        x1 = max(0, x1)
        x2 = min(frame.shape[1], x2)
        
        face_roi = frame[face_y1:face_y2, x1:x2]
        if face_roi.size > 0:
            blurred = cv2.GaussianBlur(
                face_roi, 
                (self.config['blur_kernel_size'], self.config['blur_kernel_size']), 
                self.config['blur_sigma']
            )
            frame[face_y1:face_y2, x1:x2] = blurred
        return frame
    
    def extract_clip(self, video_path, start_frame, end_frame, output_path, fps):
        """Extract video clip for anomaly evidence"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
        
        for _ in range(end_frame - start_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
        out.release()
        return True
    
    def process_video(self, video_path, output_dir, progress_callback=None):
        """
        Main processing function for video analysis
        
        Args:
            video_path: Path to input video
            output_dir: Directory for output files
            progress_callback: Optional callback function(frame_id, total_frames, anomaly_count)
        
        Returns:
            dict: Results including anomaly log, statistics, and file paths
        """
        # Create output directories
        evidence_dir = os.path.join(output_dir, "evidence")
        clips_dir = os.path.join(output_dir, "anomaly_clips")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(evidence_dir, exist_ok=True)
        os.makedirs(clips_dir, exist_ok=True)
        
        # Reset tracking
        self._reset_tracking()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.config['fps_assumed']
        
        # Create output video writer
        output_video_path = os.path.join(output_dir, "processed_output.mp4")
        out = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (W, H)
        )
        
        frame_id = 0
        start_time = time.time()
        
        print(f"▶️  Processing: {os.path.basename(video_path)}")
        print(f"📊 Total frames: {total_frames} ({total_frames/fps:.1f}s)")
        
        # Main processing loop
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated = frame.copy()
            
            # YOLO detection
            results = self.yolo(frame, conf=0.4, verbose=False)[0]
            detections = []
            
            for box in results.boxes:
                if int(box.cls[0]) == 0:  # person class
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append(([x1, y1, x2 - x1, y2 - y1], 0.9, "person"))
            
            # Track persons
            tracks = self.tracker.update_tracks(detections, frame=frame)
            centers = {}
            
            for track in tracks:
                if not track.is_confirmed():
                    continue
                pid = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                center = ((x1 + x2)//2, (y1 + y2)//2)
                centers[pid] = center
                self.movement_history[pid].append(center)
                self.height_history[pid].append(y2 - y1)
                self.y_history[pid].append(center[1])
            
            # Identify invigilators (staff)
            invigilators = set()
            for pid in self.movement_history:
                if len(self.movement_history[pid]) < self.movement_history[pid].maxlen:
                    continue
                x_disp = max([c[0] for c in self.movement_history[pid]]) - min([c[0] for c in self.movement_history[pid]])
                y_disp = max([c[1] for c in self.movement_history[pid]]) - min([c[1] for c in self.movement_history[pid]])
                avg_height = sum(self.height_history[pid])/len(self.height_history[pid])
                if (x_disp > self.config['movement_threshold'] or y_disp > self.config['movement_threshold']) and avg_height > H*self.config['height_ratio_threshold']:
                    invigilators.add(pid)
            
            current_frame_anomalies = defaultdict(set)
            
            # Process each tracked person
            for track in tracks:
                if not track.is_confirmed():
                    continue
                pid = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                
                # Blur faces if enabled
                if self.config['blur_faces']:
                    annotated = self.blur_face(annotated, x1, y1, x2, y2)
                
                # Skip invigilators
                if pid in invigilators:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(annotated, f"Staff {pid}", (x1, y1-8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    continue
                
                # Extract ROI for pose estimation
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                
                pose = self.pose_estimator.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                anomaly_types = []
                
                if pose.pose_landmarks:
                    lm = pose.pose_landmarks.landmark
                    
                    # Head turn detection
                    l_sh, r_sh = (lm[11].x, lm[11].y), (lm[12].x, lm[12].y)
                    current_angle = self.angle(l_sh, r_sh)
                    self.pose_angles[pid].append(current_angle)
                    
                    if len(self.pose_angles[pid]) > 2 and abs(self.pose_angles[pid][-1]-self.pose_angles[pid][-2])>15:
                        self.head_turns[pid] += 1
                    
                    if self.head_turns[pid] > self.config['head_turn_threshold']:
                        current_frame_anomalies[pid].add("Excessive Head Turn")
                    
                    # Hand movement detection
                    if abs(lm[15].x - lm[16].x) > 0.15:
                        self.hand_moves[pid] += 1
                    
                    if self.hand_moves[pid] > self.config['hand_move_threshold']:
                        current_frame_anomalies[pid].add("Excessive Hand Movement")
                    
                    # Looking away detection
                    shoulder_angle = self.angle(l_sh, r_sh)
                    if abs(shoulder_angle) > self.config['look_away_threshold']:
                        self.look_away_counter[pid] += 1
                    
                    if self.look_away_counter[pid] > 10:
                        current_frame_anomalies[pid].add("Looking Away")
                    
                    # Hands raised detection
                    if lm[15].y < lm[11].y - self.config['hands_raised_threshold'] and \
                       lm[16].y < lm[12].y - self.config['hands_raised_threshold']:
                        self.hands_raised_counter[pid] += 1
                    
                    if self.hands_raised_counter[pid] > 10:
                        current_frame_anomalies[pid].add("Hands Raised")
                
                # Log anomalies
                for anomaly_type in current_frame_anomalies[pid]:
                    self.active_anomalies[pid][anomaly_type] += 1
                    
                    if self.should_log_anomaly(pid, anomaly_type, frame_id):
                        anomaly_types.append(anomaly_type)
                        self.anomaly_log.append({
                            "frame": frame_id,
                            "person_id": pid,
                            "type": anomaly_type,
                            "timestamp": f"{frame_id/fps:.2f}",
                            "clip_start": max(0, frame_id - self.config['clip_before_seconds']*fps),
                            "clip_end": min(total_frames-1, frame_id + self.config['clip_after_seconds']*fps),
                            "confidence": 85.0 + (frame_id % 15)  # Mock confidence
                        })
                        cv2.imwrite(
                            os.path.join(evidence_dir, f"{anomaly_type.replace(' ','')}_{pid}_f{frame_id}.jpg"), 
                            annotated
                        )
                
                # Reset counters for inactive anomalies
                for anomaly_type in list(self.active_anomalies[pid].keys()):
                    if anomaly_type not in current_frame_anomalies[pid]:
                        self.active_anomalies[pid][anomaly_type] = 0
                
                # Draw bounding boxes
                box_color = (0, 255, 0) if not anomaly_types else (0, 0, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(annotated, f"ID {pid}", (x1, y1-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                
                for i, a_type in enumerate(anomaly_types):
                    cv2.putText(annotated, a_type, (x1, y2+20+20*i), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            
            # Check proximity between persons
            for id1, c1 in centers.items():
                for id2, c2 in centers.items():
                    if id1 >= id2 or id1 in invigilators or id2 in invigilators:
                        continue
                    if self.distance(c1, c2) < self.config['proximity_threshold']:
                        self.proximity_counter[(id1,id2)] += 1
                        if self.proximity_counter[(id1,id2)] == self.config['proximity_frames']:
                            self.anomaly_log.append({
                                "frame": frame_id,
                                "person_id": f"{id1}-{id2}",
                                "type": "Suspicious Proximity",
                                "timestamp": f"{frame_id/fps:.2f}",
                                "clip_start": max(0, frame_id - self.config['clip_before_seconds']*fps),
                                "clip_end": min(total_frames-1, frame_id + self.config['clip_after_seconds']*fps),
                                "confidence": 80.0
                            })
                            cv2.imwrite(
                                os.path.join(evidence_dir, f"proximity_{id1}_{id2}_f{frame_id}.jpg"), 
                                annotated
                            )
            
            # Add overlay info
            elapsed_time = time.time() - start_time
            fps_display = frame_id/elapsed_time if elapsed_time>0 else 0
            cv2.putText(annotated, f"FPS: {fps_display:.2f}", (10,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
            cv2.putText(annotated, f"Time: {frame_id/fps:.2f}s", (10,60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
            
            out.write(annotated)
            frame_id += 1
            
            # Progress callback
            if progress_callback and frame_id % 10 == 0:
                progress_callback(frame_id, total_frames, len(self.anomaly_log))
        
        cap.release()
        out.release()
        
        processing_time = time.time() - start_time
        
        # Save anomaly log to CSV
        anomaly_df = pd.DataFrame(self.anomaly_log)
        csv_path = os.path.join(output_dir, "anomaly_log.csv")
        anomaly_df.to_csv(csv_path, index=False)
        
        # Extract clips
        print("\n📹 Extracting anomaly clips...")
        for idx, row in anomaly_df.iterrows():
            clip_filename = f"{row['type'].replace(' ','')}_{row['person_id']}_f{row['frame']}.mp4"
            clip_path = os.path.join(clips_dir, clip_filename)
            self.extract_clip(output_video_path, int(row['clip_start']), int(row['clip_end']), clip_path, fps)
        
        # Prepare results
        results = {
            'video_name': os.path.basename(video_path),
            'total_frames': frame_id,
            'duration': frame_id / fps,
            'processing_time': processing_time,
            'fps': fps,
            'avg_processing_fps': frame_id / processing_time,
            'total_anomalies': len(self.anomaly_log),
            'anomaly_types': anomaly_df['type'].value_counts().to_dict() if len(anomaly_df) > 0 else {},
            'output_video': output_video_path,
            'csv_path': csv_path,
            'evidence_dir': evidence_dir,
            'clips_dir': clips_dir,
            'anomaly_log': self.anomaly_log
        }
        
        print(f"\n✅ Processing complete!")
        print(f"   Frames: {frame_id} | Time: {processing_time:.1f}s | Anomalies: {len(self.anomaly_log)}")
        
        return results


# Convenience function for simple usage
def detect_anomalies(video_path, output_dir, config=None, progress_callback=None):
    """
    Quick function to detect anomalies in a video
    
    Args:
        video_path: Path to input video
        output_dir: Directory for output files
        config: Optional configuration dictionary
        progress_callback: Optional progress callback function
    
    Returns:
        dict: Detection results
    """
    detector = ExamAnomalyDetector(config)
    return detector.process_video(video_path, output_dir, progress_callback)


if __name__ == "__main__":
    # Example usage
    results = detect_anomalies(
        video_path="test_video.mp4",
        output_dir="output",
        config={'blur_faces': True}
    )
    print(f"Detected {results['total_anomalies']} anomalies")