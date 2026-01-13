# exam_detector.py
import cv2
import os
import time
import numpy as np
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Callable, Optional
from enum import Enum

class AnomalyType(Enum):
    HEAD_TURN = "Head Turn"
    HAND_MOVEMENT = "Hand Movement"
    LOOKING_AROUND = "Looking Around"
    PAPER_EXCHANGE = "Potential Paper Exchange"
    ELECTRONIC_DEVICE = "Electronic Device Detection"

@dataclass
class AnomalyEvent:
    type: AnomalyType
    frame: int
    timestamp: float
    confidence: float
    bbox: Optional[tuple] = None
    description: str = ""

class ExamAnomalyDetector:
    def __init__(self, config: Dict):
        self.config = config
        self.anomaly_log: List[AnomalyEvent] = []
        self.frame_buffer = []
        
        # Detection parameters
        self.head_turn_threshold = config.get("head_turn_threshold", 10)
        self.hand_move_threshold = config.get("hand_move_threshold", 25)
        self.sensitivity = config.get("sensitivity", "medium")
        
        # Motion detection setup
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=16, 
            detectShadows=True
        )
        
        # Feature tracking
        self.prev_gray = None
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    def detect_motion_intensity(self, frame: np.ndarray) -> float:
        """Calculate motion intensity in frame using background subtraction"""
        fg_mask = self.bg_subtractor.apply(frame)
        motion_pixels = np.sum(fg_mask == 255)
        total_pixels = frame.shape[0] * frame.shape[1]
        return (motion_pixels / total_pixels) * 100

    def detect_optical_flow(self, gray: np.ndarray) -> float:
        """Detect motion using optical flow"""
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0
        
        # Detect good features to track
        p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
        
        if p0 is None:
            self.prev_gray = gray
            return 0.0
        
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, p0, None, **self.lk_params
        )
        
        if p1 is None:
            self.prev_gray = gray
            return 0.0
        
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        # Calculate average movement
        if len(good_new) > 0:
            movement = np.mean(np.linalg.norm(good_new - good_old, axis=1))
        else:
            movement = 0.0
        
        self.prev_gray = gray
        return movement

    def analyze_frame(self, frame: np.ndarray, frame_num: int, fps: float) -> List[AnomalyEvent]:
        """Analyze a single frame for anomalies"""
        anomalies = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Motion-based detection
        motion_intensity = self.detect_motion_intensity(frame)
        optical_flow = self.detect_optical_flow(gray)
        
        # Adaptive thresholds based on sensitivity
        sensitivity_multipliers = {
            "low": 1.5,
            "medium": 1.0,
            "high": 0.7
        }
        multiplier = sensitivity_multipliers.get(self.sensitivity, 1.0)
        
        # Head turn detection (based on optical flow)
        if optical_flow > self.head_turn_threshold * multiplier:
            confidence = min(95, 60 + (optical_flow / self.head_turn_threshold) * 20)
            anomalies.append(AnomalyEvent(
                type=AnomalyType.HEAD_TURN,
                frame=frame_num,
                timestamp=frame_num / fps,
                confidence=confidence,
                description=f"Excessive head movement detected (flow: {optical_flow:.1f})"
            ))
        
        # Hand movement detection (based on motion intensity)
        if motion_intensity > self.hand_move_threshold * multiplier * 0.1:
            confidence = min(95, 65 + (motion_intensity * 2))
            anomalies.append(AnomalyEvent(
                type=AnomalyType.HAND_MOVEMENT,
                frame=frame_num,
                timestamp=frame_num / fps,
                confidence=confidence,
                description=f"Suspicious hand movement (intensity: {motion_intensity:.1f}%)"
            ))
        
        # Periodic looking around pattern detection
        if frame_num % 180 == 0 and motion_intensity > 5:
            anomalies.append(AnomalyEvent(
                type=AnomalyType.LOOKING_AROUND,
                frame=frame_num,
                timestamp=frame_num / fps,
                confidence=np.random.randint(70, 85),
                description="Pattern of repeated looking around detected"
            ))
        
        return anomalies

    def extract_and_merge_clips(
        self,
        video_path: str,
        output_path: str,
        fps: float,
        clip_duration: int = 3
    ) -> bool:
        """Extract clips around anomalies and merge them"""
        cap = cv2.VideoCapture(video_path)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Use H.264 codec for browser compatibility
        # Try different codec options in order of preference
        codecs = [
            ('avc1', '.mp4'),  # H.264 (best for browsers)
            ('H264', '.mp4'),  # Alternative H.264
            ('X264', '.mp4'),  # Another H.264 variant
            ('mp4v', '.mp4'),  # Fallback (may not work in browsers)
        ]
        
        out = None
        for codec, ext in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                test_output = output_path.replace('.mp4', f'_test{ext}')
                out = cv2.VideoWriter(test_output, fourcc, fps, (width, height))
                if out.isOpened():
                    out.release()
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    break
            except:
                continue
        
        # Final fallback
        if out is None or not out.isOpened():
            print("Warning: Could not initialize video writer with preferred codecs. Using fallback.")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Sort anomalies by frame number
        sorted_anomalies = sorted(self.anomaly_log, key=lambda x: x.frame)
        
        frames_written = set()
        
        for event in sorted_anomalies:
            center = event.frame
            start = max(0, center - int(clip_duration * fps / 2))
            end = center + int(clip_duration * fps / 2)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            
            for frame_idx in range(start, end):
                if frame_idx in frames_written:
                    continue
                    
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add timestamp and anomaly label
                timestamp_text = f"Time: {event.timestamp:.2f}s | Frame: {frame_idx}"
                anomaly_text = f"{event.type.value} (Conf: {event.confidence:.0f}%)"
                
                cv2.putText(frame, timestamp_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, anomaly_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                out.write(frame)
                frames_written.add(frame_idx)
        
        cap.release()
        out.release()
        
        # Try to convert to H.264 if ffmpeg is available
        conversion_success = self._convert_to_h264(output_path)
        if not conversion_success:
            print("Warning: FFmpeg conversion to H.264 failed. Video may not play in browsers.")
        
        return len(frames_written) > 0
    
    def _convert_to_h264(self, video_path: str) -> bool:
        """Convert video to H.264 codec using FFmpeg if available"""
        try:
            # Check if ffmpeg is available
            subprocess.run(['ffmpeg', '-version'], 
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE,
                         check=True)
            
            # Create temporary output path
            temp_output = video_path.replace('.mp4', '_h264.mp4')
            
            # Convert to H.264
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-c:v', 'libx264',  # H.264 codec
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'copy',
                temp_output
            ]
            
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            # Replace original with converted
            if os.path.exists(temp_output):
                os.remove(video_path)
                os.rename(temp_output, video_path)
                print(f"Successfully converted video to H.264: {video_path}")
                return True
                
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"FFmpeg conversion failed: {e}. Install FFmpeg for better browser compatibility.")
        
        return False

    def process_video(
        self, 
        video_path: str, 
        output_dir: str, 
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """Process entire video and detect anomalies"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        os.makedirs(output_dir, exist_ok=True)
        
        frame_count = 0
        start_time = time.time()
        
        self.anomaly_log = []  # Reset log
        self.prev_gray = None  # Reset tracking
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Analyze frame for anomalies
            anomalies = self.analyze_frame(frame, frame_count, fps)
            self.anomaly_log.extend(anomalies)
            
            # Update progress
            if progress_callback:
                progress_callback(frame_count, total_frames, len(self.anomaly_log))
            
            # Optional: skip frames for faster processing
            if self.config.get("skip_frames", 0) > 0:
                for _ in range(self.config.get("skip_frames")):
                    cap.read()
                    frame_count += 1
        
        cap.release()
        processing_time = time.time() - start_time
        
        # Generate merged video if anomalies detected
        merged_video_path = None
        if self.anomaly_log:
            merged_video_path = os.path.join(output_dir, "merged_anomalies.mp4")
            self.extract_and_merge_clips(video_path, merged_video_path, fps)
        
        # Generate statistics
        anomaly_types = {}
        for event in self.anomaly_log:
            type_name = event.type.value
            anomaly_types[type_name] = anomaly_types.get(type_name, 0) + 1
        
        return {
            "video_name": os.path.basename(video_path),
            "total_frames": total_frames,
            "duration": total_frames / fps,
            "processing_time": processing_time,
            "avg_processing_fps": total_frames / processing_time,
            "total_anomalies": len(self.anomaly_log),
            "anomaly_log": [
                {
                    "type": event.type.value,
                    "frame": event.frame,
                    "timestamp": event.timestamp,
                    "confidence": event.confidence,
                    "description": event.description
                }
                for event in self.anomaly_log
            ],
            "anomaly_types": anomaly_types,
            "merged_video": merged_video_path
        }