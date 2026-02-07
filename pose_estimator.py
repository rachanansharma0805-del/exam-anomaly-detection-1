"""
Pose Estimation Module
Uses MediaPipe to detect body landmarks and analyze poses
Includes full body landmarks for better invigilator detection
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Optional, Tuple, List


class PoseEstimator:
    """Handles pose estimation and body landmark detection"""
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def estimate_pose(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[Dict]:
        """
        Estimate pose for a person in a bounding box
        
        Args:
            frame: Full video frame
            bbox: Bounding box (x, y, w, h) of person
            
        Returns:
            Dictionary containing landmarks and pose information
        """
        x, y, w, h = bbox
        
        # Add padding to bbox
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        # Extract person region
        person_roi = frame[y1:y2, x1:x2]
        
        if person_roi.size == 0:
            return None
        
        # Convert to RGB for MediaPipe
        rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb_roi)
        
        if not results.pose_landmarks:
            return None
        
        # Extract landmarks
        landmarks = {}
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            # Convert normalized coordinates to absolute coordinates in original frame
            abs_x = int(landmark.x * person_roi.shape[1] + x1)
            abs_y = int(landmark.y * person_roi.shape[0] + y1)
            
            landmarks[idx] = {
                'x': abs_x,
                'y': abs_y,
                'z': landmark.z,
                'visibility': landmark.visibility
            }
        
        return {
            'landmarks': landmarks,
            'pose_landmarks': results.pose_landmarks,
            'roi_offset': (x1, y1)
        }
    
    def get_landmark_by_name(self, landmarks: Dict, name: str) -> Optional[Dict]:
        """Get landmark by anatomical name - includes full body landmarks"""
        landmark_map = {
            'nose': 0,
            'left_eye_inner': 1,
            'left_eye': 2,
            'left_eye_outer': 3,
            'right_eye_inner': 4,
            'right_eye': 5,
            'right_eye_outer': 6,
            'left_ear': 7,
            'right_ear': 8,
            'mouth_left': 9,
            'mouth_right': 10,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_pinky': 17,
            'right_pinky': 18,
            'left_index': 19,
            'right_index': 20,
            'left_thumb': 21,
            'right_thumb': 22,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
            'left_heel': 29,
            'right_heel': 30,
            'left_foot_index': 31,
            'right_foot_index': 32,
        }
        
        if name in landmark_map:
            idx = landmark_map[name]
            return landmarks.get(idx)
        return None
    
    def calculate_angle(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """
        Calculate angle between three points
        
        Args:
            p1, p2, p3: Points with 'x' and 'y' keys
            
        Returns:
            Angle in degrees
        """
        a = np.array([p1['x'], p1['y']])
        b = np.array([p2['x'], p2['y']])
        c = np.array([p3['x'], p3['y']])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def draw_pose(self, frame: np.ndarray, pose_data: Dict) -> np.ndarray:
        """Draw pose landmarks on frame"""
        if pose_data and 'pose_landmarks' in pose_data:
            # Create a copy of the ROI
            x_offset, y_offset = pose_data['roi_offset']
            
            # Draw on full frame
            annotated_frame = frame.copy()
            
            # Draw landmarks
            for idx, landmark_data in pose_data['landmarks'].items():
                x, y = landmark_data['x'], landmark_data['y']
                cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)
            
            return annotated_frame
        
        return frame
    
    def cleanup(self):
        """Release resources"""
        self.pose.close()