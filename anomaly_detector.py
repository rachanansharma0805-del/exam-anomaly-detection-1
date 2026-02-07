"""
Anomaly Detection Module - Production Ready with PATH-BASED Invigilator Detection
Key Feature: Person who WALKS and STANDS = Invigilator
Uses actual path distance walked, not just position range
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque, defaultdict
from pose_estimator import PoseEstimator


class AnomalyDetector:
    """Production-ready anomaly detector with path-based invigilator detection"""
    
    def __init__(self, pose_estimator: PoseEstimator):
        self.pose_estimator = pose_estimator
        
        # Minimum confidence for anomaly detection (95%)
        self.min_anomaly_confidence = 0.95
        
        # Thresholds for anomaly detection
        self.head_turn_threshold = 45
        self.hand_raise_threshold = 80
        self.looking_away_threshold = 35
        self.mouth_movement_threshold = 30
        self.body_turn_threshold = 30
        self.movement_threshold = 70
        
        # History tracking
        self.pose_history = {}  # person_id -> deque of poses
        self.history_length = 30
        
        # PATH-BASED INVIGILATOR DETECTION (WALKING + STANDING = INVIGILATOR)
        self.invigilator_id = None
        self.person_position_history = {}  # person_id -> deque of (x, y)
        self.person_standing_duration = defaultdict(int)
        self.person_stable_position = {}  # person_id -> is_seated
        
        # Invigilator thresholds
        self.INVIGILATOR_PATH_LENGTH = 250  # Actual distance walked
        self.MOVEMENT_THRESHOLD = 150  # Range of movement
        self.STANDING_THRESHOLD = 40  # Standing frames
        self.SEATED_STABILITY_THRESHOLD = 60  # Max movement for seated
        self.DETECTION_FRAMES = 60  # Frames to confirm
        
        # Anomaly tracking
        self.person_anomaly_counts = defaultdict(int)
        self.person_anomaly_history = defaultdict(list)
        self.excessive_anomaly_threshold = 15
        
        # Temporal smoothing
        self.anomaly_persistence = defaultdict(lambda: defaultdict(int))
        self.persistence_threshold = 3
    
    def calculate_path_length(self, positions: List[Tuple[int, int]]) -> float:
        """
        Calculate ACTUAL distance walked (not just position range)
        This is the key to detecting walking vs sitting
        """
        if len(positions) < 2:
            return 0
        
        total_distance = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            distance = np.sqrt(dx * dx + dy * dy)
            total_distance += distance
        
        return total_distance
    
    def detect_invigilator(self, person_id: int, landmarks: Dict, 
                          bbox: Tuple[int, int, int, int],
                          frame_count: int) -> bool:
        """
        IMPROVED PATH-BASED INVIGILATOR DETECTION
        
        Key principle: Person who WALKS and STANDS = Invigilator
        
        Uses 4 criteria (need 3 out of 4):
        1. Path length - actual distance walked (MOST RELIABLE)
        2. Movement range - spatial coverage
        3. Standing duration - time spent upright
        4. NOT seated - not stable/stationary
        """
        # If already confirmed, return result
        if self.invigilator_id is not None:
            return person_id == self.invigilator_id
        
        # Get center position from bbox
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Initialize position history
        if person_id not in self.person_position_history:
            self.person_position_history[person_id] = deque(maxlen=120)  # 4 seconds at 30fps
        
        # Store current position
        self.person_position_history[person_id].append((center_x, center_y))
        
        # Need minimum frames to evaluate
        if len(self.person_position_history[person_id]) < self.DETECTION_FRAMES:
            return False
        
        positions = list(self.person_position_history[person_id])
        
        # ===== CRITERION 1: ACTUAL PATH WALKED (Most Reliable) =====
        path_length = self.calculate_path_length(positions)
        
        # ===== CRITERION 2: CHECK IF PERSON IS SEATED/STABLE =====
        # Look at last 90 frames (3 seconds)
        if person_id not in self.person_stable_position:
            if len(positions) >= 90:
                recent_positions = positions[-90:]
                xs = [p[0] for p in recent_positions]
                ys = [p[1] for p in recent_positions]
                
                # Calculate total movement in recent window
                x_movement = max(xs) - min(xs)
                y_movement = max(ys) - min(ys)
                recent_movement = x_movement + y_movement
                
                # If very low movement, mark as seated/stable
                if recent_movement < self.SEATED_STABILITY_THRESHOLD:
                    self.person_stable_position[person_id] = True
                    print(f"  Person ID {person_id}: Marked as SEATED (stable position)")
        
        # If person is seated/stable, they CANNOT be invigilator
        if self.person_stable_position.get(person_id, False):
            return False
        
        # ===== CRITERION 3: MOVEMENT RANGE =====
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        total_range = x_range + y_range
        
        # ===== CRITERION 4: ANALYZE WALKING PATTERN =====
        # Check if person is walking (moving horizontally) or just standing in place
        recent_60 = positions[-60:] if len(positions) >= 60 else positions
        y_positions = [p[1] for p in recent_60]
        y_variance = np.var(y_positions)
        
        # Low Y variance + horizontal movement = walking horizontally
        # High Y variance = walking vertically or general movement
        is_walking = (y_variance < 800 and x_range > self.MOVEMENT_THRESHOLD) or (y_variance > 1500)
        
        # Update standing duration
        if is_walking:
            self.person_standing_duration[person_id] += 1
        else:
            # Decay if not walking
            self.person_standing_duration[person_id] = max(
                0, 
                self.person_standing_duration[person_id] - 3
            )
        
        # ===== EVALUATE CRITERIA (Need 3 out of 4) =====
        criteria_met = 0
        confidences = []
        
        # 1. Path length (MOST RELIABLE INDICATOR)
        if path_length > self.INVIGILATOR_PATH_LENGTH:
            criteria_met += 1
            confidence = min(1.0, path_length / (self.INVIGILATOR_PATH_LENGTH * 2))
            confidences.append(confidence)
        
        # 2. Movement range
        if total_range > self.MOVEMENT_THRESHOLD:
            criteria_met += 1
            confidence = min(1.0, total_range / (self.MOVEMENT_THRESHOLD * 2))
            confidences.append(confidence)
        
        # 3. Standing/walking duration
        if self.person_standing_duration[person_id] > self.STANDING_THRESHOLD:
            criteria_met += 1
            confidence = min(1.0, self.person_standing_duration[person_id] / (self.STANDING_THRESHOLD * 1.5))
            confidences.append(confidence)
        
        # 4. NOT seated/stable
        if not self.person_stable_position.get(person_id, False):
            criteria_met += 1
            confidences.append(0.7)
        
        # ===== DECISION: Need 3 out of 4 criteria =====
        is_invigilator = criteria_met >= 3
        overall_confidence = np.mean(confidences) if confidences else 0.0
        
        # Confirm as THE invigilator
        if is_invigilator and overall_confidence > 0.7:
            self.invigilator_id = person_id
            print(f"\n{'='*60}")
            print(f"✓ INVIGILATOR DETECTED: Person ID {person_id}")
            print(f"{'='*60}")
            print(f"  Path walked: {path_length:.1f} pixels")
            print(f"  Movement range: {total_range:.1f} pixels")
            print(f"  Standing duration: {self.person_standing_duration[person_id]} frames")
            print(f"  Confidence: {overall_confidence:.2%}")
            print(f"  Criteria met: {criteria_met}/4")
            print(f"{'='*60}\n")
            return True
        
        return False
    
    def detect_anomalies(self, person_id: int, pose_data: Optional[Dict], 
                        bbox: Tuple[int, int, int, int],
                        frame_number: int) -> List[Dict]:
        """Detect anomalies (ONLY for students, NOT invigilator)"""
        
        # CRITICAL: Skip anomaly detection for the invigilator
        if person_id == self.invigilator_id:
            return []
        
        anomalies = []
        
        if not pose_data or 'landmarks' not in pose_data:
            return anomalies
        
        landmarks = pose_data['landmarks']
        
        # Initialize history
        if person_id not in self.pose_history:
            self.pose_history[person_id] = deque(maxlen=self.history_length)
        
        self.pose_history[person_id].append(landmarks)
        
        # Detect various anomalies
        potential_anomalies = []
        potential_anomalies.extend(self._detect_head_turn(landmarks))
        potential_anomalies.extend(self._detect_hand_raise(landmarks))
        potential_anomalies.extend(self._detect_looking_away(landmarks))
        potential_anomalies.extend(self._detect_speaking(landmarks))
        potential_anomalies.extend(self._detect_turning(person_id, landmarks, bbox))
        potential_anomalies.extend(self._detect_excessive_movement(person_id, landmarks))
        potential_anomalies.extend(self._detect_peeping(person_id, landmarks, bbox))
        
        # Apply temporal smoothing
        for anomaly in potential_anomalies:
            anomaly_type = anomaly['type']
            self.anomaly_persistence[person_id][anomaly_type] += 1
            
            # Only report if persists and has high confidence
            if (self.anomaly_persistence[person_id][anomaly_type] >= self.persistence_threshold and
                anomaly.get('confidence', 0) >= self.min_anomaly_confidence):
                
                anomalies.append(anomaly)
                self.person_anomaly_counts[person_id] += 1
                self.person_anomaly_history[person_id].append({
                    'frame': frame_number,
                    'type': anomaly_type,
                    'severity': anomaly['severity']
                })
        
        # Reset persistence for undetected anomalies
        detected_types = {a['type'] for a in potential_anomalies}
        for anomaly_type in list(self.anomaly_persistence[person_id].keys()):
            if anomaly_type not in detected_types:
                self.anomaly_persistence[person_id][anomaly_type] = 0
        
        # Check for excessive anomalies
        if self.person_anomaly_counts[person_id] >= self.excessive_anomaly_threshold:
            for anomaly in anomalies:
                anomaly['excessive_violations'] = True
                anomaly['total_violations'] = self.person_anomaly_counts[person_id]
        
        return anomalies
    
    def _detect_head_turn(self, landmarks: Dict) -> List[Dict]:
        """Detect significant head turns (potential copying)"""
        anomalies = []
        
        left_ear = self.pose_estimator.get_landmark_by_name(landmarks, 'left_ear')
        right_ear = self.pose_estimator.get_landmark_by_name(landmarks, 'right_ear')
        nose = self.pose_estimator.get_landmark_by_name(landmarks, 'nose')
        
        if not all([left_ear, right_ear, nose]):
            return anomalies
        
        if left_ear['visibility'] > 0.6 and right_ear['visibility'] > 0.6:
            ear_midpoint = (left_ear['x'] + right_ear['x']) / 2
            nose_offset = abs(nose['x'] - ear_midpoint)
            ear_distance = abs(left_ear['x'] - right_ear['x'])
            
            if ear_distance > 0:
                turn_ratio = nose_offset / ear_distance
                
                if turn_ratio > 0.5:
                    confidence = min(left_ear['visibility'], right_ear['visibility'], nose['visibility'])
                    
                    if confidence >= self.min_anomaly_confidence:
                        severity = min(turn_ratio * 120, 100)
                        anomalies.append({
                            'type': 'head_turn',
                            'severity': severity,
                            'description': 'Suspicious head turn - possible copying',
                            'confidence': confidence
                        })
        
        return anomalies
    
    def _detect_hand_raise(self, landmarks: Dict) -> List[Dict]:
        """Detect hand raises (signaling or suspicious activity)"""
        anomalies = []
        
        for side in ['left', 'right']:
            shoulder = self.pose_estimator.get_landmark_by_name(landmarks, f'{side}_shoulder')
            wrist = self.pose_estimator.get_landmark_by_name(landmarks, f'{side}_wrist')
            elbow = self.pose_estimator.get_landmark_by_name(landmarks, f'{side}_elbow')
            
            if not all([shoulder, wrist, elbow]):
                continue
            
            if wrist['y'] < shoulder['y'] - self.hand_raise_threshold:
                raise_amount = shoulder['y'] - wrist['y']
                confidence = min(wrist['visibility'], shoulder['visibility'], elbow['visibility'])
                
                if confidence >= self.min_anomaly_confidence:
                    anomalies.append({
                        'type': 'hand_raise',
                        'side': side,
                        'severity': min((raise_amount / 150) * 100, 100),
                        'description': f'{side.capitalize()} hand raised - signaling or passing objects',
                        'confidence': confidence
                    })
        
        return anomalies
    
    def _detect_looking_away(self, landmarks: Dict) -> List[Dict]:
        """Detect looking away from exam paper"""
        anomalies = []
        
        nose = self.pose_estimator.get_landmark_by_name(landmarks, 'nose')
        left_eye = self.pose_estimator.get_landmark_by_name(landmarks, 'left_eye')
        right_eye = self.pose_estimator.get_landmark_by_name(landmarks, 'right_eye')
        
        if not all([nose, left_eye, right_eye]):
            return anomalies
        
        eye_center_x = (left_eye['x'] + right_eye['x']) / 2
        nose_offset = abs(nose['x'] - eye_center_x)
        eye_distance = abs(left_eye['x'] - right_eye['x'])
        
        if eye_distance > 0:
            offset_ratio = nose_offset / eye_distance
            
            if offset_ratio > 0.6:
                confidence = min(nose['visibility'], left_eye['visibility'], right_eye['visibility'])
                
                if confidence >= self.min_anomaly_confidence:
                    anomalies.append({
                        'type': 'looking_away',
                        'severity': min(offset_ratio * 100, 100),
                        'description': 'Looking away from paper - potential cheating',
                        'confidence': confidence
                    })
        
        return anomalies
    
    def _detect_speaking(self, landmarks: Dict) -> List[Dict]:
        """Detect speaking/communication"""
        anomalies = []
        
        mouth_left = self.pose_estimator.get_landmark_by_name(landmarks, 'mouth_left')
        mouth_right = self.pose_estimator.get_landmark_by_name(landmarks, 'mouth_right')
        
        if not all([mouth_left, mouth_right]):
            return anomalies
        
        mouth_width = abs(mouth_left['x'] - mouth_right['x'])
        
        if (mouth_left['visibility'] > 0.95 and mouth_right['visibility'] > 0.95 and
            mouth_width > self.mouth_movement_threshold):
            
            confidence = min(mouth_left['visibility'], mouth_right['visibility'])
            anomalies.append({
                'type': 'speaking',
                'severity': 70,
                'description': 'Possible communication with other students',
                'confidence': confidence
            })
        
        return anomalies
    
    def _detect_turning(self, person_id: int, landmarks: Dict, 
                       bbox: Tuple[int, int, int, int]) -> List[Dict]:
        """Detect body turning"""
        anomalies = []
        
        left_shoulder = self.pose_estimator.get_landmark_by_name(landmarks, 'left_shoulder')
        right_shoulder = self.pose_estimator.get_landmark_by_name(landmarks, 'right_shoulder')
        left_hip = self.pose_estimator.get_landmark_by_name(landmarks, 'left_hip')
        right_hip = self.pose_estimator.get_landmark_by_name(landmarks, 'right_hip')
        
        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return anomalies
        
        shoulder_angle = np.arctan2(
            right_shoulder['y'] - left_shoulder['y'],
            right_shoulder['x'] - left_shoulder['x']
        )
        
        hip_angle = np.arctan2(
            right_hip['y'] - left_hip['y'],
            right_hip['x'] - left_hip['x']
        )
        
        angle_diff = abs(np.degrees(shoulder_angle - hip_angle))
        
        if angle_diff > self.body_turn_threshold:
            confidence = min(
                left_shoulder['visibility'], right_shoulder['visibility'],
                left_hip['visibility'], right_hip['visibility']
            )
            
            if confidence >= self.min_anomaly_confidence:
                anomalies.append({
                    'type': 'body_turn',
                    'severity': min((angle_diff / 50) * 100, 100),
                    'description': 'Body turned - possible attempt to view neighbor\'s paper',
                    'confidence': confidence
                })
        
        return anomalies
    
    def _detect_excessive_movement(self, person_id: int, landmarks: Dict) -> List[Dict]:
        """Detect excessive movement"""
        anomalies = []
        
        if person_id not in self.pose_history or len(self.pose_history[person_id]) < 15:
            return anomalies
        
        current_nose = self.pose_estimator.get_landmark_by_name(landmarks, 'nose')
        if not current_nose or current_nose['visibility'] < 0.95:
            return anomalies
        
        history = list(self.pose_history[person_id])
        
        if len(history) >= 15:
            old_landmarks = history[-15]
            old_nose = self.pose_estimator.get_landmark_by_name(old_landmarks, 'nose')
            
            if old_nose and old_nose['visibility'] >= 0.95:
                movement = np.sqrt(
                    (current_nose['x'] - old_nose['x'])**2 + 
                    (current_nose['y'] - old_nose['y'])**2
                )
                
                if movement > self.movement_threshold:
                    confidence = min(current_nose['visibility'], old_nose['visibility'])
                    anomalies.append({
                        'type': 'excessive_movement',
                        'severity': min((movement / 100) * 100, 100),
                        'description': 'Excessive movement detected',
                        'confidence': confidence
                    })
        
        return anomalies
    
    def _detect_peeping(self, person_id: int, landmarks: Dict, 
                       bbox: Tuple[int, int, int, int]) -> List[Dict]:
        """Detect peeping behavior"""
        anomalies = []
        
        nose = self.pose_estimator.get_landmark_by_name(landmarks, 'nose')
        left_shoulder = self.pose_estimator.get_landmark_by_name(landmarks, 'left_shoulder')
        right_shoulder = self.pose_estimator.get_landmark_by_name(landmarks, 'right_shoulder')
        
        if not all([nose, left_shoulder, right_shoulder]):
            return anomalies
        
        shoulder_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
        lean_offset = abs(nose['x'] - shoulder_center_x)
        shoulder_width = abs(left_shoulder['x'] - right_shoulder['x'])
        
        if shoulder_width > 0:
            lean_ratio = lean_offset / shoulder_width
            
            if lean_ratio > 0.7:
                confidence = min(
                    nose['visibility'],
                    left_shoulder['visibility'],
                    right_shoulder['visibility']
                )
                
                if confidence >= self.min_anomaly_confidence:
                    anomalies.append({
                        'type': 'peeping',
                        'severity': min(lean_ratio * 100, 100),
                        'description': 'Leaning to side - possible attempt to view neighbor\'s paper',
                        'confidence': confidence
                    })
        
        return anomalies
    
    def get_person_anomaly_summary(self, person_id: int) -> Dict:
        """Get anomaly summary for a specific person"""
        return {
            'total_anomalies': self.person_anomaly_counts.get(person_id, 0),
            'anomaly_history': self.person_anomaly_history.get(person_id, []),
            'excessive_violations': self.person_anomaly_counts.get(person_id, 0) >= self.excessive_anomaly_threshold
        }