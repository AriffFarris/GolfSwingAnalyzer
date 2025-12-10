import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Tuple, Optional

class SwingAnalyzer:
    def __init__(self):
        """Initialize MediaPipe Pose detection"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def calculate_angle(self, point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
        """
        Calculate angle between three points
        point2 is the vertex of the angle
        """
        vector1 = point1 - point2
        vector2 = point3 - point2
        
        # Calculate angle using dot product
        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def get_landmark_coords(self, landmarks, landmark_id: int, frame_shape) -> np.ndarray:
        """Extract x, y coordinates for a specific landmark"""
        landmark = landmarks.landmark[landmark_id]
        h, w = frame_shape[:2]
        return np.array([landmark.x * w, landmark.y * h])
    
    def analyze_frame(self, frame: np.ndarray, landmarks) -> Dict:
        """Analyze a single frame for swing metrics"""
        if landmarks is None:
            return None
        
        h, w = frame.shape[:2]
        
        # Key landmarks for golf swing analysis
        left_shoulder = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, frame.shape)
        right_shoulder = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value, frame.shape)
        left_elbow = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ELBOW.value, frame.shape)
        right_elbow = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_ELBOW.value, frame.shape)
        left_wrist = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_WRIST.value, frame.shape)
        right_wrist = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_WRIST.value, frame.shape)
        left_hip = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP.value, frame.shape)
        right_hip = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP.value, frame.shape)
        left_knee = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE.value, frame.shape)
        right_knee = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_KNEE.value, frame.shape)
        
        # Calculate key angles
        left_arm_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_knee_angle = self.calculate_angle(left_hip, left_knee, np.array([left_knee[0], h]))
        right_knee_angle = self.calculate_angle(right_hip, right_knee, np.array([right_knee[0], h]))
        
        # Calculate spine angle (using shoulders and hips)
        shoulder_midpoint = (left_shoulder + right_shoulder) / 2
        hip_midpoint = (left_hip + right_hip) / 2
        spine_angle = np.degrees(np.arctan2(
            shoulder_midpoint[0] - hip_midpoint[0],
            shoulder_midpoint[1] - hip_midpoint[1]
        ))
        
        # Calculate shoulder rotation (angle between shoulder line and vertical)
        shoulder_rotation = np.degrees(np.arctan2(
            right_shoulder[1] - left_shoulder[1],
            right_shoulder[0] - left_shoulder[0]
        ))
        
        return {
            'left_arm_angle': left_arm_angle,
            'right_arm_angle': right_arm_angle,
            'left_knee_angle': left_knee_angle,
            'right_knee_angle': right_knee_angle,
            'spine_angle': spine_angle,
            'shoulder_rotation': shoulder_rotation
        }
    
    def generate_feedback(self, metrics: Dict) -> List[Dict[str, str]]:
        """Generate feedback based on swing metrics"""
        feedback = []
        
        # Lead arm analysis (assuming right-handed golfer with left arm as lead)
        if metrics['left_arm_angle'] < 160:
            feedback.append({
                'issue': 'Bent Lead Arm',
                'severity': 'high',
                'description': f"Your lead arm is bending too much ({metrics['left_arm_angle']:.1f}°). Try to keep it straighter throughout the swing for better consistency and power transfer.",
                'tip': 'Focus on maintaining arm extension from setup through impact.'
            })
        
        # Trail arm analysis
        if metrics['right_arm_angle'] > 150 and metrics['right_arm_angle'] < 170:
            feedback.append({
                'issue': 'Trail Arm Position',
                'severity': 'medium',
                'description': f"Your trail arm angle ({metrics['right_arm_angle']:.1f}°) is in good position.",
                'tip': 'Keep this consistent!'
            })
        
        # Knee flex analysis
        if metrics['left_knee_angle'] < 140 or metrics['left_knee_angle'] > 170:
            feedback.append({
                'issue': 'Lead Knee Flex',
                'severity': 'medium',
                'description': f"Your lead knee angle ({metrics['left_knee_angle']:.1f}°) could be adjusted. Ideal range is 140-165°.",
                'tip': 'Maintain stable knee flex to create a solid base for rotation.'
            })
        
        # Spine angle
        if abs(metrics['spine_angle']) < 5:
            feedback.append({
                'issue': 'Spine Angle',
                'severity': 'high',
                'description': f"Your spine is too upright ({abs(metrics['spine_angle']):.1f}°). You need more forward tilt.",
                'tip': 'Tilt from your hips, not your back. Aim for 25-35° of forward spine angle at address.'
            })
        elif abs(metrics['spine_angle']) > 45:
            feedback.append({
                'issue': 'Spine Angle',
                'severity': 'medium',
                'description': f"Your spine tilt ({abs(metrics['spine_angle']):.1f}°) might be too extreme.",
                'tip': 'Reduce forward bend slightly for better balance and rotation.'
            })
        
        # Shoulder rotation
        if abs(metrics['shoulder_rotation']) < 10:
            feedback.append({
                'issue': 'Shoulder Rotation',
                'severity': 'high',
                'description': f"Limited shoulder rotation detected ({abs(metrics['shoulder_rotation']):.1f}°).",
                'tip': 'Focus on turning your shoulders fully in the backswing. Aim for 90° of shoulder turn.'
            })
        
        if not feedback:
            feedback.append({
                'issue': 'Good Form',
                'severity': 'low',
                'description': 'Your swing mechanics look solid in this frame!',
                'tip': 'Keep up the good work and maintain consistency.'
            })
        
        return feedback
    
    def process_video(self, video_path: str) -> Tuple[List[np.ndarray], List[Dict], List[List[Dict]]]:
        """
        Process entire video and return frames with pose overlay, metrics, and feedback
        Returns: (annotated_frames, metrics_per_frame, feedback_per_frame)
        """
        cap = cv2.VideoCapture(video_path)
        annotated_frames = []
        all_metrics = []
        all_feedback = []
        
        frame_count = 0
        sample_rate = 3  # Process every 3rd frame for efficiency
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % sample_rate != 0:
                continue
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.pose.process(rgb_frame)
            
            # Draw pose landmarks
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
                # Analyze frame
                metrics = self.analyze_frame(frame, results.pose_landmarks)
                if metrics:
                    all_metrics.append(metrics)
                    feedback = self.generate_feedback(metrics)
                    all_feedback.append(feedback)
            
            annotated_frames.append(frame)
        
        cap.release()
        return annotated_frames, all_metrics, all_feedback
    
    def get_summary_feedback(self, all_metrics: List[Dict], all_feedback: List[List[Dict]]) -> List[Dict]:
        """Generate overall summary feedback from all frames"""
        if not all_metrics:
            return [{'issue': 'No Data', 'severity': 'high', 
                    'description': 'Unable to detect pose in video.', 
                    'tip': 'Ensure good lighting and full body is visible.'}]
        
        # Average metrics across all frames
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        
        # Count issue frequency
        issue_counts = {}
        for frame_feedback in all_feedback:
            for fb in frame_feedback:
                issue = fb['issue']
                if issue not in issue_counts:
                    issue_counts[issue] = 0
                issue_counts[issue] += 1
        
        # Generate summary based on averaged metrics
        summary = self.generate_feedback(avg_metrics)
        
        # Add frequency information
        for fb in summary:
            issue = fb['issue']
            if issue in issue_counts:
                frequency = (issue_counts[issue] / len(all_feedback)) * 100
                fb['frequency'] = f"Occurred in {frequency:.0f}% of frames"
        
        return summary
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'pose'):
            self.pose.close()