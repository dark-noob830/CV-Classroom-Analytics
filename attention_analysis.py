#!/usr/bin/env python3
"""
Student Attention & Engagement Tracking System
=============================================

Advanced attention analysis system using MediaPipe for:
- Eye gaze tracking and direction detection
- Head pose estimation (pitch, yaw, roll)
- Attention level measurement and zone detection
- Behavioral pattern analysis over time
"""

import cv2
import numpy as np
import mediapipe as mp
import torch
from facenet_pytorch import MTCNN
import argparse
import os
from collections import defaultdict, deque
import time
import math
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
import json
from datetime import datetime

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ================== Data Classes ==================
@dataclass
class HeadPose:
    """Class for storing head angle information"""
    pitch: float  # up/down
    yaw: float    # left/right
    roll: float   # rotation

@dataclass
class GazeDirection:
    """Class for storing gaze direction"""
    x: float
    y: float
    direction: str  # 'center', 'left', 'right', 'up', 'down'

@dataclass
class AttentionState:
    """Class for storing attention state"""
    level: float  # 0-1 (0=no attention, 1=full attention)
    target: str   # 'instructor', 'laptop', 'distracted', 'peer'
    confidence: float

# ================== Settings ==================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'🖥️  Running on device: {device}')

# Attention zones (as percentage of frame width/height)
ATTENTION_ZONES = {
    'instructor': {'x': (0.3, 0.7), 'y': (0.2, 0.5)},  # center-top (instructor area)
    'laptop': {'x': (0.3, 0.7), 'y': (0.6, 0.9)},      # center-bottom (laptop area)
    'left': {'x': (0.0, 0.3), 'y': (0.0, 1.0)},        # left side
    'right': {'x': (0.7, 1.0), 'y': (0.0, 1.0)},       # right side
}

# Colors for displaying attention levels
ATTENTION_COLORS = {
    'high': (0, 255, 0),      # Green - high attention
    'medium': (0, 255, 255),  # Yellow - medium attention
    'low': (0, 0, 255),       # Red - low attention
}

# Attention thresholds
ATTENTION_THRESHOLDS = {
    'high': 0.7,
    'medium': 0.4,
    'low': 0.0
}

class AttentionAnalyzer:
    """Main class for attention analysis"""
    
    def __init__(self):
        """Initialize the analyzer"""
        # Face detection
        self.mtcnn = MTCNN(keep_all=True, device=device, min_face_size=40)
        
        # MediaPipe Face Mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Tracking data
        self.attention_history = defaultdict(lambda: deque(maxlen=30))
        self.gaze_history = defaultdict(lambda: deque(maxlen=30))
        self.head_pose_history = defaultdict(lambda: deque(maxlen=30))
        
        # Statistics
        self.attention_stats = defaultdict(lambda: defaultdict(float))
        self.engagement_timeline = []
        
    def calculate_head_pose(self, landmarks, img_h, img_w) -> Optional[HeadPose]:
        """Calculate head pose from landmarks"""
        try:
            # Key points for pose estimation
            # Based on MediaPipe face landmarks
            face_3d = []
            face_2d = []
            
            # Important points: nose, chin, eye corners, forehead
            key_points = [1, 33, 61, 199, 291, 263]  # MediaPipe landmark indices
            
            for idx in key_points:
                lm = landmarks.landmark[idx]
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z * 3000])  # Scale z coordinate
            
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            
            # Camera matrix (assumed - for standard camera)
            focal_length = img_w
            center = (img_w / 2, img_h / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype=np.float64
            )
            
            # Distortion coefficients (assumed zero)
            dist_coeffs = np.zeros((4, 1))
            
            # Solve PnP
            success, rotation_vec, translation_vec = cv2.solvePnP(
                face_3d, face_2d, camera_matrix, dist_coeffs
            )
            
            if success:
                # Convert rotation vector to angles
                rotation_mat, _ = cv2.Rodrigues(rotation_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)
                
                # Convert to degrees
                pitch = angles[0] * 1
                yaw = angles[1] * 1
                roll = angles[2] * 1
                
                return HeadPose(pitch=pitch, yaw=yaw, roll=roll)
            
        except Exception as e:
            print(f"Error calculating head pose: {e}")
        
        return None
    
    def estimate_gaze_direction(self, landmarks, img_h, img_w) -> Optional[GazeDirection]:
        """Estimate gaze direction from eye landmarks"""
        try:
            # Eye centers
            # Left eye landmarks
            left_eye_indices = [33, 133, 157, 158, 159, 160, 161, 163]
            right_eye_indices = [362, 263, 387, 388, 389, 390, 391, 393]
            
            # Calculate center of each eye
            left_eye_center = np.mean([[landmarks.landmark[i].x, landmarks.landmark[i].y] 
                                      for i in left_eye_indices], axis=0)
            right_eye_center = np.mean([[landmarks.landmark[i].x, landmarks.landmark[i].y] 
                                       for i in right_eye_indices], axis=0)
            
            # Average of eye centers
            eyes_center = (left_eye_center + right_eye_center) / 2
            
            # Nose tip (reference point)
            nose_tip = [landmarks.landmark[1].x, landmarks.landmark[1].y]
            
            # Calculate gaze vector (simplified)
            gaze_x = eyes_center[0] - 0.5  # Normalized to center
            gaze_y = eyes_center[1] - 0.5
            
            # Determine gaze direction
            direction = 'center'
            threshold = 0.1
            
            if abs(gaze_x) > threshold or abs(gaze_y) > threshold:
                if abs(gaze_x) > abs(gaze_y):
                    direction = 'right' if gaze_x > 0 else 'left'
                else:
                    direction = 'down' if gaze_y > 0 else 'up'
            
            return GazeDirection(x=gaze_x, y=gaze_y, direction=direction)
            
        except Exception as e:
            print(f"Error estimating gaze: {e}")
        
        return None
    
    def analyze_eye_aspect_ratio(self, landmarks) -> float:
        """Calculate Eye Aspect Ratio for drowsiness detection"""
        try:
            # Eye landmarks for EAR calculation
            left_eye = [33, 160, 158, 133, 153, 144]
            right_eye = [362, 385, 387, 263, 373, 380]
            
            def eye_aspect_ratio(eye_points):
                # Vertical distances
                v1 = np.linalg.norm(
                    np.array([landmarks.landmark[eye_points[1]].x, landmarks.landmark[eye_points[1]].y]) -
                    np.array([landmarks.landmark[eye_points[5]].x, landmarks.landmark[eye_points[5]].y])
                )
                v2 = np.linalg.norm(
                    np.array([landmarks.landmark[eye_points[2]].x, landmarks.landmark[eye_points[2]].y]) -
                    np.array([landmarks.landmark[eye_points[4]].x, landmarks.landmark[eye_points[4]].y])
                )
                
                # Horizontal distance
                h = np.linalg.norm(
                    np.array([landmarks.landmark[eye_points[0]].x, landmarks.landmark[eye_points[0]].y]) -
                    np.array([landmarks.landmark[eye_points[3]].x, landmarks.landmark[eye_points[3]].y])
                )
                
                if h == 0:
                    return 0
                    
                ear = (v1 + v2) / (2.0 * h)
                return ear
            
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            
            return (left_ear + right_ear) / 2.0
            
        except Exception as e:
            print(f"Error calculating EAR: {e}")
            return 0.3  # Default value
    
    def calculate_attention_level(self, head_pose: HeadPose, gaze: GazeDirection, 
                                 ear: float, face_id: str) -> AttentionState:
        """Calculate attention level based on various parameters"""
        
        attention_score = 0.0
        target = 'distracted'
        
        # 1. Head pose contribution (40%)
        if head_pose:
            # If head is facing forward
            if abs(head_pose.yaw) < 20 and abs(head_pose.pitch) < 15:
                attention_score += 0.4
                
                # Determine target based on pitch
                if head_pose.pitch < -10:
                    target = 'instructor'  # Looking up (instructor)
                elif head_pose.pitch > 10:
                    target = 'laptop'  # Looking down (laptop)
                else:
                    target = 'center'  # Looking straight
            else:
                attention_score += 0.1
                if abs(head_pose.yaw) > 30:
                    target = 'peer' if head_pose.yaw > 0 else 'distracted'
        
        # 2. Gaze direction contribution (30%)
        if gaze:
            if gaze.direction == 'center':
                attention_score += 0.3
            elif gaze.direction in ['up', 'down']:
                attention_score += 0.2
            else:
                attention_score += 0.1
        
        # 3. Eye aspect ratio contribution (20%)
        # EAR > 0.2 means eyes are open
        if ear > 0.25:
            attention_score += 0.2
        elif ear > 0.15:
            attention_score += 0.1
        # ear < 0.15 might indicate closed eyes
        
        # 4. Historical consistency bonus (10%)
        if face_id in self.attention_history:
            recent_attention = [a.level for a in list(self.attention_history[face_id])[-5:]]
            if recent_attention and np.mean(recent_attention) > 0.5:
                attention_score += 0.1
        
        # Clamp to [0, 1]
        attention_score = max(0.0, min(1.0, attention_score))
        
        # Calculate confidence based on data availability
        confidence = 0.0
        if head_pose: confidence += 0.4
        if gaze: confidence += 0.3
        if ear > 0: confidence += 0.3
        
        return AttentionState(level=attention_score, target=target, confidence=confidence)
    
    def get_attention_color(self, attention_level: float) -> Tuple[int, int, int]:
        """Determine color based on attention level"""
        if attention_level >= ATTENTION_THRESHOLDS['high']:
            return ATTENTION_COLORS['high']
        elif attention_level >= ATTENTION_THRESHOLDS['medium']:
            return ATTENTION_COLORS['medium']
        else:
            return ATTENTION_COLORS['low']
    
    def draw_attention_visualization(self, frame, face_box, attention_state: AttentionState, 
                                    head_pose: Optional[HeadPose], face_id: str):
        """Draw attention visualization on frame"""
        x1, y1, x2, y2 = face_box
        color = self.get_attention_color(attention_state.level)
        
        # Draw bounding box
        thickness = 2 + int(attention_state.level * 3)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw attention meter (attention level bar)
        meter_width = 100
        meter_height = 10
        meter_x = x1
        meter_y = y1 - 40
        
        # Background
        cv2.rectangle(frame, (meter_x, meter_y), 
                     (meter_x + meter_width, meter_y + meter_height), 
                     (50, 50, 50), -1)
        
        # Filled part
        filled_width = int(meter_width * attention_state.level)
        cv2.rectangle(frame, (meter_x, meter_y), 
                     (meter_x + filled_width, meter_y + meter_height), 
                     color, -1)
        
        # Labels
        attention_text = f"Attention: {attention_state.level:.1%}"
        target_text = f"Looking at: {attention_state.target}"
        
        cv2.putText(frame, attention_text, (x1, y1 - 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, target_text, (x1, y1 - 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw head pose arrows if available
        if head_pose:
            # Arrow showing head direction
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            arrow_length = 50
            end_x = int(center_x + arrow_length * math.sin(math.radians(head_pose.yaw)))
            end_y = int(center_y - arrow_length * math.sin(math.radians(head_pose.pitch)))
            
            cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), 
                          (255, 255, 0), 2, tipLength=0.3)
    
    def draw_class_statistics(self, frame, width, height):
        """Draw overall class statistics"""
        # Background panel
        panel_height = 150
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "CLASS ENGAGEMENT ANALYTICS", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Calculate overall statistics
        total_students = len(self.attention_history)
        if total_students > 0:
            current_attention_levels = []
            attention_targets = defaultdict(int)
            
            for face_id, history in self.attention_history.items():
                if history:
                    latest = history[-1]
                    current_attention_levels.append(latest.level)
                    attention_targets[latest.target] += 1
            
            if current_attention_levels:
                avg_attention = np.mean(current_attention_levels)
                high_attention = sum(1 for a in current_attention_levels if a >= 0.7)
                
                # Display metrics
                metrics_y = 55
                
                # Average attention
                cv2.putText(frame, f"Avg Attention: {avg_attention:.1%}", 
                           (10, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           self.get_attention_color(avg_attention), 2)
                
                # Number of students
                cv2.putText(frame, f"Students: {total_students}", 
                           (250, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (255, 255, 255), 1)
                
                # High attention count
                cv2.putText(frame, f"Engaged: {high_attention}/{total_students}", 
                           (400, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (0, 255, 0), 1)
                
                # Target distribution
                targets_y = 85
                x_offset = 10
                for target, count in attention_targets.items():
                    text = f"{target}: {count}"
                    cv2.putText(frame, text, (x_offset, targets_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    x_offset += 120
                
                # Engagement timeline (last 30 seconds)
                if self.engagement_timeline:
                    timeline_y = 120
                    timeline_x = 10
                    timeline_width = min(len(self.engagement_timeline) * 2, width - 20)
                    
                    # Draw timeline graph
                    for i, engagement in enumerate(self.engagement_timeline[-timeline_width//2:]):
                        bar_height = int(engagement * 20)
                        x = timeline_x + i * 2
                        y = timeline_y - bar_height
                        color = self.get_attention_color(engagement)
                        cv2.line(frame, (x, timeline_y), (x, y), color, 2)
    
    def process_frame(self, frame):
        """Process frame and analyze attention"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        
        # Detect faces with MTCNN
        boxes, probs = self.mtcnn.detect(frame_rgb)
        
        current_attention_levels = []
        
        if boxes is not None:
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob > 0.9:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Ensure coordinates are within bounds
                    x1 = max(0, min(x1, width-1))
                    y1 = max(0, min(y1, height-1))
                    x2 = max(0, min(x2, width-1))
                    y2 = max(0, min(y2, height-1))
                    
                    # Extract face region for MediaPipe
                    face_roi = frame_rgb[y1:y2, x1:x2]
                    
                    if face_roi.size > 0:
                        # Process with MediaPipe
                        results = self.face_mesh.process(face_roi)
                        
                        if results.multi_face_landmarks:
                            landmarks = results.multi_face_landmarks[0]
                            
                            # Calculate metrics
                            roi_h, roi_w = face_roi.shape[:2]
                            head_pose = self.calculate_head_pose(landmarks, roi_h, roi_w)
                            gaze = self.estimate_gaze_direction(landmarks, roi_h, roi_w)
                            ear = self.analyze_eye_aspect_ratio(landmarks)
                            
                            # Calculate attention
                            face_id = f"face_{i}"
                            attention_state = self.calculate_attention_level(
                                head_pose, gaze, ear, face_id
                            )
                            
                            # Store in history
                            self.attention_history[face_id].append(attention_state)
                            if head_pose:
                                self.head_pose_history[face_id].append(head_pose)
                            if gaze:
                                self.gaze_history[face_id].append(gaze)
                            
                            # Update statistics
                            self.attention_stats[face_id]['total_frames'] += 1
                            self.attention_stats[face_id]['attention_sum'] += attention_state.level
                            self.attention_stats[face_id][attention_state.target] += 1
                            
                            current_attention_levels.append(attention_state.level)
                            
                            # Draw visualization
                            self.draw_attention_visualization(
                                frame, (x1, y1, x2, y2), attention_state, head_pose, face_id
                            )
                            
                            # Draw face mesh landmarks (optional)
                            if head_pose:  # Only if we have good tracking
                                for landmark in landmarks.landmark:
                                    x = int(landmark.x * roi_w) + x1
                                    y = int(landmark.y * roi_h) + y1
                                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        # Update engagement timeline
        if current_attention_levels:
            avg_engagement = np.mean(current_attention_levels)
            self.engagement_timeline.append(avg_engagement)
            if len(self.engagement_timeline) > 500:  # Keep last 500 frames
                self.engagement_timeline.pop(0)
        
        # Draw class statistics
        self.draw_class_statistics(frame, width, height)
        
        return frame
    
    def generate_report(self, output_path: str):
        """Generate attention analysis report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_faces_tracked': len(self.attention_stats),
            'individual_stats': {},
            'class_summary': {}
        }
        
        all_attention_levels = []
        target_distribution = defaultdict(int)
        
        for face_id, stats in self.attention_stats.items():
            total_frames = stats['total_frames']
            if total_frames > 0:
                avg_attention = stats['attention_sum'] / total_frames
                all_attention_levels.append(avg_attention)
                
                # Individual stats
                report['individual_stats'][face_id] = {
                    'average_attention': round(avg_attention, 3),
                    'total_frames': total_frames,
                    'target_distribution': {
                        k: v for k, v in stats.items() 
                        if k not in ['total_frames', 'attention_sum']
                    }
                }
                
                # Update overall distribution
                for target in ['instructor', 'laptop', 'peer', 'distracted']:
                    if target in stats:
                        target_distribution[target] += stats[target]
        
        # Class summary
        if all_attention_levels:
            report['class_summary'] = {
                'average_attention': round(np.mean(all_attention_levels), 3),
                'std_attention': round(np.std(all_attention_levels), 3),
                'highly_engaged': sum(1 for a in all_attention_levels if a >= 0.7),
                'moderately_engaged': sum(1 for a in all_attention_levels if 0.4 <= a < 0.7),
                'low_engaged': sum(1 for a in all_attention_levels if a < 0.4),
                'target_distribution': dict(target_distribution)
            }
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report

def main():
    parser = argparse.ArgumentParser(description='Student Attention Tracking System')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--output', type=str, default='attention_output.mp4', 
                       help='Output video path')
    parser.add_argument('--report', type=str, default='attention_report.json', 
                       help='Path for JSON report')
    parser.add_argument('--no-show', action='store_true', 
                       help='Process without showing window')
    parser.add_argument('--skip-frames', type=int, default=1, 
                       help='Process every N frames')
    args = parser.parse_args()
    
    # Check video file
    if not os.path.exists(args.video):
        print(f"❌ Error: Video file '{args.video}' not found!")
        return
    
    # Initialize analyzer
    print("🚀 Initializing Attention Analysis System...")
    analyzer = AttentionAnalyzer()
    
    # Video setup
    cap = cv2.VideoCapture(args.video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {args.video}")
    print(f"📊 Properties: {fps} FPS, {width}x{height}, {total_frames} frames")
    
    # Output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    print("\n🎬 Starting attention analysis...")
    print("📊 Tracking: Head pose, Eye gaze, Attention level")
    if not args.no_show:
        print("Controls: Q=Quit, SPACE=Pause, R=Generate Report")
    
    paused = False
    
    try:
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames if specified
                if frame_count % args.skip_frames != 0:
                    out.write(frame)
                    continue
                
                # Process frame
                processed_frame = analyzer.process_frame(frame)
                
                # Add frame counter and FPS
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(processed_frame, f"Frame: {frame_count}/{total_frames} | FPS: {fps_current:.1f}", 
                           (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Write output
                out.write(processed_frame)
                
                # Show frame
                if not args.no_show:
                    cv2.imshow('Student Attention Analysis', processed_frame)
            
            # Handle keyboard input
            if not args.no_show:
                key = cv2.waitKey(1 if not paused else 30) & 0xFF
                
                if key == ord('q'):
                    print("\n⏹️ Stopping...")
                    break
                elif key == ord(' '):
                    paused = not paused
                    print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
                elif key == ord('r'):
                    print("📊 Generating report...")
                    report = analyzer.generate_report(args.report)
                    print(f"✅ Report saved to {args.report}")
    
    finally:
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Generate final report
        print("\n📊 Generating final report...")
        report = analyzer.generate_report(args.report)
        
        # Print summary
        print("\n" + "="*60)
        print("📈 ATTENTION ANALYSIS SUMMARY")
        print("="*60)
        
        if 'class_summary' in report:
            summary = report['class_summary']
            print(f"\n👥 Total Students Tracked: {len(report['individual_stats'])}")
            print(f"📊 Average Class Attention: {summary.get('average_attention', 0):.1%}")
            print(f"✨ Highly Engaged: {summary.get('highly_engaged', 0)} students")
            print(f"📝 Moderately Engaged: {summary.get('moderately_engaged', 0)} students")
            print(f"⚠️  Low Engagement: {summary.get('low_engaged', 0)} students")
            
            if 'target_distribution' in summary:
                print("\n👀 Attention Targets:")
                for target, count in summary['target_distribution'].items():
                    print(f"  • {target}: {count} frames")
        
        print(f"\n✅ Analysis complete!")
        print(f"💾 Video saved to: {args.output}")
        print(f"📄 Report saved to: {args.report}")

if __name__ == "__main__":
    main()