#!/usr/bin/env python3
"""
Advanced Emotion Analysis System - Only Emotions
Advanced emotion analysis without identity recognition
Using deep learning models
"""

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
import argparse
import os
from collections import defaultdict, deque
import time
from fer import FER  # For real emotion detection

# ================== Settings ==================
# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'🖥️  Running on device: {device}')

# Load models
print("📦 Loading face detection model...")
mtcnn = MTCNN(keep_all=True, device=device, min_face_size=30, thresholds=[0.6, 0.7, 0.7])

# Initialize FER emotion detector
print("🧠 Loading emotion recognition model...")
emotion_detector = FER(mtcnn=False)  # We use our own MTCNN

# Emotion colors with better visibility
EMOTION_COLORS = {
    'angry': (0, 0, 255),        # Red
    'disgust': (128, 0, 128),    # Purple
    'fear': (255, 0, 255),       # Magenta
    'happy': (0, 255, 0),        # Green
    'sad': (255, 0, 0),          # Blue
    'surprise': (0, 255, 255),   # Cyan
    'neutral': (200, 200, 200)   # Light Gray
}

# Emotion emojis for better visualization
EMOTION_EMOJIS = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'sad': '😢',
    'surprise': '😲',
    'neutral': '😐'
}

def preprocess_face(face_crop):
    """Preprocess face image for improved detection accuracy"""
    try:
        # Convert to RGB if needed
        if len(face_crop.shape) == 2:
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_GRAY2RGB)
        elif face_crop.shape[2] == 4:
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGBA2RGB)
        
        # Improve contrast with CLAHE
        lab = cv2.cvtColor(face_crop, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # Remove noise
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return enhanced
    except Exception as e:
        print(f"⚠️ Error in preprocessing: {e}")
        return face_crop

def analyze_emotion_advanced(face_crop):
    """Advanced emotion analysis with FER"""
    try:
        # Preprocess image
        processed_face = preprocess_face(face_crop)
        
        # Resize for best performance (FER works better with 48x48 or larger)
        h, w = processed_face.shape[:2]
        
        # If too small, enlarge
        if h < 48 or w < 48:
            scale = max(48/h, 48/w) * 1.2
            new_h, new_w = int(h * scale), int(w * scale)
            processed_face = cv2.resize(processed_face, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Detect emotions with FER
        emotions = emotion_detector.detect_emotions(processed_face)
        
        if emotions and len(emotions) > 0:
            emotion_scores = emotions[0]['emotions']
            
            # Find dominant emotion
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            confidence = emotion_scores[dominant_emotion]
            
            # If confidence is too low, consider neutral
            if confidence < 0.3:
                return 'neutral', 0.5
            
            return dominant_emotion, confidence
        else:
            return 'neutral', 0.5
            
    except Exception as e:
        print(f"⚠️ Error in emotion analysis: {e}")
        return 'neutral', 0.5

def smooth_emotions(emotion_history, current_emotion):
    """Smooth emotions to prevent severe fluctuations"""
    if len(emotion_history) < 3:
        return current_emotion
    
    # Count emotions in last 3 frames
    recent_emotions = [e[0] for e in list(emotion_history)[-3:]]
    emotion_counts = defaultdict(int)
    for emotion in recent_emotions:
        emotion_counts[emotion] += 1
    
    # If current emotion is in majority, return it
    if emotion_counts[current_emotion[0]] >= 2:
        return current_emotion
    
    # Otherwise, return dominant emotion
    dominant = max(emotion_counts.items(), key=lambda x: x[1])[0]
    return (dominant, current_emotion[1] * 0.8)  # Reduce confidence

def draw_emotion_stats(frame, emotion_stats, x=10, y=60):
    """Draw emotion statistics on frame"""
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x-5, y-30), (x+250, y+len(emotion_stats)*30+10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Title
    cv2.putText(frame, "Emotion Statistics:", (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Display statistics for each emotion
    for i, (emotion, count) in enumerate(sorted(emotion_stats.items(), key=lambda x: x[1], reverse=True)):
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))
        emoji = EMOTION_EMOJIS.get(emotion, '')
        text = f"{emoji} {emotion}: {count}"
        cv2.putText(frame, text, (x, y + i*25 + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def analyze_emotion_trends(emotion_history):
    """Analyze emotion trends"""
    if not emotion_history:
        return {}, {}
    
    # Count emotions
    emotion_counts = defaultdict(int)
    confidence_sum = defaultdict(float)
    total_frames = 0
    
    for track_emotions in emotion_history.values():
        for emotion, confidence in track_emotions:
            emotion_counts[emotion] += 1
            confidence_sum[emotion] += confidence
            total_frames += 1
    
    if total_frames == 0:
        return {}, {}
    
    # Calculate percentages and average confidence
    emotion_percentages = {}
    avg_confidence = {}
    for emotion, count in emotion_counts.items():
        emotion_percentages[emotion] = (count / total_frames) * 100
        avg_confidence[emotion] = confidence_sum[emotion] / count
    
    return emotion_percentages, avg_confidence

def main():
    parser = argparse.ArgumentParser(description='Advanced Emotion Analysis System')
    parser.add_argument('--video', type=str, default='videos/Team_3.mp4', help='Path to video file')
    parser.add_argument('--output', type=str, default='emotion_output.mp4', help='Output video path')
    parser.add_argument('--no-show', action='store_true', help='Do not show video window')
    parser.add_argument('--skip-frames', type=int, default=2, help='Process every N frames (default: 2)')
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.video):
        print(f"❌ Error: Video file '{args.video}' not found!")
        return
    
    # Video setup
    video_capture = cv2.VideoCapture(args.video)
    if not video_capture.isOpened():
        print(f"❌ Error: Could not open video {args.video}")
        return
    
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {args.video}")
    print(f"📊 Properties: {fps} FPS, {width}x{height}, {total_frames} frames")
    print(f"⏭️  Processing every {args.skip_frames} frame(s)")
    
    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Tracking variables
    face_tracks = {}  # Simple tracking based on position
    emotion_history = defaultdict(lambda: deque(maxlen=10))  # Keep last 10 emotions per face
    emotion_stats = defaultdict(int)  # Overall emotion statistics
    
    frame_count = 0
    processed_frames = 0
    start_time = time.time()
    
    print("\n🎬 Starting advanced emotion analysis...")
    if not args.no_show:
        print("📺 Controls: SPACE=Pause, S=Stats, Q=Quit")
    
    paused = False
    show_stats = True
    
    while True:
        if not paused:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % args.skip_frames != 0:
                out.write(frame)
                continue
            
            processed_frames += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Face detection
            boxes, probs = mtcnn.detect(frame_rgb)
            
            current_faces = []
            
            if boxes is not None and len(boxes) > 0:
                # Filter valid detections
                valid_indices = probs > 0.8
                if np.any(valid_indices):
                    valid_boxes = boxes[valid_indices].astype(np.int32)
                    valid_probs = probs[valid_indices]
                    
                    # Process each detected face
                    for i, (box, prob) in enumerate(zip(valid_boxes, valid_probs)):
                        x1, y1, x2, y2 = box
                        
                        # Ensure coordinates are within bounds
                        x1 = max(0, min(x1, width-1))
                        y1 = max(0, min(y1, height-1))
                        x2 = max(0, min(x2, width-1))
                        y2 = max(0, min(y2, height-1))
                        
                        if x2 > x1 + 10 and y2 > y1 + 10:  # Minimum face size
                            # Add padding for better emotion detection
                            padding = 5
                            x1_pad = max(0, x1 - padding)
                            y1_pad = max(0, y1 - padding)
                            x2_pad = min(width, x2 + padding)
                            y2_pad = min(height, y2 + padding)
                            
                            # Extract face crop
                            face_crop = frame_rgb[y1_pad:y2_pad, x1_pad:x2_pad]
                            
                            if face_crop.size > 0:
                                # Analyze emotion
                                emotion, confidence = analyze_emotion_advanced(face_crop)
                                
                                # Simple face tracking by position
                                face_id = f"face_{i}"
                                
                                # Add to history and smooth
                                emotion_history[face_id].append((emotion, confidence))
                                emotion, confidence = smooth_emotions(emotion_history[face_id], (emotion, confidence))
                                
                                # Update statistics
                                emotion_stats[emotion] += 1
                                
                                # Draw bounding box with emotion color
                                color = EMOTION_COLORS.get(emotion, (255, 255, 255))
                                thickness = 2 + int(confidence * 2)  # Thicker for higher confidence
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                                
                                # Draw emotion label with background
                                emoji = EMOTION_EMOJIS.get(emotion, '')
                                label = f"{emoji} {emotion} ({confidence:.2f})"
                                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                                
                                # Label background
                                cv2.rectangle(frame, (x1, y1-30), (x1+label_size[0]+10, y1), color, -1)
                                
                                # Label text
                                cv2.putText(frame, label, (x1+5, y1-8), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                
                                current_faces.append((face_id, emotion, confidence))
            
            # Draw statistics if enabled
            if show_stats and emotion_stats:
                draw_emotion_stats(frame, emotion_stats)
            
            # Draw progress bar
            progress = frame_count / total_frames
            bar_height = 5
            cv2.rectangle(frame, (0, height-bar_height), (int(width*progress), height), (0, 255, 0), -1)
            
            # Draw status bar
            elapsed = time.time() - start_time
            fps_current = processed_frames / elapsed if elapsed > 0 else 0
            status = f"Frame: {frame_count}/{total_frames} | FPS: {fps_current:.1f} | Faces: {len(current_faces)}"
            
            # Status background
            cv2.rectangle(frame, (0, height-35), (width, height-bar_height), (0, 0, 0), -1)
            cv2.putText(frame, status, (10, height-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Write frame
            out.write(frame)
            
            # Show frame
            if not args.no_show:
                cv2.imshow('Advanced Emotion Analysis', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # Space to pause
                    paused = not paused
                    print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
                elif key == ord('s'):  # Toggle stats
                    show_stats = not show_stats
                    print(f"📊 Stats: {'ON' if show_stats else 'OFF'}")
                elif key == ord('q') or key == 27:  # Quit
                    print("\n⏹️  Stopping...")
                    break
        
        else:  # Paused
            if not args.no_show:
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    paused = False
                    print("▶️  Resumed")
                elif key == ord('q') or key == 27:
                    break
    
    # Cleanup
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Final analysis
    print("\n" + "="*50)
    print("📊 FINAL EMOTION ANALYSIS REPORT")
    print("="*50)
    
    percentages, avg_conf = analyze_emotion_trends(emotion_history)
    
    if percentages:
        print("\n🎭 Emotion Distribution:")
        for emotion, percentage in sorted(percentages.items(), key=lambda x: x[1], reverse=True):
            emoji = EMOTION_EMOJIS.get(emotion, '')
            bar = '█' * int(percentage/2)
            print(f"  {emoji} {emotion:8s}: {bar} {percentage:.1f}% (avg conf: {avg_conf[emotion]:.2f})")
        
        # Find dominant emotion
        dominant = max(percentages.items(), key=lambda x: x[1])
        print(f"\n🏆 Dominant Emotion: {EMOTION_EMOJIS.get(dominant[0], '')} {dominant[0]} ({dominant[1]:.1f}%)")
    
    print(f"\n📹 Processing Statistics:")
    print(f"  • Total frames: {frame_count}")
    print(f"  • Processed frames: {processed_frames}")
    print(f"  • Processing time: {time.time()-start_time:.2f}s")
    print(f"  • Average FPS: {processed_frames/(time.time()-start_time):.2f}")
    
    print(f"\n✅ Analysis complete!")
    print(f"💾 Output saved to: {args.output}")

if __name__ == "__main__":
    main()