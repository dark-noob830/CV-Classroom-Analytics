import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import supervision as sv
from torchvision import transforms
import argparse
import time
from collections import deque
import os
from fer import FER  # For emotion recognition

# Image transformations
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ================== Settings ==================
# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

# Similarity threshold for recognition
RECOGNITION_THRESHOLD = 0.40  # Very low threshold - always use best match

# Load models
mtcnn = MTCNN(keep_all=True, device=device, min_face_size=20, thresholds=[0.5, 0.6, 0.6])  # More sensitive
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()

# Initialize emotion detector with custom settings
try:
    emotion_detector = FER(mtcnn=False)  # Disable MTCNN as we're using our own detection
    print("✅ Emotion detector loaded successfully")
except Exception as e:
    print(f"❌ Error loading emotion detector: {e}")
    print("Using fallback emotion detection...")
    emotion_detector = None

# Load face database
def load_database(database_path):
    try:
        database = torch.load(database_path, map_location=device)
        print("✅ Student database loaded successfully.")
        print("Known identities:", list(database.keys()))
        return database
    except FileNotFoundError:
        print(f"❌ Error: Database file '{database_path}' not found.")
        return None
    except Exception as e:
        print(f"❌ Error loading database: {str(e)}")
        return None

# Initialize ByteTrack tracker
tracker = sv.ByteTrack()

# Dictionary to store recognized identity and emotion for each track_id
track_identities = {}
track_emotions = {}
# Track the order of track IDs for memory management
track_id_order = deque(maxlen=100)

# Separate box drawing from label drawing
box_annotator = sv.BoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
label_annotator = sv.LabelAnnotator(
    text_scale=0.6,
    text_thickness=1,
    text_position=sv.Position.TOP_CENTER,
    color_lookup=sv.ColorLookup.TRACK
)
# ===============================================

def cosine_similarity(emb1, emb2):
    emb2_tensor = emb2 if isinstance(emb2, torch.Tensor) else torch.tensor(emb2, device=emb1.device)
    return torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2_tensor.unsqueeze(0)).item()

def find_identity(embedding, db, threshold):
    max_similarity = -1
    best_match = "Unknown"
    
    # Always find the best similarity
    for name, db_embedding in db.items():
        sim = cosine_similarity(embedding, db_embedding)
        if sim > max_similarity:
            max_similarity = sim
            best_match = name
    
    # Always return the best match, even if similarity is low
    if max_similarity > 0.1:  # Minimum similarity
        print(f"  Best match: {best_match} (similarity: {max_similarity:.3f})")
        return best_match, max_similarity
    else:
        print(f"  Very low similarity: {max_similarity:.3f} - using best match anyway")
        return best_match, max_similarity

def analyze_emotion(face_crop):
    try:
        if emotion_detector is None:
            return "neutral", 0.5  # Fallback if FER not available
        
        # Ensure the face_crop is in the correct format
        if len(face_crop.shape) == 3:
            if face_crop.shape[2] == 4:  # RGBA to RGB
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGBA2RGB)
            elif face_crop.shape[2] == 3:  # BGR to RGB
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        # Resize for emotion detection (FER works best with 48x48)
        emotion_crop = cv2.resize(face_crop, (48, 48))
        
        # Analyze emotion
        emotions = emotion_detector.detect_emotions(emotion_crop)
        
        if emotions and len(emotions) > 0:
            emotion_scores = emotions[0]['emotions']
            if emotion_scores:
                dominant_emotion = max(emotion_scores.items(), key=lambda item: item[1])[0]
                confidence = emotion_scores[dominant_emotion]
                return dominant_emotion, confidence
        
        return "neutral", 0.3  # Default fallback
    except Exception as e:
        print(f"Error in emotion analysis: {str(e)}")
        return "neutral", 0.3

# ================== Main video processing loop ==================
def process_video(video_path, database):
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file '{video_path}' not found.")
        return
    
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"❌ Error: Could not open video file {video_path}")
        return

    print("Processing video... Press 'q' to quit.")
    frame_count = 0
    start_time = time.time()

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_count += 1

        # 1. Face detection
        boxes, probs = mtcnn.detect(frame_rgb)
        print(f"Frame {frame_count}: Detected {len(boxes) if boxes is not None else 0} faces")

        detections_sv = sv.Detections.empty()
        if boxes is not None and len(boxes) > 0:
            # Lower threshold for better detection
            valid_indices = probs > 0.7  # Lowered from 0.9
            if np.any(valid_indices):
                valid_boxes = boxes[valid_indices].astype(np.float32)
                valid_probs = probs[valid_indices].astype(np.float32)
                print(f"  Valid faces after filtering: {len(valid_boxes)}")

                detections_sv = sv.Detections(
                    xyxy=valid_boxes,
                    confidence=valid_probs
                )

        # 2. Update tracker
        tracked_detections = tracker.update_with_detections(detections_sv)

        # Process tracked detections
        labels = []
        if tracked_detections.tracker_id is not None:
            for xyxy, track_id in zip(tracked_detections.xyxy, tracked_detections.tracker_id):
                # Memory management
                if len(track_identities) >= 100:
                    oldest_id = track_id_order.popleft()
                    if oldest_id in track_identities:
                        del track_identities[oldest_id]
                    if oldest_id in track_emotions:
                        del track_emotions[oldest_id]
                
                if track_id not in track_identities:
                    x1, y1, x2, y2 = map(int, xyxy)
                    face_crop = frame_rgb[y1:y2, x1:x2]
                    
                    if face_crop.size == 0:
                        print(f"  Empty face crop for track {track_id}")
                        continue
                    
                    # Face recognition
                    try:
                        face_crop_pil = Image.fromarray(face_crop)
                        with torch.no_grad():
                            face_tensor = transform(face_crop_pil).to(device)
                            embedding = resnet(face_tensor.unsqueeze(0)).squeeze()
                        
                        identity, similarity = find_identity(embedding, database, RECOGNITION_THRESHOLD)
                        track_identities[track_id] = (identity, similarity)
                        print(f"  Track {track_id}: Identified as {identity} (similarity: {similarity:.3f})")
                        
                        # Emotion analysis
                        emotion, emotion_conf = analyze_emotion(face_crop)
                        track_emotions[track_id] = (emotion, emotion_conf)
                        print(f"  Track {track_id}: Emotion = {emotion} (confidence: {emotion_conf:.3f})")
                        
                        track_id_order.append(track_id)
                    except Exception as e:
                        print(f"Error processing face for track {track_id}: {str(e)}")
                        continue
                
                # Get stored identity and emotion
                identity, similarity = track_identities[track_id]
                emotion, emotion_conf = track_emotions[track_id]
                
                # Create label with all information
                label = f"ID:{track_id} {identity} ({similarity:.2f}) - {emotion} ({emotion_conf:.2f})"
                labels.append(label)

        # Draw boxes and labels
        frame = box_annotator.annotate(
            scene=frame,
            detections=tracked_detections
        )
        frame = label_annotator.annotate(
            scene=frame,
            detections=tracked_detections,
            labels=labels
        )

        # Show FPS and emotion summary
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show emotion summary
        emotion_summary = "Emotions: "
        emotion_counts = {}
        for emotion, _ in track_emotions.values():
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        for emotion, count in emotion_counts.items():
            emotion_summary += f"{emotion}:{count} "
        
        cv2.putText(frame, emotion_summary, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.imshow('Classroom Attendance & Emotion Analysis - Computer Vision Project', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print(f"✅ Processing finished. Processed {frame_count} frames in {elapsed_time:.2f} seconds.")

# Argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description='Face Recognition, Tracking and Emotion Analysis in Classroom Videos')
    parser.add_argument('--video', type=str, default='videos/Team_3.mp4', help='Path to the video file')
    parser.add_argument('--database', type=str, default='data_extract/person_medoids.pt', help='Path to the face database')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    database = load_database(args.database)
    
    if database:
        process_video(args.video, database)
    else:
        print("❌ Exiting due to database loading error.")
