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
RECOGNITION_THRESHOLD = 0.75

# Load models
mtcnn = MTCNN(keep_all=True, device=device, min_face_size=40, thresholds=[0.6, 0.7, 0.7])
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()

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

# Dictionary to store recognized identity for each track_id
track_identities = {}
# Track the order of track IDs for memory management
track_id_order = deque(maxlen=100)  # Keep last 100 track IDs

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
    identity = "Unknown"
    for name, db_embedding in db.items():
        sim = cosine_similarity(embedding, db_embedding)
        if sim > max_similarity:
            max_similarity = sim
            identity = name
    return identity, max_similarity

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

        detections_sv = sv.Detections.empty()
        if boxes is not None:
            valid_indices = probs > 0.9
            valid_boxes = boxes[valid_indices].astype(np.float32)
            valid_probs = probs[valid_indices].astype(np.float32)

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
                # Memory management - remove oldest track if we exceed limit
                if len(track_identities) >= 100:
                    oldest_id = track_id_order.popleft()
                    if oldest_id in track_identities:
                        del track_identities[oldest_id]
                
                if track_id not in track_identities:
                    x1, y1, x2, y2 = map(int, xyxy)
                    face_crop = frame_rgb[y1:y2, x1:x2]
                    
                    # Check if crop is valid
                    if face_crop.size == 0:
                        continue
                    
                    try:
                        face_crop_pil = Image.fromarray(face_crop)
                        with torch.no_grad():
                            face_tensor = transform(face_crop_pil).to(device)
                            embedding = resnet(face_tensor.unsqueeze(0)).squeeze()
                        
                        identity, similarity = find_identity(embedding, database, RECOGNITION_THRESHOLD)
                        track_identities[track_id] = (identity, similarity)
                        track_id_order.append(track_id)
                    except Exception as e:
                        print(f"Error processing face for track {track_id}: {str(e)}")
                        continue
                
                identity, similarity = track_identities[track_id]
                labels.append(f"ID:{track_id} {identity} ({similarity:.2f})")

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

        # Show FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Classroom Attendance Analysis - Computer Vision Project', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print(f"✅ Processing finished. Processed {frame_count} frames in {elapsed_time:.2f} seconds.")

# Argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description='Face Recognition and Tracking in Classroom Videos')
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