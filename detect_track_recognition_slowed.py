# Simple and Stable Face Detection & Tracking System
# Improvements over your code:
# - Better identity persistence
# - Periodic re-recognition for accuracy
# - Smoothed bounding boxes
# - Better detection thresholds

import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import supervision as sv
from torchvision import transforms
from collections import defaultdict, deque

# Image transformations
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ================== Settings ==================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

# Recognition settings
RECOGNITION_THRESHOLD = 0.70  # Lowered for better matching
DETECTION_CONFIDENCE = 0.85   # Balanced threshold

# Load models
print("Loading models...")
mtcnn = MTCNN(
    keep_all=True, 
    device=device, 
    min_face_size=40,  # Minimum face size
    thresholds=[0.6, 0.7, 0.7]  # Detection thresholds
)
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
print("✅ Models loaded")

# Load face database
try:
    database = torch.load('data_extract/person_medoids.pt', map_location=device)
    print("✅ Student database loaded successfully.")
    print("Known identities:", list(database.keys()))
except FileNotFoundError:
    print("❌ Error: 'person_medoids.pt' not found.")
    exit()

# ================== Enhanced Tracking System ==================
class EnhancedTracker:
    def __init__(self):
        # Initialize ByteTrack with default parameters
        self.tracker = sv.ByteTrack()
        
        # Track management
        self.track_identities = {}  # track_id -> identity
        self.track_embeddings = {}  # track_id -> embedding
        self.track_confidence = {}  # track_id -> confidence score
        self.identity_votes = defaultdict(lambda: defaultdict(int))  # Voting system
        
        # Recognition management
        self.last_recognition_frame = {}
        self.recognition_interval = 30  # Re-recognize every 30 frames
        self.frame_count = 0
        
        # Position smoothing
        self.position_history = defaultdict(lambda: deque(maxlen=5))
        
        print("✅ Enhanced tracker initialized")
    
    def smooth_bbox(self, track_id, bbox):
        """Smooth bounding box positions"""
        self.position_history[track_id].append(bbox)
        
        if len(self.position_history[track_id]) > 1:
            # Average recent positions
            positions = np.array(list(self.position_history[track_id]))
            smoothed = np.mean(positions, axis=0)
            return smoothed
        return bbox
    
    def update_identity(self, track_id, embedding, force=False):
        """Update track identity with voting system"""
        # Check if we should update
        should_update = (
            force or
            track_id not in self.track_identities or
            self.track_identities[track_id] == "Unknown" or
            self.frame_count - self.last_recognition_frame.get(track_id, 0) >= self.recognition_interval
        )
        
        if not should_update:
            return self.track_identities.get(track_id, "Unknown")
        
        # Find identity
        identity, similarity = find_identity(embedding, database, RECOGNITION_THRESHOLD)
        
        # Use voting system for stability
        if identity != "Unknown":
            self.identity_votes[track_id][identity] += 1
            
            # Need at least 2 votes for stable identity
            if self.identity_votes[track_id][identity] >= 2:
                self.track_identities[track_id] = identity
                self.track_confidence[track_id] = similarity
            elif track_id not in self.track_identities:
                self.track_identities[track_id] = "Unknown"
        else:
            if track_id not in self.track_identities:
                self.track_identities[track_id] = "Unknown"
        
        self.track_embeddings[track_id] = embedding
        self.last_recognition_frame[track_id] = self.frame_count
        
        return self.track_identities.get(track_id, "Unknown")
    
    def process_frame(self, frame_rgb, detections_sv):
        """Process a frame with detection and tracking"""
        self.frame_count += 1
        
        # Update ByteTrack
        tracked_detections = self.tracker.update_with_detections(detections_sv)
        
        # Process tracked objects
        labels = []
        smoothed_detections = tracked_detections.xyxy.copy() if tracked_detections.xyxy is not None else None
        
        if tracked_detections.tracker_id is not None and len(tracked_detections.tracker_id) > 0:
            for i, (xyxy, track_id) in enumerate(zip(tracked_detections.xyxy, tracked_detections.tracker_id)):
                # Smooth bounding box
                smoothed_bbox = self.smooth_bbox(track_id, xyxy)
                smoothed_detections[i] = smoothed_bbox
                
                # Extract face for recognition
                x1, y1, x2, y2 = map(int, smoothed_bbox)
                
                # Ensure valid coordinates
                h, w = frame_rgb.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    try:
                        face_crop = frame_rgb[y1:y2, x1:x2]
                        face_crop_pil = Image.fromarray(face_crop)
                        
                        with torch.no_grad():
                            face_tensor = transform(face_crop_pil).to(device)
                            embedding = resnet(face_tensor.unsqueeze(0)).squeeze()
                        
                        # Update identity
                        identity = self.update_identity(track_id, embedding)
                    except Exception as e:
                        print(f"Error processing track {track_id}: {e}")
                        identity = self.track_identities.get(track_id, "Unknown")
                else:
                    identity = self.track_identities.get(track_id, "Unknown")
                
                # Create label
                if identity != "Unknown":
                    confidence = self.track_confidence.get(track_id, 0)
                    label = f"ID:{track_id} {identity} ({confidence:.2f})"
                else:
                    label = f"ID:{track_id} Unknown"
                
                labels.append(label)
        
        # Update tracked detections with smoothed positions
        if smoothed_detections is not None:
            tracked_detections.xyxy = smoothed_detections
        
        return tracked_detections, labels
    
    def cleanup_old_tracks(self, active_track_ids):
        """Remove old tracks that are no longer active"""
        all_track_ids = set(self.track_identities.keys())
        inactive_tracks = all_track_ids - set(active_track_ids) if active_track_ids is not None else all_track_ids
        
        for track_id in inactive_tracks:
            if self.frame_count - self.last_recognition_frame.get(track_id, 0) > 100:
                # Remove old inactive tracks
                for storage in [self.track_identities, self.track_embeddings, 
                              self.track_confidence, self.identity_votes,
                              self.last_recognition_frame, self.position_history]:
                    storage.pop(track_id, None)

# Initialize enhanced tracker
enhanced_tracker = EnhancedTracker()

# Visualization setup
box_annotator = sv.BoxAnnotator(
    thickness=2,
    color_lookup=sv.ColorLookup.TRACK
)
label_annotator = sv.LabelAnnotator(
    text_scale=0.5,
    text_thickness=2,
    text_position=sv.Position.TOP_LEFT,
    color_lookup=sv.ColorLookup.TRACK
)

def cosine_similarity(emb1, emb2):
    """Calculate cosine similarity between embeddings"""
    emb2_tensor = emb2 if isinstance(emb2, torch.Tensor) else torch.tensor(emb2, device=emb1.device)
    return torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2_tensor.unsqueeze(0)).item()

def find_identity(embedding, db, threshold):
    """Find best matching identity in database"""
    max_similarity = -1
    identity = "Unknown"
    
    for name, db_embedding in db.items():
        sim = cosine_similarity(embedding, db_embedding)
        if sim > max_similarity:
            max_similarity = sim
            if sim > threshold:
                identity = name
    
    return identity, max_similarity

# ================== Main video processing loop ==================
video_path = 'videos/Team_3.mp4'
video_capture = cv2.VideoCapture(video_path)

if not video_capture.isOpened():
    print(f"❌ Error: Could not open video file {video_path}")
    exit()

# Get video info
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"🎥 Video: {fps} FPS, {total_frames} frames")
print("Press 'q' to quit, 'r' to reset tracker")

# Processing variables
frame_count = 0
fps_counter = 0
import time
fps_start_time = time.time()

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break
    
    frame_count += 1
    fps_counter += 1
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Face detection with MTCNN
    boxes, probs = mtcnn.detect(frame_rgb)
    
    # Create supervision detections
    detections_sv = sv.Detections.empty()
    if boxes is not None and len(boxes) > 0:
        # Filter by confidence
        valid_indices = probs > DETECTION_CONFIDENCE
        valid_boxes = boxes[valid_indices].astype(np.float32)
        valid_probs = probs[valid_indices].astype(np.float32)
        
        if len(valid_boxes) > 0:
            detections_sv = sv.Detections(
                xyxy=valid_boxes,
                confidence=valid_probs
            )
    
    # Process frame with enhanced tracker
    tracked_detections, labels = enhanced_tracker.process_frame(frame_rgb, detections_sv)
    
    # Cleanup old tracks periodically
    if frame_count % 100 == 0:
        enhanced_tracker.cleanup_old_tracks(
            tracked_detections.tracker_id if tracked_detections.tracker_id is not None else []
        )
    
    # Draw annotations
    frame = box_annotator.annotate(
        scene=frame,
        detections=tracked_detections
    )
    
    if labels:
        frame = label_annotator.annotate(
            scene=frame,
            detections=tracked_detections,
            labels=labels
        )
    
    # Display FPS and stats
    if fps_counter % 30 == 0:
        fps_end_time = time.time()
        current_fps = 30 / (fps_end_time - fps_start_time)
        fps_start_time = fps_end_time
        
        active_tracks = len(tracked_detections.tracker_id) if tracked_detections.tracker_id is not None else 0
        known_identities = sum(1 for id in enhanced_tracker.track_identities.values() if id != "Unknown")
        
        print(f"FPS: {current_fps:.1f} | Frame: {frame_count}/{total_frames} | "
              f"Tracks: {active_tracks} | Identified: {known_identities}")
    
    # Draw status bar
    cv2.rectangle(frame, (0, 0), (300, 25), (0, 0, 0), -1)
    status_text = f"Frame: {frame_count} | Tracks: {len(enhanced_tracker.track_identities)}"
    cv2.putText(frame, status_text, (10, 18), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Show frame
    cv2.imshow('Enhanced Face Tracking System', frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        enhanced_tracker = EnhancedTracker()
        print("Tracker reset!")

# Cleanup
video_capture.release()
cv2.destroyAllWindows()

# Print summary
print("\n" + "="*50)
print("📊 Processing Summary:")
print(f"Total frames: {frame_count}")
print(f"Total tracks: {len(enhanced_tracker.track_identities)}")

identified = [id for id in enhanced_tracker.track_identities.values() if id != "Unknown"]
unique_identified = set(identified)
print(f"Identified persons: {unique_identified if unique_identified else 'None'}")
print(f"Unknown tracks: {sum(1 for id in enhanced_tracker.track_identities.values() if id == 'Unknown')}")
print("="*50)