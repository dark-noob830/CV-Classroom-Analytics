# main.py (Final corrected version)
import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import supervision as sv
from torchvision import transforms

# Image transformations
transform = transforms.Compose([
    transforms.Resize((160, 160)),  # Expected size for ResNet
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
try:
    database = torch.load('data_extract/person_medoids.pt', map_location=device)
    print("✅ Student database loaded successfully.")
    print("Known identities:", list(database.keys()))
except FileNotFoundError:
    print("❌ Error: 'person_medoids.pt' not found. Please run 'create_database.py' first.")
    exit()

# Initialize ByteTrack tracker
tracker = sv.ByteTrack()

# Dictionary to store recognized identity for each track_id
track_identities = {}

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
video_path = 'videos/Team_3.mp4'
video_capture = cv2.VideoCapture(video_path)

if not video_capture.isOpened():
    print(f"❌ Error: Could not open video file {video_path}")
    exit()

print("Processing video... Press 'q' to quit.")

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

    # ======== Main change to fix ValueError ========
    labels = []
    if tracked_detections.tracker_id is not None:
        for xyxy, track_id in zip(tracked_detections.xyxy, tracked_detections.tracker_id):
            if track_id not in track_identities:
                x1, y1, x2, y2 = map(int, xyxy)
                face_crop_pil = Image.fromarray(frame_rgb[y1:y2, x1:x2])

                with torch.no_grad():
                    face_tensor = transform(face_crop_pil).to(device)
                    embedding = resnet(face_tensor.unsqueeze(0)).squeeze()

                identity, _ = find_identity(embedding, database, RECOGNITION_THRESHOLD)
                track_identities[track_id] = identity

            identity = track_identities[track_id]
            labels.append(f"ID:{track_id} {identity}")
    # =======================================================

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

    cv2.imshow('Classroom Attendance Analysis - Computer Vision Project', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
print("✅ Processing finished.")
