import cv2
import os
import datetime
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import shutil
import json
from ultralytics import YOLO

# from mtcnn import MTCNN
VIDEO_DIR = 'videos'

DATA_EXTRACT_DIR = 'data_extract'

TEMP_FACES_DIR = 'temp_faces'
GROUPED_FACES_DIR = 'temp_grouped_faces'
FINAL_GROUPED_FACES_DIR = 'final_grouped_faces'
PROCESSED_FACES_DIR = 'processed_faces'
DATABASE_DIR = 'database'

if not os.path.exists(DATA_EXTRACT_DIR):
    os.makedirs(DATA_EXTRACT_DIR)

for dir_path in [TEMP_FACES_DIR, GROUPED_FACES_DIR, FINAL_GROUPED_FACES_DIR, PROCESSED_FACES_DIR, DATABASE_DIR]:
    data_path = os.path.join(DATA_EXTRACT_DIR, dir_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)


# step1: extract faces from videos
def extract_faces_from_videos():
    print("extract faces from videos")

    # detector = MTCNN()
    model = YOLO('./models/yolov12n-face.pt')
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]
    print(f"count videos: {len(video_files)}")

    face_id_counter = 0
    face_metadata = []

    for video_file in tqdm(video_files, desc="videos processing"):
        video_path = os.path.join(VIDEO_DIR, video_file)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # extract in every 600 frame
            if frame_count % 600 == 0:
                try:
                    faces = model(frame)

                    for face in faces:
                        boxes = face.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                        scores = face.boxes.conf.cpu().numpy()
                        # MTCNN
                        # x, y, w, h = face['box']
                        # confidence = face['confidence']
                        for box, score in zip(boxes, scores):
                            x1, y1, x2, y2 = map(int, box)
                            face_area = (x2 - x1) * (y2 - y1)
                            if face_area < 450:
                                continue
                            elif face_area < 800:
                                min_confidence = 0.4
                            elif face_area < 1200:
                                min_confidence = 0.5
                            elif face_area < 1500:
                                min_confidence = 0.65
                            elif face_area < 2500:
                                min_confidence = 0.75
                            else:
                                min_confidence = 0.8

                            if score > min_confidence:
                                face_img = frame[y1:y2, x1:x2]

                                face_filename = os.path.join('data_extract', TEMP_FACES_DIR,
                                                             f'face_{face_id_counter}.jpg')
                                cv2.imwrite(face_filename, face_img)
                                time_sec = frame_count / fps

                                face_metadata.append({
                                    'id': int(face_id_counter),
                                    'video': str(video_file),
                                    'time': str(datetime.timedelta(seconds=int(time_sec))),
                                    'frame': str(frame_count),
                                    'box': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                    'confidence': float(score),
                                    'filename': str(face_filename)
                                })

                                face_id_counter += 1

                except Exception as e:
                    print(f"Error in frame process fram_id:{frame_count} error: {e}")

            frame_count += 1

        cap.release()

    print(f"{face_id_counter} faces were successfully extracted")
    json.dump(face_metadata, open(f'{DATA_EXTRACT_DIR}/face_metadata.json', 'w'))
    return face_metadata


def main():
    print("start processing on videos")

    face_metadata = extract_faces_from_videos()

    if len(face_metadata) == 0:
        print("no face detected")
        return


if __name__ == "__main__":
    main()
