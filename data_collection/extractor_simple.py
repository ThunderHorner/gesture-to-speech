import json
import os
import cv2 as cv
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import numpy as np
import uuid

# Initialize the Hands model
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize the Face Mesh model
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2)

# Function to calculate the angle between three points
def calculate_angle(p1, p2, p3):
    a = np.array([p1.x, p1.y, p1.z])
    b = np.array([p2.x, p2.y, p2.z])
    c = np.array([p3.x, p3.y, p3.z])

    # Calculate vectors
    ba = a - b
    bc = c - b

    # Compute cosine of the angle
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cos_angle)

    return np.degrees(angle)

class ExtractSimplifiedVideoData:
    def __init__(self, file_path, label):
        self.file_path = file_path
        self.label = label

    def process_file(self):
        cam = cv.VideoCapture(self.file_path)
        collected_data = []

        while cam.isOpened():
            success, frame = cam.read()
            if not success:
                break

            # Convert the frame from BGR to RGB
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Process the frame for hand and face detection
            hands_detected = hands.process(frame_rgb)
            face_detected = face_mesh.process(frame_rgb)

            # Initialize dictionary to store important distances and angles
            features = {}

            # Hand features (use only specific key points: thumb tip, index finger tip, etc.)
            if hands_detected.multi_hand_landmarks:
                for hand_landmarks in hands_detected.multi_hand_landmarks:
                    # Thumb tip to index finger tip distance
                    thumb_tip = hand_landmarks.landmark[4]
                    index_tip = hand_landmarks.landmark[8]
                    middle_tip = hand_landmarks.landmark[12]

                    thumb_index_dist = calculate_distance(thumb_tip, index_tip)
                    index_middle_dist = calculate_distance(index_tip, middle_tip)

                    # Store distances
                    features['thumb_index_dist'] = thumb_index_dist
                    features['index_middle_dist'] = index_middle_dist

            # Face features (e.g., distance between nose and chin, angle between eyes and nose)
            if face_detected.multi_face_landmarks:
                for face_landmarks in face_detected.multi_face_landmarks:
                    nose = face_landmarks.landmark[1]
                    chin = face_landmarks.landmark[152]
                    left_eye_outer = face_landmarks.landmark[33]
                    right_eye_outer = face_landmarks.landmark[263]

                    # Calculate distances
                    nose_chin_dist = calculate_distance(nose, chin)
                    features['nose_chin_dist'] = nose_chin_dist

                    # Calculate angles (e.g., angle between nose and eyes)
                    eye_nose_angle = calculate_angle(left_eye_outer, nose, right_eye_outer)
                    features['eye_nose_angle'] = eye_nose_angle

            if features:  # Only append if there's valid data
                collected_data.append(features)

        cam.release()
        return collected_data, self.label


if __name__ == '__main__':
    base_path = '/home/thunderhorn/Videos/alphabet'
    directories = os.listdir(base_path)
    for label in directories:
        dir_path = os.path.join(base_path, label)
        dir_files = os.listdir(dir_path)
        for file in dir_files:
            file_path = os.path.join(dir_path, file)
            extracted_data, label = ExtractSimplifiedVideoData(file_path, label).process_file()

            dist_dir = f'/home/thunderhorn/PycharmProjects/gesture-to-speech/training_data_simplified/{label}'
            os.makedirs(dist_dir, exist_ok=True)

            # Save the simplified data to a .json file
            with open(os.path.join(dist_dir, str(uuid.uuid4()) + '.json'), 'w') as f:
                json.dump(extracted_data, f)
