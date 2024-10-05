import cv2 as cv
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import numpy as np
import joblib
import time

# Load the trained model and scaler
model_path = '/home/thunderhorn/PycharmProjects/gesture-to-speech/model_generator/gesture_model_knn_simplified.pkl'
scaler_path = '/home/thunderhorn/PycharmProjects/gesture-to-speech/model_generator/scaler_knn_simplified.pkl'
knn = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Initialize the Hands model
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize the Face Mesh model
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5)


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


# Function to extract features from the frame
def extract_features(frame_rgb):
    features = {}

    # Process the frame for hand and face detection
    hands_detected = hands.process(frame_rgb)
    face_detected = face_mesh.process(frame_rgb)

    # Hand features
    if hands_detected.multi_hand_landmarks:
        for hand_landmarks in hands_detected.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]

            # Calculate distances
            features['thumb_index_dist'] = calculate_distance(thumb_tip, index_tip)
            features['index_middle_dist'] = calculate_distance(index_tip, middle_tip)

    # Face features
    if face_detected.multi_face_landmarks and hands_detected.multi_hand_landmarks:
        for face_landmarks in face_detected.multi_face_landmarks:
            nose = face_landmarks.landmark[1]
            chin = face_landmarks.landmark[152]
            left_eye_outer = face_landmarks.landmark[33]
            right_eye_outer = face_landmarks.landmark[263]

            # Calculate distances and angles
            features['nose_chin_dist'] = calculate_distance(nose, chin)
            features['eye_nose_angle'] = calculate_angle(left_eye_outer, nose, right_eye_outer)
        return features
    return None


# Function to make a prediction based on aggregated features (average of all frames)
def make_prediction_from_sequence(features_sequence):
    # List all required keys
    required_keys = [
        'thumb_index_dist', 'index_middle_dist',
        'nose_chin_dist', 'eye_nose_angle'
    ]

    # Average the features over the sequence
    averaged_features = {key: np.mean([frame[key] for frame in features_sequence if key in frame]) for key in required_keys}

    # Ensure that all keys are present and fill missing ones with 0
    feature_vector = np.array([averaged_features.get(key, 0) for key in required_keys]).reshape(1, -1)

    # Scale the feature vector
    feature_vector_scaled = scaler.transform(feature_vector)

    # Make the prediction
    prediction = knn.predict(feature_vector_scaled)
    prediction_prob = knn.predict_proba(feature_vector_scaled)

    # Get the letter and the accuracy (probability)
    letter = prediction[0]
    from pprint import pprint
    pprint(prediction)
    accuracy = np.max(prediction_prob)

    return letter, accuracy


# Function to start the webcam and make predictions every 2 seconds
def run_webcam_test():
    cam = cv.VideoCapture(0)
    features_sequence = []  # To store features of the sequence
    start_time = time.time()  # Timer to keep track of 2-second intervals
    interval = 2  # Set to 2 seconds

    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally (inverting by x-axis)
        frame = cv.flip(frame, 1)

        # Convert the frame from BGR to RGB
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Extract features from the current frame
        extracted_features = extract_features(frame_rgb)
        if extracted_features:
            features_sequence.append(extracted_features)

        # Check if 2 seconds have passed
        if time.time() - start_time >= interval:
            if features_sequence:
                # Make a prediction using the averaged features of the sequence
                letter, accuracy = make_prediction_from_sequence(features_sequence)

                # Display the letter and accuracy on the frame
                cv.putText(frame, f'Letter: {letter}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv.putText(frame, f'Accuracy: {accuracy * 100:.2f}%', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Reset timer and feature collection
            start_time = time.time()
            features_sequence = []

        # Display the frame
        cv.imshow("Gesture Recognition", frame)

        # Press 'q' to quit the webcam
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    run_webcam_test()
