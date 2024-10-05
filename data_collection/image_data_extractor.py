import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
import joblib
import mediapipe as mp

# Load the trained LSTM model and unique labels
model = load_model('gesture_lstm_model.h5')

# Load unique labels from the training data
with open('/home/thunderhorn/PycharmProjects/gesture-to-speech/data_collection/lstm_max_augmented_labelled.json', 'r') as file:
    data = json.load(file)
unique_labels = list(data.keys())

# Initialize MediaPipe Hands and Face Mesh
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# Function to compute Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)

# Function to extract hand and facial landmarks and compute distances (same as training process)
def extract_landmarks_and_distances(image):
    distances = []
    wrist_coords = None
    fingertip_coords = []
    nose_coords = None

    # Process the image for hand landmarks
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        hand_results = hands.process(image)
        if hand_results.multi_hand_landmarks:
            for hand_landmark in hand_results.multi_hand_landmarks:
                needed_data = [
                    0,  # WRIST
                    4,  # THUMB_TIP
                    8,  # INDEX_FINGER_TIP
                    12,  # MIDDLE_FINGER_TIP
                    16,  # RING_FINGER_TIP
                    20]  # PINKY_TIP

                # Extract wrist and finger tips
                wrist_coords = [hand_landmark.landmark[0].x, hand_landmark.landmark[0].y, hand_landmark.landmark[0].z]
                for _i in needed_data[1:]:
                    fingertip = hand_landmark.landmark[_i]
                    fingertip_coords.append([fingertip.x, fingertip.y, fingertip.z])

    # Process the image for facial landmarks (nose)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        face_results = face_mesh.process(image)
        if face_results.multi_face_landmarks:
            for _face_landmarks in face_results.multi_face_landmarks:
                nose = _face_landmarks.landmark[0]  # Nose landmark
                nose_coords = [nose.x, nose.y, nose.z]

    # Calculate distances from wrist to each fingertip
    if wrist_coords and fingertip_coords:
        for fingertip in fingertip_coords:
            dist_from_wrist = euclidean_distance(wrist_coords, fingertip)
            distances.append(dist_from_wrist)

    # Calculate distance from nose to wrist
    if nose_coords and wrist_coords:
        dist_from_nose = euclidean_distance(nose_coords, wrist_coords)
        distances.append(dist_from_nose)

    return distances if distances else None

# Function to make a prediction based on the last 5 frames (like training)
def make_prediction_from_sequence(sequence):
    # Prepare the input sequence as used in training
    input_sequence = np.array(sequence, dtype=np.float32).reshape(1, len(sequence), len(sequence[0]))

    # Predict with the model
    prediction = model.predict(input_sequence)

    # Decode the label
    predicted_label_index = np.argmax(prediction)
    predicted_label = unique_labels[predicted_label_index]
    prediction_accuracy = np.max(prediction)

    return predicted_label, prediction_accuracy

# Function to start the live gesture recognition
def live_gesture_recognition():
    cam = cv2.VideoCapture(0)
    sequence = []
    frame_count = 0
    predicted_gesture, acc = ['', 0.00]
    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally (inverting by x-axis)
        frame = cv2.flip(frame, 1)

        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract landmarks and distances
        distances = extract_landmarks_and_distances(frame_rgb)
        if distances:
            sequence.append(distances)
            frame_count += 1
            print(f"Captured frame {frame_count} for gesture recognition")

        # Once we have 5 frames, make a prediction
        if len(sequence) == 5:
            predicted_gesture, acc = make_prediction_from_sequence(sequence)
            print(f"Predicted Gesture: {predicted_gesture}, Accuracy: {acc * 100:.2f}%")

            # Reset the sequence for the next prediction
            sequence = []
            frame_count = 0

        cv2.putText(frame, f'Gesture: {predicted_gesture}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Accuracy: {acc * 100:.2f}%', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Display the frame with prediction
        cv2.imshow("Live Gesture Recognition", frame)

        # Press 'q' to quit the webcam
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    live_gesture_recognition()