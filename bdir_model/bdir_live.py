import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import mediapipe as mp

# Load the trained Bi-directional LSTM model and LabelEncoder
model = load_model('bi_lstm_model.h5')
label_encoder = joblib.load('label_encoder.pkl')

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

    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
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

                wrist_coords = [hand_landmark.landmark[0].x, hand_landmark.landmark[0].y, hand_landmark.landmark[0].z]
                for _i in needed_data[1:]:
                    fingertip = hand_landmark.landmark[_i]
                    fingertip_coords.append([fingertip.x, fingertip.y, fingertip.z])

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        face_results = face_mesh.process(image)
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                nose = face_landmarks.landmark[1]  # Nose landmark
                nose_coords = [nose.x, nose.y, nose.z]

    if wrist_coords and fingertip_coords:
        for fingertip in fingertip_coords:
            dist_from_wrist = euclidean_distance(wrist_coords, fingertip)
            distances.append(dist_from_wrist)

    if nose_coords and wrist_coords:
        dist_from_nose = euclidean_distance(nose_coords, wrist_coords)
        distances.append(dist_from_nose)

    return distances if distances else None


# Function to make a prediction based on the last 5 frames (like training)
def make_prediction_from_sequence(sequence):
    # Flatten the sequence of 5 frames (just like in training)
    input_sequence = np.array(sequence).flatten().tolist()

    # Reshape to (1, number_of_features)
    input_sequence = np.array(input_sequence).reshape(1, -1)

    # Predict with the model
    prediction = model.predict(input_sequence)

    # Decode the label
    predicted_label_index = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_label_index])
    prediction_accuracy = np.max(prediction)

    return predicted_label[0], prediction_accuracy


# Function to start the live gesture recognition
def live_gesture_recognition():
    cam = cv2.VideoCapture(0)
    sequence = []
    frame_count = 0
    frame_interval = 1  # Capture every 30th frame
    frame_skip_count = 0  # To track frames to skip
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

        # Capture every 30th frame
        # if frame_skip_count % frame_interval == 0:
        if True:
            distances = extract_landmarks_and_distances(frame_rgb)
            if distances:
                sequence.append(distances)
                frame_count += 1
                print(f"Captured frame {frame_count} for gesture recognition")

            # Once we have 5 frames, make a prediction
            if len(sequence) == 5:
                try:
                    predicted_gesture, acc = make_prediction_from_sequence(sequence)
                    print(f"Predicted Gesture: {predicted_gesture}, Accuracy: {acc * 100:.2f}%")

                    # Display the gesture and accuracy on the frame

                except ValueError as ve:
                    print(f"Error: {ve}")

                # Reset the sequence for the next prediction
                sequence = []
                frame_count = 0

        frame_skip_count += 1
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
