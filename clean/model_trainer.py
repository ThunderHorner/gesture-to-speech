import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').disabled = True
logging.getLogger('mediapipe').disabled = True
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pickle
import cv2
import mediapipe as mp
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
# Initialize MediaPipe Hands and Face Mesh
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# Function to compute Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
from feature_extractor import extract_landmarks
# Function to extract hand and facial landmarks and compute offsets
def _extract_landmarks(image):
    wrist_coords = None
    fingertip_coords = []
    nose_coords = None
    offsets = {}

    # Process the image for hand landmarks
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

                # Extract wrist and finger tips
                wrist_coords = [hand_landmark.landmark[0].x, hand_landmark.landmark[0].y]
                finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                for idx, _i in enumerate(needed_data[1:]):
                    fingertip = hand_landmark.landmark[_i]
                    fingertip_coords.append([fingertip.x, fingertip.y])
                    offsets[f'{finger_names[idx]}_offset'] = [(fingertip.x - wrist_coords[0]) * 10, (fingertip.y - wrist_coords[1]) * 10]

    # Process the image for facial landmarks (nose)
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        face_results = face_mesh.process(image)
        if face_results.multi_face_landmarks:
            for _face_landmarks in face_results.multi_face_landmarks:
                nose = _face_landmarks.landmark[0]  # Nose landmark
                nose_coords = [nose.x, nose.y]

                # Compute offsets from wrist to nose
                if wrist_coords is not None:
                    offsets['Nose_offset'] = [(nose_coords[0] - wrist_coords[0]) * 10, (nose_coords[1] - wrist_coords[1]) * 10]
    return offsets

# Load the training data from the file
def load_training_data(filepath):
    training_data = []
    labels = []
    with open(filepath, 'r') as f:
        for line in f:
            data_entry = json.loads(line.strip().replace("'", '"'))
            frames = data_entry['frames']
            for frame in frames:
                offsets = list(frame.values())
                # Flatten the list of offsets to create a feature vector
                feature_vector = [item for sublist in offsets for item in sublist]
                training_data.append(feature_vector)
                labels.append(data_entry['label'])
    return np.array(training_data, dtype=object), labels

# Encode the labels as integers and create a dictionary
def encode_labels(labels):
    label_to_int = {label: idx for idx, label in enumerate(set(labels))}
    int_to_label = {idx: label for label, idx in label_to_int.items()}
    encoded_labels = [label_to_int[label] for label in labels]
    return np.array(encoded_labels), label_to_int, int_to_label

# Create a predictive model using LSTM
def create_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def create_lstm_model_2(input_shape, num_classes):
    model = Sequential([
        LSTM(32, input_shape=input_shape, return_sequences=True, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(32, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # Update with the path to your training data file
    training_data_file = '/home/thunderhorn/PycharmProjects/gesture-to-speech/training_data/training_data.log'
    model_file = 'gesture_recognition_model2.h5'
    label_map_file = 'label_map.pkl'

    # Load the training data and labels
    X, labels = load_training_data(training_data_file)
    y, label_to_int, int_to_label = encode_labels(labels)

    # Ensure all elements of X have the same length (expected_features_count)
    expected_features_count = len(X[0])  # Assuming the first element has the correct number of features

    # Filter out inconsistent elements
    X_valid = [xi for xi in X if len(xi) == expected_features_count]
    labels_valid = [labels[i] for i in range(len(X)) if len(X[i]) == expected_features_count]
    print(set(labels_valid))

    # Convert the filtered data to a NumPy array
    X_valid = np.array([np.array(xi, dtype=np.float32) for xi in X_valid])

    # Reshape the input data to match the LSTM input requirements (samples, timesteps, features)
    X_valid = X_valid.reshape((X_valid.shape[0], 1, X_valid.shape[1]))
    X = X_valid

    # Update the labels to match the filtered data
    y_valid = [label_to_int[label] for label in labels_valid]

    # Convert labels to a NumPy array
    y_valid = np.array(y_valid)

    # Create the LSTM model
    input_shape = (X_valid.shape[1], X_valid.shape[2])
    num_classes = len(label_to_int)

    if input("Enter 'train' to train a new model, or any other key to use existing model: ").lower() == 'train':
        model = create_lstm_model(input_shape, num_classes)

        # Train the model
        model.fit(X, y, epochs=50, batch_size=4, validation_split=0.2)

        # Save the trained model
        model.save(model_file)

        # Save the label mapping
        with open(label_map_file, 'wb') as f:
            pickle.dump((label_to_int, int_to_label), f)

        print(f"Model and label mapping saved to {model_file} and {label_map_file}")
    else:
        from tensorflow.keras.models import Sequential, load_model
        # Load the trained model for prediction
        model = load_model(model_file)

        # Load the label mapping
        with open(label_map_file, 'rb') as f:
            label_to_int, int_to_label = pickle.load(f)

        print(f"Model and label mapping loaded from {model_file} and {label_map_file}")

    # Capture video from webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2))  # Resize the frame to 2x size
        if not ret:
            print("Reached the end of the video.")
            break

        # Convert frame to RGB (required by MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract landmarks and compute offsets
        _, offsets = extract_landmarks(frame_rgb)

        # If offsets are detected, make a prediction
        if offsets:
            feature_vector = [item for sublist in offsets.values() for item in sublist]
            feature_vector = np.array(feature_vector).reshape(1, 1, -1)
            prediction = model.predict(feature_vector)
            prediction_accuracy = np.max(prediction)
            if (prediction_accuracy * 100) > 95:
                predicted_label = int_to_label[np.argmax(prediction.flatten())]

                # Display the predicted label on the frame
                cv2.putText(frame, f'Predicted: {predicted_label} {prediction_accuracy * 100:.2f}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Display the frame
        cv2.imshow('Gesture Recognition', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()