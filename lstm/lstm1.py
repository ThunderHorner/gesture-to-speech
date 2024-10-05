import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import numpy as np
import json


# Load your JSON data
def load_data(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data


# Convert JSON data into features and labels
def prepare_data(data):
    features, labels = [], []
    for label, frames in data.items():
        features.append(frames)
        labels.append(label)

    # Convert labels to numerical values
    unique_labels = list(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    labels = [label_to_idx[label] for label in labels]

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32), unique_labels


if __name__ == "__main__":
    # Load and prepare the data
    json_file_path = '/home/thunderhorn/PycharmProjects/gesture-to-speech/data_collection/lstm_max_augmented_labelled.json'  # Replace with your JSON file path
    data = load_data(json_file_path)
    X, y, unique_labels = prepare_data(data)

    # Reshape X for LSTM (samples, timesteps, features)
    X = np.array([np.array(sequence) for sequence in X])  # Assuming each frame is already an array of features

    # LSTM model
    timesteps = X.shape[1]  # 5 frames per sequence
    features = X.shape[2]  # Features per frame
    num_classes = len(unique_labels)

    model = Sequential([
        Input(shape=(timesteps, features)),
        LSTM(50, activation='relu', return_sequences=False),
        Dense(num_classes, activation='softmax')
    ])

    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X, y, epochs=100, batch_size=4, validation_split=0.2)

    # Save the model
    model.save('gesture_lstm_model.h5')