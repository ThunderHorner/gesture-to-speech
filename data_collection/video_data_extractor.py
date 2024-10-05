import os
import json
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib


# Step 1: Load Gesture Data from .json Files
def load_gesture_data(data_dir):
    gestures = []
    labels = []

    # Iterate over the label directories (e.g., A, B, C...)
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue

        # Iterate over all the .json files in each directory
        for file_name in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file_name)

            # Open and load the JSON data
            with open(file_path, 'r') as f:
                gesture_data = json.load(f)
                gestures.append(gesture_data)
                labels.append(label)  # Use the directory name as the label

    return gestures, labels


# Step 2: Preprocess Gesture Data (Flatten it and ensure consistent shape)
def flatten_gesture(gesture):
    flattened_gesture = []

    # Ensure we have consistent length by filling missing parts with zeros
    def extract_landmarks(key, size=63):
        if key in gesture:
            return gesture[key]
        else:
            return [0] * size  # Fill missing hand/face data with zeros

    # Flatten the gesture data
    flattened_gesture.extend(extract_landmarks('h1'))
    flattened_gesture.extend(extract_landmarks('h2'))
    flattened_gesture.extend(extract_landmarks('nose', 3))
    flattened_gesture.extend(extract_landmarks('chin', 3))
    flattened_gesture.extend(extract_landmarks('leye', 3))
    flattened_gesture.extend(extract_landmarks('reye', 3))
    flattened_gesture.extend(extract_landmarks('lear', 3))
    flattened_gesture.extend(extract_landmarks('rear', 3))

    return flattened_gesture


# Step 3: Load and preprocess the data
def preprocess_data(data_dir):
    gestures, labels = load_gesture_data(data_dir)

    # Flatten each gesture data and normalize if needed
    X = np.array([flatten_gesture(gesture) for gesture in gestures])
    Y = np.array(labels)

    return X, Y


# Step 4: Train the KNN model
def train_knn_model(X, Y):
    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Initialize and train KNN model
    knn = KNeighborsClassifier(n_neighbors=3)  # You can change n_neighbors to tune performance
    knn.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'KNN Accuracy: {accuracy * 100:.2f}%')

    return knn


# Step 5: Save the trained model
def save_model(clf, model_path):
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")


# Step 6: Load a previously saved model
def load_model(model_path):
    return joblib.load(model_path)


# Step 7: Cross-validate the KNN model
def cross_validate_knn(X, Y):
    knn = KNeighborsClassifier(n_neighbors=3)
    scores = cross_val_score(knn, X, Y, cv=5)
    print(f'Cross-validation scores: {scores}')
    print(f'Average accuracy: {np.mean(scores) * 100:.2f}%')


# Main execution
if __name__ == "__main__":
    # Define your data directory where .json files are stored
    data_dir = '/home/thunderhorn/PycharmProjects/gesture-to-speech/training_data/'
    model_path = 'gesture_knn_model.pkl'

    # Preprocess the data
    X, Y = preprocess_data(data_dir)
    print(f'Number of gestures: {len(X)}')
    print(f'Number of labels: {len(Y)}')

    # Train the KNN model
    knn_model = train_knn_model(X, Y)

    # Save the trained KNN model
    save_model(knn_model, model_path)

    # Cross-validate the KNN model
    cross_validate_knn(X, Y)

    # (Optional) Load the model for future predictions
    # loaded_clf = load_model(model_path)
