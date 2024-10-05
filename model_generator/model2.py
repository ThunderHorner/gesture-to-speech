import os
import json
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib


# Step 1: Load Simplified Gesture Data from .json Files
def load_simplified_gesture_data(data_dir):
    gestures = []
    labels = []

    # Iterate over the label directories (e.g., A, B, C...)
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue

        # Iterate over all the .json files in each directory
        for file_name in os.listdir(label_dir):
            if not file_name.endswith('.json'):
                continue  # Skip non-json files
            file_path = os.path.join(label_dir, file_name)

            # Open and load the simplified JSON data
            with open(file_path, 'r') as f:
                try:
                    gesture_data = json.load(f)
                    for frame in gesture_data:  # Each frame contains feature vectors
                        gestures.append(frame)  # Append each frame's feature vector
                        labels.append(label)  # The directory name is used as the label
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file {file_path}: {e}")
                except Exception as e:
                    print(f"Unexpected error loading file {file_path}: {e}")

    return gestures, labels


# Step 2: Ensure consistent feature vector length by filling missing features
def ensure_consistent_features(gesture, required_keys):
    """ Ensure that each gesture contains all required features. """
    return [gesture.get(key, 0) for key in required_keys]


# Step 3: Preprocess the Data (Convert lists to arrays, normalize)
def preprocess_simplified_data(data_dir):
    gestures, labels = load_simplified_gesture_data(data_dir)

    # List all possible keys (features) that each frame should have
    # These are the feature names we expect in each JSON object
    required_keys = [
        'thumb_index_dist', 'index_middle_dist',
        'nose_chin_dist', 'eye_nose_angle'
        # Add other feature names here if needed
    ]

    # Flatten each gesture data and ensure consistency of features
    X = np.array([ensure_consistent_features(gesture, required_keys) for gesture in gestures])
    Y = np.array(labels)

    print(f'Number of frames: {len(X)}')
    print(f'Number of labels: {len(Y)}')

    return X, Y


# Step 4: Train the KNN model with cross-validation
def train_knn_model(X, Y):
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize KNN model
    knn = KNeighborsClassifier(n_neighbors=3)

    # Use Stratified K-Fold to maintain label distribution in folds
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform cross-validation
    cv_scores = cross_val_score(knn, X_scaled, Y, cv=skf, scoring='accuracy')
    print(f'Cross-validation Accuracy scores: {cv_scores}')
    print(f'Average Cross-validation Accuracy: {np.mean(cv_scores) * 100:.2f}%')

    # Fit the model on the entire dataset
    knn.fit(X_scaled, Y)

    return knn, scaler


# Step 5: Evaluate the model with a train-test split
def evaluate_model(knn, scaler, X, Y):
    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

    # Scale the data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the KNN model
    knn.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = knn.predict(X_test_scaled)

    # Evaluate model accuracy
    accuracy = np.mean(y_pred == y_test) * 100
    print(f'Test Accuracy: {accuracy:.2f}%')

    # Detailed classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return accuracy


# Step 6: Save the trained model and scaler
def save_model(knn, scaler, model_path, scaler_path):
    joblib.dump(knn, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")


# Step 7: Load a previously saved model and scaler
def load_model_and_scaler(model_path, scaler_path):
    knn = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return knn, scaler


# Main execution
if __name__ == "__main__":
    # Define your data directory where simplified .json files are stored
    data_dir = '/home/thunderhorn/PycharmProjects/gesture-to-speech/training_data_simplified/'
    model_path = 'gesture_model_knn_simplified.pkl'
    scaler_path = 'scaler_knn_simplified.pkl'

    # Preprocess the simplified data
    X, Y = preprocess_simplified_data(data_dir)

    # Train the KNN model with cross-validation
    knn, scaler = train_knn_model(X, Y)

    # Evaluate the model on a train-test split
    evaluate_model(knn, scaler, X, Y)

    # Save the trained model and scaler
    save_model(knn, scaler, model_path, scaler_path)

    # (Optional) Load the model and scaler for future use
    # loaded_knn, loaded_scaler = load_model_and_scaler(model_path, scaler_path)
