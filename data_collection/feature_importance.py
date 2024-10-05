import json
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


def load_and_preprocess_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    X = []
    y = []

    for label, gestures in data.items():
        for gesture in gestures:
            # Ensure all gestures have the same length by padding or truncating
            fixed_length = 126  # Adjust this based on your data
            if len(gesture) < fixed_length:
                gesture += [0] * (fixed_length - len(gesture))
            elif len(gesture) > fixed_length:
                gesture = gesture[:fixed_length]

            X.append(gesture)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_encoded, label_encoder


def train_and_evaluate_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Print classification report
    print(classification_report(y_test, y_pred))

    return clf, accuracy


def plot_feature_importance(clf, label_encoder):
    feature_importance = clf.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    print(np.arange(sorted_idx.shape[0])[-30:])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(pos[-20:], feature_importance[sorted_idx][-20:], align='center')
    ax.set_yticks(pos[-20:])
    ax.set_yticklabels((sorted_idx[-20:]).astype(str))
    ax.set_title('Top 20 Most Important Features')
    ax.set_xlabel('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Feature importance plot saved as 'feature_importance.png'")


if __name__ == '__main__':
    json_data_path = 'lstm_max.json'

    # Load and preprocess data
    X, y, label_encoder = load_and_preprocess_data(json_data_path)

    # Train and evaluate the model
    clf, accuracy = train_and_evaluate_model(X, y)

    print(f"Model accuracy: {accuracy * 100:.2f}%")

    # Plot feature importance
    plot_feature_importance(clf, label_encoder)

    # Save label encoder
    with open('label_encoder.json', 'w') as f:
        json.dump(label_encoder.classes_.tolist(), f)