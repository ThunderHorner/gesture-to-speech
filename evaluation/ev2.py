import json
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the trained Bi-directional LSTM model
model_path = '/path_to_your_model/bidirectional_lstm_model.h5'
bi_lstm_model = tf.keras.models.load_model(model_path)

# Load the JSON data file
with open('lstm_max.json', 'r') as f:
    data = json.load(f)

# Prepare the dataset
def prepare_data(data):
    X = []
    y = []
    for label, sequences in data.items():
        for sequence in sequences:
            X.append(sequence)
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y

# Prepare features (X) and labels (y)
X, y = prepare_data(data)

# Reshape X to have a third dimension representing features (if needed)
X = np.expand_dims(X, axis=-1)  # Shape: (samples, timesteps, features)

# Encode the labels (letters) as integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# One-hot encode the labels
y_categorical = to_categorical(y_encoded)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Scale the input data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# Evaluate the model
loss, accuracy = bi_lstm_model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%")

# Make predictions on test data
predictions = bi_lstm_model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

# Decode the predicted and true labels back to original letters
predicted_letters = label_encoder.inverse_transform(predicted_labels)
true_letters = label_encoder.inverse_transform(true_labels)

# Display results
for i in range(len(predicted_letters)):
    print(f"True Letter: {true_letters[i]}, Predicted Letter: {predicted_letters[i]}")
