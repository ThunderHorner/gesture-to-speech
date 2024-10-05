import json
from decimal import Decimal

import numpy as np
from keras.src.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os


# Step 1: Load the extracted gesture data from the JSON file
def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.loads(f.read())
    return data


# Step 2: Preprocess the data for LSTM model
def preprocess_data(data):
    X = []
    y = []

    # Iterate over labels and gestures in the JSON data
    for label, gestures in data.items():
        for gesture in gestures:

            X.append([float(Decimal(i).quantize(Decimal('0.01'))) for i in gesture])
            y.append(label)
            print(label)

    # Convert X and y to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    # Pad the sequences to ensure consistent length
    max_sequence_length = max(len(seq) for seq in X)
    X_padded = pad_sequences(X, maxlen=max_sequence_length, padding='post')
    X_padded = X_padded / np.max(X_padded)  # Normalize the feature values


    # Reshape X_padded if it is missing the feature dimension
    if len(X_padded.shape) == 2:  # If the shape is (batch_size, sequence_length)
        X_padded = np.expand_dims(X_padded, axis=-1)  # Add a new axis for the feature dimension

    # Encode the labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    # Optionally, return label encoder for inverse transformation
    return X_padded, y_encoded, max_sequence_length, label_encoder


# Step 3: Define the LSTM model architecture
# Modify the LSTM model architecture
def _create_lstm_model(input_shape, num_classes):
    model = Sequential()

    # Add more LSTM layers and units
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.5))

    # Add a dense layer before the output layer for more learning capacity
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer with softmax for classification
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model with a lower learning rate
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# from keras.optimizers import Adam
# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Dropout
#
def create_lstm_model(input_shape, num_classes):
    model = Sequential()

    # Increase LSTM units and remove dropout layers to encourage overfitting
    model.add(LSTM(512, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512, return_sequences=False))

    # Add dense layers with many units
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile with a lower learning rate for fine-tuned learning
    optimizer = Adam(learning_rate=1.0)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model



# Step 4: Train the LSTM model
def train_model(model, X, y, model_save_path):
    # Remove early stopping and train for more epochs
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    # Train the model
    history = model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2,
                        callbacks=[checkpoint])

    return history

# Step 5: Main execution for loading, preprocessing, training, and saving the model
if __name__ == '__main__':
    # Path to the JSON file containing the preprocessed gesture data
    json_data_path = 'lstm_max.json'

    # Load the data
    data = load_data(json_data_path)

    # Preprocess the data
    X_padded, y_encoded, max_sequence_length, label_encoder = preprocess_data(data)
    num_classes = len(np.unique(y_encoded))

    # Create the LSTM model
    input_shape = (max_sequence_length, X_padded.shape[-1])
    model = create_lstm_model(input_shape, num_classes)

    # Print model summary
    model.summary()

    # Path to save the trained model
    model_save_path = 'gesture_lstm_modelf.keras'

    # Train the LSTM model
    history = train_model(model, X_padded, y_encoded, model_save_path)

    # Print final accuracy on training and validation sets
    loss, accuracy = model.evaluate(X_padded, y_encoded)
    print(f"Model training complete with accuracy: {accuracy * 100:.2f}%")

    # Save the label encoder for future use
    with open('label_encoder.json', 'w') as f:
        json.dump(label_encoder.classes_.tolist(), f)
