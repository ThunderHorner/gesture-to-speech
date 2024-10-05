import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Function to create the Bi-directional LSTM model
def create_bi_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Function to prepare the dataset
def prepare_data(data):
    X = []
    y = []
    for label, sequences in data.items():
        for sequence in sequences:
            X.append(sequence)
            y.append(label)
    try:
        X = np.array(X)
    except Exception as e:
        print(y)
        raise (e)
    y = np.array(y)
    return X, y

# Function to compile, train, and evaluate the model
def compile_and_train_model(model, X_train, y_train, X_test, y_test, num_classes):
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=300, batch_size=50, validation_split=0.2, verbose=1)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    return loss, accuracy

if __name__ == '__main__':
    # Load the dataset from your JSON file
    with open('/home/thunderhorn/PycharmProjects/gesture-to-speech/data_collection/lstm_max_augmented_labelled_o.json', 'r') as f:
        data = json.load(f)

    # Prepare the dataset
    X, y = prepare_data(data)

    # Reshape X to have a third dimension representing features
    X = np.expand_dims(X, axis=-1)  # Shape: (samples, timesteps, features)

    # Encode the labels (letters) as integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Save the label encoder for later use
    with open('label_encoder.pkl', 'wb') as le_file:
        joblib.dump(label_encoder, le_file)

    # One-hot encode the labels
    y_categorical = to_categorical(y_encoded)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    # Define the input shape (timesteps, features)
    input_shape = (X_train.shape[0], X_train.shape[1])
    print(X_train.shape[1])
    print(X_train.shape[0])
    input()
    # Define the number of output classes
    num_classes = len(label_encoder.classes_)

    # Create the Bi-directional LSTM model
    bi_lstm_model = create_bi_lstm_model(input_shape, num_classes)

    # Train and evaluate the model
    loss, accuracy = compile_and_train_model(bi_lstm_model, X_train, y_train, X_test, y_test, num_classes)

    # Print results
    print(f"Bi-directional LSTM - Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Save the trained Bi-directional LSTM model
    bi_lstm_model.save('bi_lstm_model.h5')
    print("Bi-directional LSTM model saved as 'bi_lstm_model.h5'")
