import json
import numpy as np
from tensorflow.keras.layers import GlobalAveragePooling1D
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Bidirectional, MultiHeadAttention, LayerNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Function to create the LSTM model
def create_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Function to create the GRU model
def create_gru_model(input_shape, num_classes):
    model = Sequential()
    model.add(GRU(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(GRU(64, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Function to create the 1D CNN model
# Function to create the 1D CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential()
    # Reduce the kernel size to 1 to avoid negative dimensions
    model.add(Conv1D(64, kernel_size=1, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(128, kernel_size=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


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

# Function to create the MLP model
def create_mlp_model(input_shape, num_classes):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Function to create the Transformer model
def create_transformer_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Multi-head self-attention layer
    attention_output = MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
    attention_output = Dropout(0.3)(attention_output)

    # Global average pooling to collapse the time dimension
    pooled_output = GlobalAveragePooling1D()(attention_output)

    # Fully connected feed-forward network
    feedforward_output = Dense(64, activation='relu')(pooled_output)
    feedforward_output = Dense(num_classes, activation='softmax')(feedforward_output)

    model = Model(inputs=inputs, outputs=feedforward_output)
    return model

# Function to prepare the dataset
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

# Function to compile, train and evaluate the model
def compile_and_train_model(model, X_train, y_train, X_test, y_test, num_classes):
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=300, batch_size=32, validation_split=0.1, verbose=1)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    return loss, accuracy

if __name__ == '__main__':
    # Load the dataset from your JSON file
    with open('lstm_max.json', 'r') as f:
        data = json.load(f)

    # Prepare the dataset
    X, y = prepare_data(data)

    # Reshape X to have a third dimension representing features (if needed)
    X = np.expand_dims(X, axis=-1)  # Shape: (samples, timesteps, features)

    # Encode the labels (letters) as integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # One-hot encode the labels
    y_categorical = to_categorical(y_encoded)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.1, random_state=42)

    # Define the input shape (timesteps, features)
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Define the number of output classes
    num_classes = len(label_encoder.classes_)

    # List of models to evaluate
    models = {
        'LSTM': create_lstm_model(input_shape, num_classes),
        'GRU': create_gru_model(input_shape, num_classes),
        '1D CNN': create_cnn_model(input_shape, num_classes),
        'Bi-directional LSTM': create_bi_lstm_model(input_shape, num_classes),
        'MLP': create_mlp_model(input_shape, num_classes),
        'Transformer': create_transformer_model(input_shape, num_classes)
    }
    with open('result.txt', 'w') as f:
        f.write('')
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"Training and evaluating {model_name}...")
        loss, accuracy = compile_and_train_model(model, X_train, y_train, X_test, y_test, num_classes)
        with open('result.txt', 'a') as f:
            f.write(f"{model_name} - Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}\n")
            print(f"{model_name} - Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}\n")
