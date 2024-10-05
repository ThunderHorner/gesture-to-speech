import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib


def create_bi_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Bidirectional(LSTM(32, return_sequences=False)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    return model


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


def compile_and_train_model(model, X_train, y_train, X_val, y_val, num_classes):
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy')

    history = model.fit(
        X_train, y_train,
        epochs=3000,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    return history


if __name__ == '__main__':
    with open('lstm_max_augmented.json', 'r') as f:
        data = json.load(f)

    X, y = prepare_data(data)
    X = np.expand_dims(X, axis=-1)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    with open('label_encoder.pkl', 'wb') as le_file:
        joblib.dump(label_encoder, le_file)

    y_categorical = to_categorical(y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(label_encoder.classes_)

    bi_lstm_model = create_bi_lstm_model(input_shape, num_classes)

    history = compile_and_train_model(bi_lstm_model, X_train, y_train, X_val, y_val, num_classes)

    # Evaluate the model using the best weights
    bi_lstm_model.load_weights('best_model.keras')
    loss, accuracy = bi_lstm_model.evaluate(X_test, y_test, verbose=1)
    print(f"Bi-directional LSTM - Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    bi_lstm_model.save('optimized_bi_lstm_model.keras')
    print("Optimized Bi-directional LSTM model saved as 'optimized_bi_lstm_model.h5'")