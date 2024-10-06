import os

from tensorboard.compat.tensorflow_stub import string

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging

logging.getLogger('tensorflow').disabled = True
logging.getLogger('mediapipe').disabled = True
import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np
import warnings

warnings.filterwarnings('ignore')
from feature_extractor import extract_landmarks

# Initialize MediaPipe Hands and Face Mesh
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh


# Function to compute Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# Function to record gestures and save training data
def record_gesture_training_data(cap, num_entries=30, frames_count=20, words=None):
    if words is None:
        words = ['hello', 'world', 'sign', 'language']  # Default words if none provided

    training_data = []
    word_index = 0
    activation_count = 0
    recording = False
    recorded_frames = []

    while word_index < len(words):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2))  # Resize the frame to 2x size
        if not ret:
            print("Reached the end of the video.")
            break

        # Convert frame to RGB (required by MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract landmarks and draw them on the frame
        frame_with_landmarks, offsets = extract_landmarks(frame_rgb)

        # Overlay text to show current word, activation count, and recording status
        cv2.putText(frame_with_landmarks, f"Word: {words[word_index]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
        cv2.putText(frame_with_landmarks, f"Activations: {activation_count}/{frames_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if recording:
            cv2.putText(frame_with_landmarks, "Recording", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame_with_landmarks, "Press SPACE to start", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2)

        # Display the frame with landmarks
        cv2.imshow('Recording Gesture', frame_with_landmarks)

        # Handle key events
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Press space to start/stop recording
            if not recording:
                recording = True
                recorded_frames = []
            else:
                recording = False
                if recorded_frames:
                    # Save the gesture data with label
                    training_data.append({'label': words[word_index], 'frames': recorded_frames})
                    activation_count += 1
                    print(f"Saved activation {activation_count} for word '{words[word_index]}'")

                    if activation_count >= frames_count:
                        word_index += 1
                        activation_count = 0
                        if word_index < len(words):
                            print(f"Moving to next word: {words[word_index]}")
                        else:
                            print("Finished recording all words")
                            break

        elif key == ord('q'):  # Press 'q' to quit
            break

        # Record frames if recording is active
        if recording:
            recorded_frames.append(offsets)

    # Save training data to file
    if training_data:
        with open('/home/thunderhorn/PycharmProjects/gesture-to-speech/training_data/training_data.log', 'a') as f:
            for data in training_data:
                f.write(f"{data}\n")

        print("Training data saved successfully.")


if __name__ == '__main__':
    # Update with the path to your video file or use 0 for webcam
    video_path = '/home/thunderhorn/PycharmProjects/gesture-to-speech/videos/your_video.mp4'  # Update path or set to 0 for webcam
    cap = cv2.VideoCapture(0)

    # Check if video capture is successful
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    import string
    # Define the array of words to be presented
    words_to_present = ['is']

    # Call the function with the array of words
    record_gesture_training_data(cap, frames_count=20, words=words_to_present)

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()