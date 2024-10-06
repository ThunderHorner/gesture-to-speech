import os
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

# Initialize MediaPipe Hands and Face Mesh
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# Function to compute Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Function to extract hand and facial landmarks and compute offsets
def extract_landmarks(image):
    wrist_coords = None
    fingertip_coords = []
    nose_coords = None
    offsets = {}

    # Process the image for hand landmarks
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        hand_results = hands.process(image)
        if hand_results.multi_hand_landmarks:
            for hand_landmark in hand_results.multi_hand_landmarks:
                needed_data = [
                    0,  # WRIST
                    4,  # THUMB_TIP
                    8,  # INDEX_FINGER_TIP
                    12,  # MIDDLE_FINGER_TIP
                    16,  # RING_FINGER_TIP
                    20]  # PINKY_TIP

                # Extract wrist and finger tips
                wrist_coords = [hand_landmark.landmark[0].x, hand_landmark.landmark[0].y]
                finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                for idx, _i in enumerate(needed_data[1:]):
                    fingertip = hand_landmark.landmark[_i]
                    fingertip_coords.append([fingertip.x, fingertip.y])
                    offsets[f'{finger_names[idx]}_offset'] = [(fingertip.x - wrist_coords[0]) * 10, (fingertip.y - wrist_coords[1]) * 10]

                # Draw hand landmarks on the image
                for idx, landmark in enumerate(needed_data):
                    x = int(hand_landmark.landmark[landmark].x * image.shape[1])
                    y = int(hand_landmark.landmark[landmark].y * image.shape[0])
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Draw green circle on the hand landmarks
                    if idx > 0:
                        cv2.putText(image, finger_names[idx - 1], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Process the image for facial landmarks (nose)
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        face_results = face_mesh.process(image)
        if face_results.multi_face_landmarks:
            for _face_landmarks in face_results.multi_face_landmarks:
                nose = _face_landmarks.landmark[0]  # Nose landmark
                nose_coords = [nose.x, nose.y]

                # Compute offsets from wrist to nose
                if wrist_coords is not None:
                    offsets['Nose_offset'] = [(nose_coords[0] - wrist_coords[0]) * 10, (nose_coords[1] - wrist_coords[1]) * 10]

                # Draw nose landmark on the image
                nose_x = int(nose.x * image.shape[1])
                nose_y = int(nose.y * image.shape[0])
                cv2.circle(image, (nose_x, nose_y), 5, (255, 0, 0), -1)  # Draw blue circle on the nose landmark

    # Validate the offsets count
    expected_offsets_count = 6  # Five fingertips offsets
    # if wrist_coords is not None and nose_coords is not None:
    #     expected_offsets_count += 1  # Add 1 if the nose offset is also available

    if len(offsets) != expected_offsets_count:
        print("Warning: Incomplete or inconsistent offsets detected.")
        offsets = {}  # Optionally, return an empty dictionary or handle this case differently

    return image, offsets

# Function to record gestures and save training data
def record_gesture_training_data(cap, num_entries=30, frames_count=10, label='no label'):
    training_data = []
    entry_count = 0
    label_count = 1
    recording = False
    recorded_frames = []

    while True:
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

        # Overlay text to show recording status, label count, and frame count
        if recording:
            cv2.putText(frame_with_landmarks, f"Recording Label {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame_with_landmarks, f"Frame Count: {len(recorded_frames)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

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
                    # Adjust frames to ensure exactly `frames_count` frames are saved
                    if len(recorded_frames) < frames_count:
                        recorded_frames.extend([recorded_frames[-1]] * (frames_count - len(recorded_frames)))
                    elif len(recorded_frames) > frames_count:
                        indices = np.linspace(0, len(recorded_frames) - 1, frames_count, dtype=int)
                        recorded_frames = [recorded_frames[i] for i in indices]

                    # Save the gesture data with label
                    label = label
                    training_data.append({'label': label, 'frames': recorded_frames})
                    entry_count += 1

                    print(f"Saved gesture {entry_count} with label '{label}'")

        elif key == ord('x'):  # Press 'x' to increment label count
            label_count += 1
            print(f"Incremented label to {label_count}")

        elif key == ord('q'):  # Press 'q' to quit
            break

        # Record frames if recording is active
        if recording:
            recorded_frames.append(offsets)

    # Save training data to file
    if training_data:
        with open('/tmp/training_data.log', 'a') as f:
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

    mode = 'r'#input("Enter 'r' to record gestures or 'v' to view landmarks: ").strip().lower()
    if mode == 'r':
        record_gesture_training_data(cap, num_entries=10, frames_count=20, label='cat')
    elif mode == 'v':
        while cap.isOpened():
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

            # Display the offsets in the console
            if offsets:
                with open('/tmp/offsets.log', 'a') as f:
                    for k, v in offsets.items():
                        f.write(f'{k}: {v}\n')
                    f.write('\n\n\n')
                print(offsets)

            # Display the frame with landmarks
            cv2.imshow('Video with Landmarks', frame_with_landmarks)

            # Press 'q' to exit the video early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()