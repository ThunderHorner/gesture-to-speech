import json
import os
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands and Face Mesh
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# Function to compute Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)

# Function to extract hand and facial landmarks
def extract_landmarks(image):
    wrist_coords = None
    fingertip_coords = []
    nose_coords = None

    # Process the image for hand landmarks
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
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
                wrist_coords = [hand_landmark.landmark[0].x, hand_landmark.landmark[0].y, hand_landmark.landmark[0].z]
                for _i in needed_data[1:]:
                    fingertip = hand_landmark.landmark[_i]
                    fingertip_coords.append([fingertip.x, fingertip.y, fingertip.z])

                # Draw hand landmarks on the image
                for landmark in needed_data:
                    x = int(hand_landmark.landmark[landmark].x * image.shape[1])
                    y = int(hand_landmark.landmark[landmark].y * image.shape[0])
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Draw green circle on the hand landmarks

    # Process the image for facial landmarks (nose)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        face_results = face_mesh.process(image)
        if face_results.multi_face_landmarks:
            for _face_landmarks in face_results.multi_face_landmarks:
                nose = _face_landmarks.landmark[0]  # Nose landmark
                nose_coords = [nose.x, nose.y, nose.z]

                # Draw nose landmark on the image
                nose_x = int(nose.x * image.shape[1])
                nose_y = int(nose.y * image.shape[0])
                cv2.circle(image, (nose_x, nose_y), 5, (255, 0, 0), -1)  # Draw blue circle on the nose landmark

    return image

if __name__ == '__main__':
    training_images_path = '/home/thunderhorn/PycharmProjects/gesture-to-speech/labeled_images/'  # Update with your image path

    image_files = sorted(os.listdir(training_images_path))  # Sort files to maintain sequence order

    # Process images and display with landmarks
    for path in sorted(os.listdir(training_images_path)):
        image_path = os.path.join(training_images_path, path)
        image = cv2.imread(image_path)
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract landmarks and draw them on the image
        image_with_landmarks = extract_landmarks(frame_rgb)

        # Display the image with landmarks
        cv2.imshow('Image with Landmarks', image_with_landmarks)
        cv2.waitKey(0)  # Wait for a key press to display the next image

    cv2.destroyAllWindows()
