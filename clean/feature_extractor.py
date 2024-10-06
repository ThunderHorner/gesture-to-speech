import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging

logging.getLogger('tensorflow').disabled = True
logging.getLogger('mediapipe').disabled = True
import cv2
import mediapipe as mp
import warnings

warnings.filterwarnings('ignore')

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh


def extract_landmarks(image):
    wrist_coords = None
    fingertip_coords = []
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
                    offsets[f'{finger_names[idx]}_offset'] = [(fingertip.x - wrist_coords[0]) * 10,
                                                              (fingertip.y - wrist_coords[1]) * 10]

                # Draw hand landmarks on the image
                for idx, landmark in enumerate(needed_data):
                    x = int(hand_landmark.landmark[landmark].x * image.shape[1])
                    y = int(hand_landmark.landmark[landmark].y * image.shape[0])
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Draw green circle on the hand landmarks
                    if idx > 0:
                        cv2.putText(image, finger_names[idx - 1], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 1)

    # Process the image for facial landmarks (nose)
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        face_results = face_mesh.process(image)
        if face_results.multi_face_landmarks:
            for _face_landmarks in face_results.multi_face_landmarks:
                nose = _face_landmarks.landmark[0]  # Nose landmark
                nose_coords = [nose.x, nose.y]

                if wrist_coords is not None:
                    offsets['Nose_offset'] = [(nose_coords[0] - wrist_coords[0]) * 10,
                                              (nose_coords[1] - wrist_coords[1]) * 10]

                nose_x = int(nose.x * image.shape[1])
                nose_y = int(nose.y * image.shape[0])
                cv2.circle(image, (nose_x, nose_y), 5, (255, 0, 0), -1)

    expected_offsets_count = 6  # Five fingertips offsets

    if len(offsets) != expected_offsets_count:
        print("Warning: Incomplete or inconsistent offsets detected.")
        offsets = {}

    return image, offsets
