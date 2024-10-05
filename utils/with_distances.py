import json

import cv2 as cv
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles
import numpy as np

# Initialize the Hands model
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# Initialize the Face Mesh model
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Open the video file instead of the camera
video_path = "/home/thunderhorn/Downloads/test_img.mp4"  # Replace with your actual video file path
cam = cv.VideoCapture(video_path)
distances = []
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)

while cam.isOpened():
    success, frame = cam.read()
    if not success:
        print("End of video or unable to read the video")
        break

    # Convert the frame from BGR to RGB
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Process the frame for hand and face detection
    hands_detected = hands.process(frame_rgb)
    face_detected = face_mesh.process(frame_rgb)

    # Convert back to BGR
    frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)

    # Initialize lists to store distances
    distances = []

    # If hands are detected, draw landmarks
    if hands_detected.multi_hand_landmarks:
        for hand_landmarks in hands_detected.multi_hand_landmarks:
            drawing.draw_landmarks(
                frame_bgr,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style(),
            )

    # If face is detected, draw landmarks and calculate distances
    if face_detected.multi_face_landmarks:
        for face_landmarks in face_detected.multi_face_landmarks:
            drawing.draw_landmarks(
                frame_bgr,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                drawing_styles.get_default_face_mesh_tesselation_style(),
            )

            # Extract key facial landmarks
            nose = face_landmarks.landmark[0]  # Nose
            chin = face_landmarks.landmark[152]  # Chin
            left_ear = face_landmarks.landmark[234]  # Left Ear
            right_ear = face_landmarks.landmark[454]  # Right Ear
            left_eye_outer = face_landmarks.landmark[159]  # Left Eye Outer
            right_eye_outer = face_landmarks.landmark[386]  # Right Eye Outer

            if hands_detected.multi_hand_landmarks:
                hand_data = {'h1': [0 for i in range(63)], 'h2': [0 for i in range(63)]}

                for index,  hand_landmarks in enumerate(hands_detected.multi_hand_landmarks):
                    hand_data[f'h{index+1}'] = []
                    for i in hand_landmarks.landmark:
                        hand_data[f'h{index+1}'].append(i.x)
                        hand_data[f'h{index+1}'].append(i.y)
                        hand_data[f'h{index+1}'].append(i.z)
                    hand_data['nose'] = [nose.x, nose.y, nose.z]
                    hand_data['chin'] = [chin.x, chin.y, chin.z]
                    hand_data['leye'] = [left_eye_outer.x, left_eye_outer.y, left_eye_outer.z]
                    hand_data['reye'] = [right_eye_outer.x, right_eye_outer.y, right_eye_outer.z]
                    hand_data['lear'] = [left_ear.x, left_ear.y, left_ear.z]
                    hand_data['rear'] = [right_ear.x, right_ear.y, right_ear.z]
                with open('ff.txt', 'a')as f:
                    f.write(json.dumps(hand_data) + '\n')
                break
    # Display the frame with annotations
    cv.imshow("Hands and Face Detection", frame_bgr)

    # Exit the loop if 'q' key is pressed
    if cv.waitKey(20) & 0xff == ord('q'):
        break

# Release the video file and close OpenCV window
cam.release()
cv.destroyAllWindows()

# Print calculated distances (for debugging)
print(distances)
