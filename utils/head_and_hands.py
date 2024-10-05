import cv2 as cv
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles

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

# Open the camera
# cam = cv.VideoCapture(0)
# Open the video file instead of the camera
video_path = "/home/thunderhorn/Downloads/test_img.mp4"  # Replace with your actual video file path
cam = cv.VideoCapture(video_path)
while cam.isOpened():
    # Read a frame from the camera
    success, frame = cam.read()

    # If the frame is not available, skip this iteration
    if not success:
        print("Camera Frame not available")
        continue

    # Convert the frame from BGR to RGB (required by MediaPipe)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Process the frame for hand detection
    hands_detected = hands.process(frame_rgb)

    # Process the frame for face detection
    face_detected = face_mesh.process(frame_rgb)

    # Convert the frame back from RGB to BGR (required by OpenCV)
    frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)

    # If hands are detected, draw landmarks and connections on the frame
    if hands_detected.multi_hand_landmarks:
        for hand_landmarks in hands_detected.multi_hand_landmarks:
            drawing.draw_landmarks(
                frame_bgr,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style(),
            )

    # If face is detected, draw landmarks and connections on the frame
    if face_detected.multi_face_landmarks:
        for face_landmarks in face_detected.multi_face_landmarks:
            # drawing.draw_landmarks(
            #     frame_bgr,
            #     face_landmarks,
            #     mp_face_mesh.FACEMESH_TESSELATION,
            #     drawing_styles.get_default_face_mesh_tesselation_style(),
            # )
            try:
                drawing.draw_landmarks(
                    frame_bgr,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    drawing_styles.get_default_face_mesh_contours_style(),
                )
            except:
                pass
            try:
                drawing.draw_landmarks(
                    frame_bgr,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_IRISES,
                    drawing_styles.get_default_face_mesh_iris_connections_style(),
                )
            except:
                pass

    # Display the frame with annotations
    cv.imshow("Hands and Face Detection", frame_bgr)

    # Exit the loop if 'q' key is pressed
    if cv.waitKey(20) & 0xff == ord('q'):
        break

# Release the camera and close OpenCV window
cam.release()
cv.destroyAllWindows()
