import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    img_h, img_w, _ = frame.shape

    if results.multi_face_landmarks:
        mesh_points = results.multi_face_landmarks[0].landmark
        for id in range(468, 478):
            landmark = mesh_points[id]
            x = int(landmark.x * img_w)
            y = int(landmark.y * img_h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow('Eye Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
