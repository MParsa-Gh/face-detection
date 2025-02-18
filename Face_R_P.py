import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

video_path = "vide (2).mp4" #browse video file
cap = cv2.VideoCapture(video_path) #open video file (0 for webcam)

fps = cap.get(cv2.CAP_PROP_FPS) #get fps of the video
if fps == 0:
    fps = 25
delay = int(1000/fps) #delay for each frame

while True: #raad video frame by frame
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20) #increase brightnes of the frame

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convet frame BGR to RGB
    results = face_detection.process(image_rgb)#detect face in the frame

    if results.detections:
        for face in results.detections:

            # Draws detection points and lines on the frame
            mp_drawing.draw_detection(frame, face)
            bbox = face.location_data.relative_bounding_box
            frame_height, frame_width, _ = frame.shape

            #connvert relative coordinates to absolute (nesbi be motlagh)
            x = int(bbox.xmin * frame_width)
            y = int(bbox.ymin * frame_height)
            w = int(bbox.width * frame_width)
            h = int(bbox.height * frame_height)

            #rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            #vertical linea center of the face
            center_x = x + w // 2
            cv2.line(frame, (center_x, y), (center_x, y + h), (0, 0, 255), 2)

            #confidence
            confidence = face.score[0] * 100
            cv2.putText(frame, f"{confidence:.1f}%", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    cv2.imshow("Face Detection | @MParsa_Ai", frame)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

