import cv2
import mediapipe as mp
import time
import numpy as np
import os
import face_recognition
from threading import Thread, Lock

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

recognized_name = None
recognized_lock = Lock()

# Load known faces
def load_known_faces(path="D:/Pictures/"):
    known_faces = []
    known_face_names = []
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(path, filename)
            image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_image)
            if face_encodings:
                known_faces.append(face_encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])
    return known_faces, known_face_names

known_faces, known_face_names = load_known_faces()

# Questions and answers
questions = [
    {"question": "Solve: 2x + 3 = 7", "answers": ["x = 1", "x = 2", "x = 3", "x = 4"], "correct": 2},
    {"question": "Find the derivative of x^2 with respect to x", "answers": ["1", "x", "2x", "x^2"], "correct": 3},
]

# Gesture detection
def count_fingers(hand_landmarks):
    """Detect the number of fingers raised."""
    finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky
    thumb_tip = 4
    raised_fingers = sum(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y for tip in finger_tips)
    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 2].x:
        raised_fingers += 1
    return raised_fingers

# Face recognition thread
def face_recognition_thread(face_encodings, known_faces, known_face_names):
    global recognized_name
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        if True in matches:
            match_index = matches.index(True)
            with recognized_lock:
                recognized_name = known_face_names[match_index]
            return

# Main application
def main():
    global recognized_name
    recognized_name = None
    recognized_lock = Lock()
    cap = cv2.VideoCapture(0)
    frame_count = 0
    face_recognition_interval = 5
    question_index = 0
    score = 0
    selected_answer = None
    confirmation_needed = False
    last_gesture_time = time.time()
    gesture_delay = 7

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            frame_count += 1

            # Face recognition logic
            if frame_count % face_recognition_interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                if face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    if face_encodings:
                        Thread(target=face_recognition_thread, args=(face_encodings, known_faces, known_face_names)).start()

            # Welcome screen if no face recognized
            with recognized_lock:
                if not recognized_name:
                    cv2.putText(frame, "Please show your face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("Hand Gesture Exam", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

            # Display question
            current_question = questions[question_index]
            question_text = current_question["question"]
            answers = current_question["answers"]
            correct_answer = current_question["correct"]

            # Draw question box
            text_scale = 1.2 if len(question_text) <= 30 else 0.8
            question_box_height = 50 + int((len(question_text) / 40) * 40)
            cv2.rectangle(frame, (50, 50), (w - 50, 50 + question_box_height), (255, 255, 255), -1)
            cv2.putText(frame, f"Welcome {recognized_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, question_text, (60, 80), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 0), 2)

            # Draw 4 Answer Boxes
            answer_positions = [
                (int(w * 0.05), int(h * 0.3), int(w * 0.35), int(h * 0.4)),
                (int(w * 0.35), int(h * 0.3), int(w * 0.65), int(h * 0.4)),
                (int(w * 0.65), int(h * 0.3), int(w * 0.95), int(h * 0.4)),
                (int(w * 0.05), int(h * 0.5), int(w * 0.35), int(h * 0.6)),
            ]

            # Draw the answer boxes
            for i, (x1, y1, x2, y2) in enumerate(answer_positions):
                color = (255, 255, 255)
                if selected_answer == i + 1:
                    color = (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, answers[i], (x1 + 10, y1 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Process hands
            if not confirmation_needed:
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        fingers_count = count_fingers(hand_landmarks)
                        if fingers_count:
                            # Add a delay before showing confirmation popup
                            if time.time() - last_gesture_time > gesture_delay:
                                x1, y1, x2, y2 = answer_positions[fingers_count - 1]
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                                selected_answer = fingers_count
                                confirmation_needed = True
                                last_gesture_time = time.time()  # Update time of last gesture

            # Confirm popup
            if confirmation_needed:
                cv2.rectangle(frame, (int(w * 0.35), int(h * 0.7)), (int(w * 0.65), int(h * 0.8)), (0, 255, 255), -1)
                cv2.putText(frame, "Confirm? (y/n)", (int(w * 0.37), int(h * 0.75)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # Handle key inputs
            key = cv2.waitKey(1) & 0xFF
            if confirmation_needed:
                if key == ord('y'):
                    if selected_answer == correct_answer:
                        score += 1
                    question_index += 1
                    confirmation_needed = False
                    selected_answer = None
                    if question_index >= len(questions):
                        break
                elif key == ord('n'):
                    confirmation_needed = False
                    selected_answer = None

            # Quit key
            if key == ord('q'):
                break

            # Show the frame
            cv2.imshow("Hand Gesture Exam", frame)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Your final score is: {score}/{len(questions)}")

if __name__ == "__main__":
    main()