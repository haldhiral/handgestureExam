import cv2
import time
import face_recognition
import mediapipe as mp
from face_recognition_utils import load_known_faces, recognize_face
from gesture_recognition import count_fingers
from ui import display_welcome_screen, draw_question_and_answers
from config import KNOWN_FACES_PATH, FACE_RECOGNITION_INTERVAL, GESTURE_DELAY

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def main():
    # Load known faces
    known_faces, known_face_names = load_known_faces(KNOWN_FACES_PATH)
    recognized_name = None  # Initialize name as None
    cap = cv2.VideoCapture(0)
    question_index = 0
    score = 0
    confirmation_needed = False
    selected_answer = None
    last_gesture_time = time.time()
    gesture_delay = GESTURE_DELAY

    # Questions for the quiz
    questions = [
        {"question": "Solve: 2x + 3 = 7", "answers": ["x = 1", "x = 2", "x = 3", "x = 4"], "correct": 2},
        {"question": "Find the derivative of x^2 with respect to x", "answers": ["1", "x", "2x", "x^2"], "correct": 3},
    ]
  
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize hands model with detection and tracking confidence
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
        # Start the video capture and display loop
        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Failed to capture frame.")
                break

            h, w, _ = frame.shape

            # Face recognition only happens at the start
            if recognized_name is None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                if face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    if face_encodings:
                        recognized_name = recognize_face(face_encodings[0], known_faces, known_face_names)

            # Display the welcome screen if no face is recognized
            if not recognized_name:
                display_welcome_screen(frame, recognized_name)
            else:
                current_question = questions[question_index]
                question_text = current_question["question"]
                answers = current_question["answers"]
                correct_answer = current_question["correct"]
                draw_question_and_answers(frame, question_text, answers, selected_answer, w, h)

            # Process hand landmarks for gesture recognition if no confirmation needed
            if not confirmation_needed:
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        fingers_count = count_fingers(hand_landmarks)
                        if fingers_count:
                            if time.time() - last_gesture_time > gesture_delay:
                                # Define the positions for the answers
                                answer_positions = [
                                    (int(w * 0.05), int(h * 0.3), int(w * 0.35), int(h * 0.4)),
                                    (int(w * 0.35), int(h * 0.3), int(w * 0.65), int(h * 0.4)),
                                    (int(w * 0.65), int(h * 0.3), int(w * 0.95), int(h * 0.4)),
                                    (int(w * 0.05), int(h * 0.5), int(w * 0.35), int(h * 0.6)),
                                ]
                                x1, y1, x2, y2 = answer_positions[fingers_count - 1]
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                                selected_answer = fingers_count
                                confirmation_needed = True
                                last_gesture_time = time.time()

            # Confirm popup
            if confirmation_needed:
                cv2.rectangle(frame, (int(w * 0.35), int(h * 0.7)), (int(w * 0.65), int(h * 0.8)), (0, 255, 255), -1)
                cv2.putText(frame, "Confirm? (y/n)", (int(w * 0.37), int(h * 0.75)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # Handle key inputs for confirmation
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

            # Show the current frame
            cv2.imshow("Hand Gesture Exam", frame)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
