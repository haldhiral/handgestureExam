import cv2
import time
import face_recognition
import mediapipe as mp
import requests
from face_recognition_utils import load_known_faces, recognize_face
from gesture_recognition import count_fingers
from ui import display_welcome_screen, draw_question_and_answers, display_user_name, display_countdown, send_grade
from config import KNOWN_FACES_PATH, FACE_RECOGNITION_INTERVAL, GESTURE_DELAY

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# API URL for fetching the questions
API_URL = "http://localhost/onlinetest/api/exams"

def fetch_exam_data():
    """Fetch questions from the API"""
    try:
        response = requests.post(API_URL)
        if response.status_code == 200:
            return response.json()  # Return the JSON data from the API
        else:
            print(f"Failed to fetch data. Status Code: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def main():
    # Load known faces
    known_faces, known_face_names = load_known_faces(KNOWN_FACES_PATH)
    recognized_name = None  # Initialize name as None
    cap = cv2.VideoCapture(0)

    # Set the OpenCV window to full screen
    cv2.namedWindow("Hand Gesture Exam", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Hand Gesture Exam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    question_index = 0
    score = 0
    confirmation_needed = False
    countdown_started = False
    countdown_start_time = None
    selected_answer = None
    last_gesture_time = time.time()
    gesture_delay = GESTURE_DELAY

    # Fetch questions from the API
    questions = fetch_exam_data()
    if not questions:
        print("No questions available. Exiting...")
        return

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
                if not countdown_started:
                    countdown_started = True
                    countdown_start_time = time.time()  # Start the countdown

                # Calculate the remaining time
                elapsed_time = time.time() - countdown_start_time
                seconds_left = max(0, 3 - int(elapsed_time))  # Calculate remaining seconds

                if seconds_left > 0:
                    # Show user's name and countdown on the screen
                    success, frame = cap.read()
                    if not success:
                        print("Error: Failed to capture frame.")
                        break
                    display_user_name(frame, recognized_name, w)
                    display_countdown(frame, w, h, seconds_left)
                    cv2.imshow("Hand Gesture Exam", frame)
                    cv2.waitKey(1)  # Small delay to allow frame update
                else:
                    # Countdown finished, show the question and answers
                    current_question = questions[question_index]
                    question_text = current_question["question"]
                    answers = current_question["answers"]
                    correct_answer = current_question["correct"]
                    draw_question_and_answers(frame, question_text, answers, selected_answer, w, h)

                # After the countdown, continue with the quiz
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
                            # Draw hand skeleton
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

    # Print final score
    send_grade(recognized_name, score, len(questions))
    print(f"Exam Completed! Your final score is: {score}/{len(questions)}")

if __name__ == "__main__":
    main()
