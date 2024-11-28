import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load OpenCV face detection model (Haar Cascade Classifier)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def count_fingers(hand_landmarks):
    """Detect the number of fingers raised."""
    # Finger tips: index, middle, ring, pinky
    finger_tips = [8, 12, 16, 20]
    # Thumb tip is 4
    thumb_tip = 4
    raised_fingers = 0

    # Count the raised fingers
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            raised_fingers += 1

    # Check thumb separately: Thumb is raised if the tip is to the left of the base knuckle
    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 2].x:
        raised_fingers += 1

    return raised_fingers

# Questions and answers
questions = [
    {"question": "Solve: 2x + 3 = 7", "answers": ["x = 1", "x = 2", "x = 3", "x = 4"], "correct": 2},
    {"question": "Find the derivative of x^2 with respect to x", "answers": ["1", "x", "2x", "x^2"], "correct": 3},
]
question_index = 0
score = 0
confirmation_needed = False
selected_answer = None

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Full screen settings
screen_width = 1920
screen_height = 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

# Time delay before showing confirmation popup (in seconds)
gesture_delay = 5  # 1 second delay before showing confirmation

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    last_gesture_time = time.time()  # Track the time of the last gesture detected

    while cap.isOpened():
        success, frame = cap.read()
        
        # Check if frame is valid
        if not success or frame is None:
            # Show popup if camera is off or no frame is captured
            blank_frame = 255 * np.ones((400, 600, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "Please show your face", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Hand Gesture Exam", blank_frame)

            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Get the current question and answers
        current_question = questions[question_index]
        question_text = current_question["question"]
        answers = current_question["answers"]
        correct_answer = current_question["correct"]

        # Display a popup if no face is detected
        if len(faces) == 0:
            cv2.putText(frame, "Please show your face", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Hand Gesture Exam", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue

        # Display question box
        text_scale = 1.2 if len(question_text) <= 30 else 0.8
        question_box_height = 50 + int((len(question_text) / 40) * 40)
        cv2.rectangle(frame, (50, 50), (w - 50, 50 + question_box_height), (255, 255, 255), -1)
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
