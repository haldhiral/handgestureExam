import cv2
import mediapipe as mp

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def count_fingers(hand_landmarks):
    """Detect the number of fingers raised."""
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    thumb_tip = 4

    # Calculate whether each finger is raised
    fingers = [1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y else 0
               for tip in finger_tips]

    # Check if thumb is open (horizontally to the left)
    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 2].x:
        fingers.insert(0, 1)  # Thumb is open
    else:
        fingers.insert(0, 0)

    return sum(fingers)  # Total fingers raised

# Initialize Webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    # Set the OpenCV window to full screen
    cv2.namedWindow("Hand Gesture Exam", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Hand Gesture Exam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Draw Question Box
        cv2.rectangle(frame, (50, 50), (w - 50, 150), (255, 255, 255), -1)
        cv2.putText(frame, "goks", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

        # Draw 4 Answer Boxes
        answer_positions = [
            (int(w * 0.05), int(h * 0.3), int(w * 0.3), int(h * 0.4)),   # Box 1
            (int(w * 0.35), int(h * 0.3), int(w * 0.6), int(h * 0.4)),   # Box 2
            (int(w * 0.65), int(h * 0.3), int(w * 0.9), int(h * 0.4)),   # Box 3
            (int(w * 0.05), int(h * 0.5), int(w * 0.3), int(h * 0.6)),   # Box 4
        ]

        # Draw the answer boxes with labels
        for i, (x1, y1, x2, y2) in enumerate(answer_positions):
            color = (255, 255, 255)  # Default color
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"test {i + 1}", (x1 + 10, y1 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # If hands are detected, process the gesture
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the number of fingers raised
                fingers_count = count_fingers(hand_landmarks)

                # Highlight the corresponding answer box
                if 1 <= fingers_count <= 4:
                    x1, y1, x2, y2 = answer_positions[fingers_count - 1]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

        # Display the frame in full screen
        cv2.imshow("Hand Gesture Exam", frame)

        # Press 'q' to exit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
