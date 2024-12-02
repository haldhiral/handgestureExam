def count_fingers(hand_landmarks):
    """Detect the number of fingers raised."""
    finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky
    thumb_tip = 4
    raised_fingers = sum(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y for tip in finger_tips)
    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 2].x:
        raised_fingers += 1
    return raised_fingers
