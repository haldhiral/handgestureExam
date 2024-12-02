import cv2

def display_welcome_screen(frame, recognized_name):
    if not recognized_name:
        cv2.putText(frame, "Welcome to Smart Exam System", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Please Show Your Face to Sign Up", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Press 'Q' to Exit", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return frame

    return frame

def draw_question_and_answers(frame, question_text, answers, selected_answer, w, h):
    # Draw question box
    text_scale = 1.2 if len(question_text) <= 30 else 0.8
    question_box_height = 50 + int((len(question_text) / 40) * 40)
    cv2.rectangle(frame, (50, 50), (w - 50, 50 + question_box_height), (255, 255, 255), -1)
    cv2.putText(frame, question_text, (60, 80), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 0), 2)

    # Draw answer boxes
    answer_positions = [
        (int(w * 0.05), int(h * 0.3), int(w * 0.35), int(h * 0.4)),
        (int(w * 0.35), int(h * 0.3), int(w * 0.65), int(h * 0.4)),
        (int(w * 0.65), int(h * 0.3), int(w * 0.95), int(h * 0.4)),
        (int(w * 0.05), int(h * 0.5), int(w * 0.35), int(h * 0.6)),
    ]
    for i, (x1, y1, x2, y2) in enumerate(answer_positions):
        color = (255, 255, 255)
        if selected_answer == i + 1:
            color = (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, answers[i], (x1 + 10, y1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
    return frame
