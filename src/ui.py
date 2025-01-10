import cv2
import time

import requests

def send_grade(user, score, total_question):
    # Define the PHP API URL
    api_url = "http://localhost/onlinetest/api/grades"
    
    # Prepare the data
    data = {
        "user": user,
        "score": score,
        "total_question": total_question
    }

    
    response = requests.post(api_url, json=data)

    if response.status_code == 201:
        print("Grade successfully submitted!")
    else:
        print("Failed to submit grade:", response.json())

def display_welcome_screen(frame, recognized_name):
    if not recognized_name:
        cv2.putText(frame, "Welcome to Smart Exam System", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Please Show Your Face to Sign Up", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Press 'Q' to Exit", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
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

def display_user_name(frame, recognized_name, w):
    if recognized_name:
        text = f"User: {recognized_name}"
        cv2.putText(frame, text, (int(w * 0.05), 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

def display_countdown(frame, w, h, seconds_left):
    # Draw background rectangle for countdown
    cv2.rectangle(frame, (int(w * 0.3), int(h * 0.4)), (int(w * 0.7), int(h * 0.6)), (0, 0, 0), -1)
    # Display countdown text
    cv2.putText(frame, "Exam will start in", (int(w * 0.32), int(h * 0.45)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, str(seconds_left), (int(w * 0.45), int(h * 0.55)),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
