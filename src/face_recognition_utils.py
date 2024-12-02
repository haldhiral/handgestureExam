import face_recognition
import os
import cv2

def load_known_faces(path):
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

def recognize_face(face_encodings, known_faces, known_face_names):
    matches = face_recognition.compare_faces(known_faces, face_encodings)
    if True in matches:
        match_index = matches.index(True)
        return known_face_names[match_index]
    return None
