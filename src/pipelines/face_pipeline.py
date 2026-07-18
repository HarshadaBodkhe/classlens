import dlib
import numpy as np
import face_recognition_models
from sklearn.svm import SVC
import streamlit as st 
import cv2 
from src.database.db import get_all_students


@st.cache_resource
def load_dlib_models():
    detector = dlib.get_frontal_face_detector()
    
    sp = dlib.shape_predictor(
        face_recognition_models.pose_predictor_model_location()
    )
    
    facerec = dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )
    
    return detector, sp, facerec

import dlib
import numpy as np


def get_face_embeddings(image_np):
    detector, sp, facerec = load_dlib_models()

    # Ensure dtype is uint8
    image_np = image_np.astype(np.uint8)

    # Convert RGB (from PIL) → BGR (for OpenCV/dlib)
    if image_np.shape[-1] == 3:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image_np

    # Detect faces
    faces = detector(image_bgr, 1)
    encodings = []

    for face in faces:
        shape = sp(image_bgr, face)
        face_descriptor = facerec.compute_face_descriptor(image_bgr, shape, 1)
        encodings.append(np.array(face_descriptor))

    return encodings


@st.cache_resource
def get_train_model():
    X = []
    y = []
    
    student_db = get_all_students()
    
    if not student_db:
        return None
    
    for student in student_db:
        embedding = student.get('face_embedding')
        if embedding:
            X.append(np.array(embedding))
            y.append(student.get('student_id'))
            
    if len(set(y)) < 2:   # only one or zero classes
        return {'clf': None, 'X': X, 'y': y}

    clf = SVC(kernel='linear', probability=True, class_weight='balanced')
    clf.fit(X, y)

    return {'clf': clf, 'X': X, 'y': y}
  
    
def train_classifier():
    st.cache_resource.clear()
    model_data = get_train_model()
    return bool(model_data)

def predict_attendance(class_image_np):
    encodings = get_face_embeddings(class_image_np)
    
    detected_student = {}
    
    model_data = get_train_model()
    
    if not model_data:
        return detected_student, [], 0  #  values-embedding, list of students, number of student
    
    clf = model_data['clf']
    X_train = model_data['X']
    y_train = model_data['y']
    
    all_students= sorted(list(set(y_train)))
    
    for encoding in encodings:
        if len(all_students)>=2:
            predicted_id = int(clf.predict([encoding])[0])
        else:
            predicted_id = int(all_students[0])
            
        student_embedding = X_train[y_train.index(predicted_id)]
        
        best_match_score = np.linalg.norm(student_embedding-encoding)
        
        resemblance_threshold = 0.7
        
        if best_match_score <= resemblance_threshold:
            detected_student[predicted_id] = True
            
    return detected_student, all_students, len(encodings)
    