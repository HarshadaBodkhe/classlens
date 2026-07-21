import dlib
import numpy as np
import face_recognition_models
import streamlit as st

from src.database.db import get_all_students


# ---------------------------
# Load Dlib Models
# ---------------------------

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


# ---------------------------
# Generate Face Embeddings
# ---------------------------

def get_face_embeddings(image_np):
    detector, sp, facerec = load_dlib_models()

    faces = detector(image_np, 1)

    encodings = []

    for face in faces:
        shape = sp(image_np, face)

        face_descriptor = facerec.compute_face_descriptor(
            image_np,
            shape,
            1
        )

        encodings.append(np.array(face_descriptor))

    return encodings


# ---------------------------
# Load Registered Students
# ---------------------------

@st.cache_resource
def get_registered_students():

    students = get_all_students()

    registered = []

    for student in students:

        embedding = student.get("face_embedding")

        if embedding:

            registered.append({
                "student_id": student["student_id"],
                "embedding": np.array(embedding)
            })

    return registered


# ---------------------------
# Refresh Cache
# ---------------------------

def train_classifier():
    st.cache_resource.clear()
    return True


# ---------------------------
# Predict Attendance
# ---------------------------

def predict_attendance(class_image_np):

    encodings = get_face_embeddings(class_image_np)

    detected_student = {}

    registered_students = get_registered_students()

    if len(registered_students) == 0:
        return detected_student, [], len(encodings)

    all_students = [
        student["student_id"]
        for student in registered_students
    ]

    THRESHOLD = 0.60

    for encoding in encodings:

        best_distance = float("inf")
        best_student = None

        for student in registered_students:

            distance = np.linalg.norm(
                student["embedding"] - encoding
            )

            if distance < best_distance:
                best_distance = distance
                best_student = student["student_id"]

        print("--------------------------------")
        print("Best Match Student :", best_student)
        print("Distance :", best_distance)

        if best_distance <= THRESHOLD:

            detected_student[int(best_student)] = True

    return detected_student, all_students, len(encodings)