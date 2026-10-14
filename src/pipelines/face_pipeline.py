# import dlib
# import numpy as np
# import face_recognition_models
# from sklearn.svm import SVC
# import streamlit as st

# from src.database.db import get_all_students


# @st.cache_resource
# def load_dlib_models():
#     detector = dlib.get_frontal_face_detector() 

#     sp = dlib.shape_predictor(
#         face_recognition_models.pose_predictor_model_location()
#     )

#     facerec = dlib.face_recognition_model_v1(
#         face_recognition_models.face_recognition_model_location()
#     )

#     return detector, sp, facerec

# def get_face_embeddings(image_np):
#     try:
#         detector, sp, facerec = load_dlib_models()
#         faces = detector(image_np, 1)

#         encodings = []
#         for face in faces:
#             shape = sp(image_np, face)
#             face_descriptor = facerec.compute_face_descriptor(image_np, shape, 1)
#             encodings.append(np.array(face_descriptor))
#         return encodings
#     except Exception:
#         # Fallback if image parsing fails or no face is found offline
#         return []

# @st.cache_resource
# def get_trained_model():
#     X = []
#     y = []

#     student_db = get_all_students()

#     if not student_db:
#         return None
    
#     for student in student_db:
#         embedding = student.get('face_embedding')
#         student_id = student.get('id') or student.get('student_id')
#         if embedding and student_id:
#             X.append(np.array(embedding))
#             y.append(student_id)

#     if len(X) == 0:
#         return None
    
#     clf = SVC(kernel='linear', probability=True, class_weight='balanced')

#     try:
#         clf.fit(X, y)
#     except ValueError:
#         return None

#     return {'clf': clf, 'X': X, "y": y}


# def train_classifier():
#     st.cache_resource.clear()
#     model_data = get_trained_model()
#     return bool(model_data)

# def predict_attendance(class_image_np):
#     # Mock return for offline testing if no trained model or encodings exist
#     model_data = get_trained_model()
#     encodings = get_face_embeddings(class_image_np)

#     detected_student = {}

#     if not model_data or not encodings:
#         # Pass test student ID 1 as present locally
#         return {1: True}, [1], 1
    
#     clf = model_data['clf']
#     X_train = model_data['X']
#     y_train = model_data['y']

#     all_students = sorted(list(set(y_train)))

#     for encoding in encodings:
#         if len(all_students) >= 2:
#             predicted_id = int(clf.predict([encoding])[0])
#         else:
#             predicted_id = int(all_students[0])

#         student_embedding = X_train[y_train.index(predicted_id)]
#         best_match_score = np.linalg.norm(student_embedding - encoding)
#         resemblance_threshold = 0.6

#         if best_match_score <= resemblance_threshold:
#             detected_student[predicted_id] = True

#     return detected_student, all_students, len(encodings)
import dlib
import numpy as np
import face_recognition_models
import streamlit as st

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

def get_face_embeddings(image_np):
    try:
        detector, sp, facerec = load_dlib_models()
        
        # Ensure image is uint8 RGB and drop alpha channel if present
        if image_np.dtype != np.uint8:
            image_np = image_np.astype(np.uint8)
        if image_np.shape[-1] == 4:
            image_np = image_np[:, :, :3]

        faces = detector(image_np, 1)

        encodings = []
        for face in faces:
            shape = sp(image_np, face)
            face_descriptor = facerec.compute_face_descriptor(image_np, shape, 1)
            encodings.append(np.array(face_descriptor))
        return encodings
    except Exception as e:
        print(f"Face encoding error: {e}")
        return []

def train_classifier():
    # Kept for compatibility with your existing UI button calls
    st.cache_resource.clear()
    return True

def predict_attendance(class_image_np):
    encodings = get_face_embeddings(class_image_np)
    student_db = get_all_students()

    detected_students = {}
    all_student_ids = [s['student_id'] for s in student_db] if student_db else []

    if not encodings or not student_db:
        return detected_students, all_student_ids, len(encodings)

    # Threshold for face match (lower = stricter match, 0.5 to 0.6 is optimal)
    resemblance_threshold = 0.55 

    for encoding in encodings:
        best_match_id = None
        min_distance = float('inf')

        for student in student_db:
            db_embedding = student.get('face_embedding')
            student_id = student.get('student_id')

            if db_embedding and student_id:
                known_embedding = np.array(db_embedding)
                distance = np.linalg.norm(known_embedding - encoding)

                if distance < min_distance:
                    min_distance = distance
                    best_match_id = student_id

        if best_match_id is not None and min_distance <= resemblance_threshold:
            detected_students[best_match_id] = True

    return detected_students, all_student_ids, len(encodings)

