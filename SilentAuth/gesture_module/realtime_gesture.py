# gesture_api.py
import os
import warnings
import logging
import time
import cv2
import mediapipe as mp
import numpy as np
import joblib
import sqlite3
from collections import Counter

# ------------------- SILENCE LOGS -------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("absl").setLevel(logging.ERROR)

# ------------------- DATABASE -------------------
DB_PATH = "---------------------------------"

def identify_user(gesture_name):
    if not gesture_name:
        return ["Unknown user"]

    gesture_name = gesture_name.strip().lower()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT users.name FROM gesture
        JOIN users ON gesture.user_id = users.user_id
        WHERE LOWER(TRIM(gesture_label)) = ?
    """, (gesture_name,))

    result = cursor.fetchall()
    conn.close()
    return [row[0] for row in result] if result else ["Unknown user"]

# ------------------- LOAD MODEL -------------------
clf = joblib.load("models/gesture_svm.pkl")
classes = clf.classes_

# ------------------- MEDIAPIPE -------------------
mp_hands = mp.solutions.hands

# ------------------- SETTINGS -------------------
REQUIRED_FRAMES = 5
CONFIDENCE_THRESHOLD = 0.6
NO_HAND_TIMEOUT_SECONDS = 10
MAX_TIMEOUT = 30

# ------------------- NORMALIZATION -------------------
def normalize(pts):
    pts = np.array(pts)
    base = pts[0]
    pts = pts - base
    scale = np.max(np.linalg.norm(pts, axis=1))
    return (pts / (scale + 1e-8)).flatten()

# ------------------- MAIN FUNCTION -------------------
def run_gesture_verification():
    PREDICTIONS = []
    no_hand_start_time = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {"modality": "gesture", "status": "failure", "reason": "Cannot open camera"}

    hands = mp_hands.Hands(max_num_hands=1)
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # SHOW CAMERA
        cv2.imshow("Gesture Verification", frame)

        # ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            cleanup(cap, hands)
            return {
                "modality": "gesture",
                "status": "failure",
                "reason": "User cancelled"
            }

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            no_hand_start_time = None
            lm = result.multi_hand_landmarks[0]
            pts = [(p.x, p.y, p.z) for p in lm.landmark]

            vec = normalize(pts).reshape(1, -1)
            probs = clf.predict_proba(vec)[0]

            idx = np.argmax(probs)
            score = probs[idx]
            gesture = classes[idx]

            if score >= CONFIDENCE_THRESHOLD:
                PREDICTIONS.append(gesture)

            if len(PREDICTIONS) >= REQUIRED_FRAMES:
                final_gesture = Counter(PREDICTIONS).most_common(1)[0][0]
                final_score = score

                user_names = identify_user(final_gesture)
                status = "success" if "Unknown user" not in user_names else "failure"

                cleanup(cap, hands)
                return {
                    "modality": "gesture",
                    "gesture": final_gesture,
                    "username": user_names,
                    "confidence": round(float(final_score), 2),
                    "status": status
                }

        else:
            if no_hand_start_time is None:
                no_hand_start_time = time.time()
            elif time.time() - no_hand_start_time >= NO_HAND_TIMEOUT_SECONDS:
                cleanup(cap, hands)
                return {
                    "modality": "gesture",
                    "status": "failure",
                    "reason": "No gesture detected"
                }

        if time.time() - start_time > MAX_TIMEOUT:
            cleanup(cap, hands)
            return {
                "modality": "gesture",
                "status": "failure",
                "reason": "Timeout"
            }

def cleanup(cap, hands):
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
