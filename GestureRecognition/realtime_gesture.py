import os
import warnings
import logging

# ------------------- SILENCE LOGS -------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("absl").setLevel(logging.ERROR)

# ------------------- IMPORTS -------------------
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import Counter
import sqlite3

# ------------------- DATABASE FUNCTION -------------------
def identify_user(gesture_name):
    conn = sqlite3.connect('gesture_users.db')
    cursor = conn.cursor()
    cursor.execute(
        "SELECT user_name FROM users WHERE gesture_name = ?",
        (gesture_name,)
    )
    result = cursor.fetchall()
    conn.close()

    if result:
        return [row[0] for row in result]  # list of usernames
    else:
        return []

# ------------------- LOAD MODEL -------------------
clf = joblib.load("models/gesture_svm.pkl")
classes = clf.classes_

# ------------------- MEDIAPIPE SETUP -------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# ------------------- CAMERA -------------------
cap = cv2.VideoCapture(0)

# ------------------- PREDICTION SETTINGS -------------------
PREDICTIONS = []
REQUIRED_FRAMES = 5
CONFIDENCE_THRESHOLD = 0.6

# ------------------- NORMALIZATION FUNCTION -------------------
def normalize(pts):
    pts = np.array(pts)
    base = pts[0]
    pts = pts - base
    scale = np.max(np.linalg.norm(pts, axis=1))
    return (pts / (scale + 1e-8)).flatten()

print("✅ Camera started. Show your gesture...")

# ------------------- MAIN LOOP -------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0]
        pts = [(p.x, p.y, p.z) for p in lm.landmark]

        vec = normalize(pts).reshape(1, -1)

        probs = clf.predict_proba(vec)[0]
        idx = np.argmax(probs)
        score = probs[idx]
        gesture = classes[idx]

        if score >= CONFIDENCE_THRESHOLD:
            PREDICTIONS.append(gesture)

        # ------------------- FINAL DECISION -------------------
        if len(PREDICTIONS) >= REQUIRED_FRAMES:
            final_gesture = Counter(PREDICTIONS).most_common(1)[0][0]
            final_score = score

            # DATABASE LOOKUP
            user_name = identify_user(final_gesture)

            output = {
                "modality": "gesture",
                "gesture": final_gesture.lower(),
                "user_id": [user_name],
                "confidence": round(float(final_score), 2),
                "status": "success" if user_name != "Unknown user" else "failure"
                 }
            print(output)
            break

    cv2.imshow("Gesture Authentication", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("❌ Cancelled by user")
        break

# ------------------- CLEANUP -------------------
cap.release()
hands.close()
cv2.destroyAllWindows()