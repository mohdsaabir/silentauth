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
import zmq
import pickle

# ================= ZMQ FRAME RECEIVER =================
context = zmq.Context()
frame_socket = context.socket(zmq.SUB)
frame_socket.setsockopt(zmq.CONFLATE, 1)
frame_socket.connect("tcp://localhost:5555")
frame_socket.setsockopt(zmq.SUBSCRIBE, b'')

# ------------------- SILENCE LOGS -------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("absl").setLevel(logging.ERROR)

# ------------------- DATABASE -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.environ.get(
    "SILENTAUTH_DB_PATH",
    os.path.abspath(os.path.join(BASE_DIR, "..", "database", "central.db"))
)

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
CONFIDENCE_THRESHOLD = 0.6
GESTURE_WINDOW_SECONDS = 3
MIN_DOMINANCE_RATIO = 0.7
HAND_LOSS_GRACE_SECONDS = 0.5
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
    PREDICTIONS = []          # [(gesture, score)]
    gesture_start_time = None
    no_hand_start_time = None

    window_name = "Gesture Verification"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    hands = mp_hands.Hands(max_num_hands=1)
    start_time = time.time()

    while True:
        frame = pickle.loads(frame_socket.recv())
        cv2.imshow(window_name, frame)

        # ESC to exit
        if cv2.waitKey(10) & 0xFF == 27:
            cleanup(hands)
            return {
                "modality": "gesture",
                "status": "failure",
                "reason": "User cancelled"
            }

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # ---------------- HAND DETECTED ----------------
        if result.multi_hand_landmarks:
            no_hand_start_time = None

            if gesture_start_time is None:
                gesture_start_time = time.time()

            lm = result.multi_hand_landmarks[0]
            pts = [(p.x, p.y, p.z) for p in lm.landmark]

            vec = normalize(pts).reshape(1, -1)
            probs = clf.predict_proba(vec)[0]

            idx = np.argmax(probs)
            score = probs[idx]
            gesture = classes[idx]

            if score >= CONFIDENCE_THRESHOLD:
                PREDICTIONS.append((gesture, score))

            # -------- TIME WINDOW COMPLETE --------
            if (
                time.time() - gesture_start_time >= GESTURE_WINDOW_SECONDS
                and len(PREDICTIONS) > 0
            ):
                gestures = [g for g, _ in PREDICTIONS]
                counts = Counter(gestures)

                final_gesture, dominant_count = counts.most_common(1)[0]
                dominance_ratio = dominant_count / len(gestures)

                if dominance_ratio >= MIN_DOMINANCE_RATIO:
                    avg_score = np.mean(
                        [s for g, s in PREDICTIONS if g == final_gesture]
                    )

                    user_names = identify_user(final_gesture)
                    cleanup(hands)

                    return {
                        "modality": "gesture",
                        "gesture": final_gesture,
                        "username": user_names,
                        "confidence": round(float(avg_score), 2),
                        "status": "success"
                        if "Unknown user" not in user_names
                        else "failure"
                    }
                else:
                    cleanup(hands)
                    return {
                        "modality": "gesture",
                        "status": "failure",
                        "reason": "Unstable gesture"
                    }

        # ---------------- NO HAND ----------------
        else:
            if no_hand_start_time is None:
                no_hand_start_time = time.time()

            # tolerate brief hand loss
            elif time.time() - no_hand_start_time > HAND_LOSS_GRACE_SECONDS:
                gesture_start_time = None
                PREDICTIONS.clear()

            # hard timeout
            if time.time() - no_hand_start_time >= NO_HAND_TIMEOUT_SECONDS:
                cleanup(hands)
                return {
                    "modality": "gesture",
                    "status": "failure",
                    "reason": "No gesture detected"
                }

        # ---------------- GLOBAL TIMEOUT ----------------
        if time.time() - start_time > MAX_TIMEOUT:
            cleanup(hands)
            return {
                "modality": "gesture",
                "status": "failure",
                "reason": "Timeout"
            }

# ------------------- CLEANUP -------------------
def cleanup(hands):
    hands.close()
    cv2.destroyAllWindows()
