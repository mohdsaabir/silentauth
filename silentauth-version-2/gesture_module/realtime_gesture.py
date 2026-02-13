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

# ================= SILENCE LOGS =================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("absl").setLevel(logging.ERROR)

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get(
    "SILENTAUTH_DB_PATH",
    os.path.abspath(os.path.join(BASE_DIR, "..", "database", "central.db"))
)

VERIFICATION_TIME = 6
NO_GESTURE_TIMEOUT = 8  # Exit if no gesture detected for 10s
MIN_CONFIDENCE = 0.55
DISPLAY_AFTER = 2
SMOOTHING_FACTOR = 0.7

# ================= API OUTPUT TEMPLATE =================
def reset_output():
    return {
        "modality": "gesture",
        "gesture": None,
        "username": ["Unknown"],  # Only username list
        "confidence": 0.0,
        "status": "failure"
    }

# ================= ZMQ FRAME RECEIVER =================
context = zmq.Context()
frame_socket = context.socket(zmq.SUB)
frame_socket.setsockopt(zmq.CONFLATE, 1)
frame_socket.connect("tcp://localhost:5555")
frame_socket.setsockopt(zmq.SUBSCRIBE, b'')

# ================= DATABASE =================
def identify_user(gesture_name):
    if not gesture_name:
        return ["Unknown"]

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT users.name
        FROM gesture
        JOIN users ON gesture.user_id = users.user_id
        WHERE LOWER(TRIM(gesture_label)) = ?
    """, (gesture_name.lower().strip(),))
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows] if rows else ["Unknown"]

# ================= LOAD MODEL =================
clf = joblib.load("models/gesture_svm.pkl")
classes = clf.classes_

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ================= NORMALIZATION =================
def normalize(pts):
    pts = np.array(pts)
    base = pts[0]
    pts = pts - base
    scale = np.max(np.linalg.norm(pts, axis=1))
    return (pts / (scale + 1e-8)).flatten()

# ================= MAIN FUNCTION =================
def run_gesture_verification():
    api_output = reset_output()
    predictions = []
    confidences = []
    smoothed_pts = None
    start_time = None
    no_gesture_start = None

    # Flush old frames
    for _ in range(5):
        try:
            frame_socket.recv(flags=zmq.NOBLOCK)
        except:
            break

    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    window_name = "Gesture Verification"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    print("▶ Gesture verification started")

    try:
        while True:
            # Receive frame from ZMQ or fallback to any camera
            try:
                frame = pickle.loads(frame_socket.recv(flags=zmq.NOBLOCK))
            except zmq.Again:
                cap = cv2.VideoCapture(0)  # default webcam
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks and result.multi_handedness:
                hand_label = result.multi_handedness[0].classification[0].label
                lm = result.multi_hand_landmarks[0]

                if start_time is None:
                    start_time = time.time()
                    print(f"✋ {hand_label} hand detected — starting timer")

                # Reset no gesture timer
                no_gesture_start = None

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2),
                    mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
                )

                pts = np.array([(p.x, p.y, p.z) for p in lm.landmark])
                if hand_label == "Right":
                    pts[:, 0] = 1 - pts[:, 0]

                if smoothed_pts is None:
                    smoothed_pts = pts
                else:
                    smoothed_pts = SMOOTHING_FACTOR * smoothed_pts + (1 - SMOOTHING_FACTOR) * pts

                # Normalize and predict
                vec = normalize(smoothed_pts).reshape(1, -1)
                probs = clf.predict_proba(vec)[0]
                idx = np.argmax(probs)
                score = probs[idx]
                gesture = classes[idx]

                cv2.putText(
                    frame,
                    f"{hand_label}: {gesture} ({score:.2f})",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )

                if score >= MIN_CONFIDENCE:
                    predictions.append(gesture)
                    confidences.append(score)

            else:
                # Start no gesture timer if not already started
                if no_gesture_start is None:
                    no_gesture_start = time.time()
                else:
                    if time.time() - no_gesture_start > NO_GESTURE_TIMEOUT:
                        print("❌ No gesture detected for 10s — exiting")
                        break

                cv2.putText(
                    frame,
                    "Show your gesture",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(window_name, frame)

            if start_time and (time.time() - start_time >= VERIFICATION_TIME):
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # ================= FINAL DECISION =================
        if predictions:
            final_gesture = Counter(predictions).most_common(1)[0][0]
            selected_conf = [confidences[i] for i in range(len(predictions)) if predictions[i] == final_gesture]
            avg_confidence = float(np.mean(selected_conf))
            users = identify_user(final_gesture)

            api_output["gesture"] = final_gesture.lower()
            api_output["username"] = users
            api_output["confidence"] = round(avg_confidence, 2)
            api_output["status"] = "success" if "Unknown" not in users else "failure"
        else:
            api_output["username"] = ["Unknown"]
            api_output["status"] = "failure"

        print("FINAL OUTPUT:", api_output)

    finally:
        hands.close()
        cv2.destroyAllWindows()

    return api_output
