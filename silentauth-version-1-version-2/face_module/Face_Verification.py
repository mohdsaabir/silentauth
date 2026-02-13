import cv2
import sqlite3
import numpy as np
import time
import os
import zmq
import pickle
from collections import deque
from insightface.app import FaceAnalysis
from blink_liveness import BlinkDetector
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preset_selector import preset_selector


# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.environ.get(
    "SILENTAUTH_DB_PATH",
    os.path.abspath(os.path.join(BASE_DIR, "..", "database", "central.db"))
)

THRESHOLD = 0.6
RUN_TIME = 10
BLINK_TIME_LIMIT = 2  # seconds

# ================= ZMQ FRAME RECEIVER =================
context = zmq.Context()
frame_socket = context.socket(zmq.SUB)
frame_socket.setsockopt(zmq.CONFLATE, 1)
frame_socket.connect("tcp://localhost:5555")
frame_socket.setsockopt(zmq.SUBSCRIBE, b'')

# ================= SIMILARITY =================
def cosine_similarity(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# ================= VERIFICATION CORE =================
def run_face_verification():

    api_output = {
        "modality": "face",
        "username": "Unknown",
        "confidence": 0.0,
        "status": "rejected",
        "preset": None
    }

    score_buffer = deque(maxlen=10)
    start_time = time.time()

    # --------- Load users from CENTRAL DB ---------
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT users.name, face.face_embedding
        FROM users
        JOIN face ON users.user_id = face.user_id
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("❌ No users found in DB")
        return api_output

    face_db = {name: np.frombuffer(blob, dtype=np.float32) for name, blob in rows}
    print("Loaded users:", list(face_db.keys()))

    # --------- InsightFace ---------
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))

    # --------- Blink / Liveness ---------
    blink_detector = BlinkDetector(threshold=2.5)
    blink_verified = False
    blink_start_time = None
    face_seen = False

    window_name = "Face Verification"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    print("▶ Face verification started")

    best_identity = None
    best_identity_score = 0.0

    while True:
        frame = pickle.loads(frame_socket.recv())
        faces = app.get(frame)

        # ================= NO FACE =================
        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow(window_name, frame)

            if time.time() - start_time >= RUN_TIME:
                break
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            continue

        # ================= FACE PRESENT =================
        for face in faces:
            kps = face.kps

            if not face_seen:
                face_seen = True
                blink_start_time = time.time()
                blink_detector.reset()
                print("👤 Face detected — starting blink timer")

            # ================= BLINK CHECK =================
            if not blink_verified:
                blinked = blink_detector.update(kps)
                if blinked:
                    blink_verified = True
                    print("✅ Blink detected — liveness confirmed")
                elif time.time() - blink_start_time > BLINK_TIME_LIMIT:
                    print("❌ No blink detected — rejecting")
                    cv2.destroyAllWindows()
                    return api_output
                cv2.putText(frame, "Blink to verify liveness",
                            (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)
                cv2.imshow(window_name, frame)
                continue

            # ================= IDENTITY CHECK =================
            emb = face.embedding.astype(np.float32)

            best_name = None
            best_score = -1.0
            for name, db_emb in face_db.items():
                score = cosine_similarity(emb, db_emb)
                if score > best_score:
                    best_score = score
                    best_name = name

            score_buffer.append(best_score)
            avg_score = sum(score_buffer) / len(score_buffer)

            x1, y1, x2, y2 = face.bbox.astype(int)
            if avg_score >= THRESHOLD:
                label = f"{best_name} ({avg_score:.2f})"
                color = (0, 255, 0)
                if avg_score > best_identity_score:
                    best_identity = best_name
                    best_identity_score = avg_score
            else:
                label = f"Unknown ({avg_score:.2f})"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow(window_name, frame)

        if time.time() - start_time >= RUN_TIME:
            break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # ================= FINAL OUTPUT =================
    if best_identity:
        api_output["username"] = best_identity
        api_output["confidence"] = round(float(best_identity_score), 2)
        api_output["status"] = "success"
        # ✅ Fetch preset via modular function
        api_output["preset"] = preset_selector(best_identity)
    else:
        api_output["preset"] = None

    return api_output
