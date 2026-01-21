import cv2
import sqlite3
import numpy as np
import time
import json
from collections import deque
from insightface.app import FaceAnalysis

# ================= CONFIG =================
DB_PATH = "--------------------------------------"
THRESHOLD = 0.6
RUN_TIME = 10

score_buffer = deque(maxlen=10)
last_printed_name = None

api_output = {
    "modality": "face",
    "username": "Nil",
    "confidence": 0,
    "status": "Nil"
}

flag = 0

# ---------- Similarity ----------
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# ================= VERIFICATION CORE =================
def run_face_verification():

    global last_printed_name, flag
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

    if len(rows) == 0:
        print("No users found in DB.")
        return api_output

    face_db = {}
    for name, blob in rows:
        face_db[name] = np.frombuffer(blob, dtype=np.float32)

    print("Loaded users:", list(face_db.keys()))

    # --------- InsightFace ---------
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return api_output

    print("Realtime face verification started...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)

        for face in faces:
            emb = face.embedding

            best_name = None
            best_score = -1

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

                if best_name != last_printed_name:
                    api_output["username"] = best_name
                    api_output["confidence"] = round(float(avg_score), 2)
                    api_output["status"] = "success"
                    print(json.dumps(api_output))
                    last_printed_name = best_name
                    flag = 0
            else:
                if not flag:
                    label = f"Unknown ({avg_score:.2f})"
                    api_output["username"] = "Unknown"
                    api_output["confidence"] = round(float(avg_score), 2)
                    api_output["status"] = "rejected"
                    print(json.dumps(api_output))
                    color = (0, 0, 255)
                    flag = 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Face Verification", frame)

        if time.time() - start_time >= RUN_TIME:
            print("Time limit reached.")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return api_output