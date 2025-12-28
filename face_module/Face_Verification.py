import cv2
import sqlite3
import numpy as np
import time
from collections import deque
from insightface.app import FaceAnalysis

DB_PATH = "face_auth.db" # db to store the face embeddings
THRESHOLD = 0.6 # decision key to identification
RUN_TIME = 10  # live window run time in seconds

score_buffer = deque(maxlen=10) # to compute average of several frames
last_printed_name = None
start_time = time.time()

#---------- Similarity Score Core----------
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# --------- Load users from SQLite ---------
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("SELECT username, embedding FROM users")
rows = cursor.fetchall()
conn.close()

if len(rows) == 0:
    print("No users found in DB. Register first.")
    exit()

face_db = {}
for username, blob in rows:
    face_db[username] = np.frombuffer(blob, dtype=np.float32)

print("Loaded users:", list(face_db.keys()))

# --------- InsightFace Core---------
app = FaceAnalysis(name="buffalo_l") #loads model
app.prepare(ctx_id=0, det_size=(640, 640)) 

cap = cv2.VideoCapture(0) # to live capture
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Realtime 1 vs N face verification started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame) #finds faces
    

    for face in faces:
        emb = face.embedding #generate embeddings

        best_name = None
        best_score = -1

        # finds the best match from db
        for name, db_emb in face_db.items():
            score = cosine_similarity(emb, db_emb)
            if score > best_score:
                best_score = score
                best_name = name

        # Score averaging across multiple frames
        score_buffer.append(best_score)
        avg_score = sum(score_buffer) / len(score_buffer)

        x1, y1, x2, y2 = face.bbox.astype(int)

        # identification decision
        if avg_score >= THRESHOLD:
            label = f"{best_name} ({avg_score:.2f})"
            color = (0, 255, 0)

            # Print only once per person
            if best_name != last_printed_name:
                print(f"[IDENTIFIED] User: {best_name}, Score: {avg_score:.2f}")
                last_printed_name = best_name

        else:
            label = f"UNKNOWN ({avg_score:.2f})"
            color = (0, 0, 255)

        # bounding box around the face
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow("Face Verification (SQLite | 1 vs N)", frame)

    # Auto close after 12 seconds
    if time.time() - start_time >= RUN_TIME:
        print("⏱️ Time limit reached. Closing verification window.")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
