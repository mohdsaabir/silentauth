import cv2
import time
import sqlite3
import numpy as np
from insightface.app import FaceAnalysis

DB_PATH = "face_auth.db" # face db
REGISTRATION_TIME = 8  # Runs 8 seconds collecting max frames : more frames = more accuracy

# connects to database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    embedding BLOB
)
""")
conn.commit()

username = input("Enter username to register: ").strip()

#-------- Registration core ------------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

cap = cv2.VideoCapture(0) # Live Registration
if not cap.isOpened():
    print("Cannot open camera")
    exit()

embeddings = []
start_time = time.time()

print(f"Registering user: {username}")
print("Look at the camera normally...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame) # gets faces

    if len(faces) == 1:
        face = faces[0]
        embeddings.append(face.embedding)

        x1, y1, x2, y2 = face.bbox.astype(int)

        #bounding box around faces
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "REGISTERING...",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
    else:
        cv2.putText(
            frame,
            "Ensure ONLY one face",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    cv2.imshow("Face Registration", frame)
    cv2.setWindowProperty("Face Registration", cv2.WND_PROP_TOPMOST, 1)


    if time.time() - start_time >= REGISTRATION_TIME:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


if len(embeddings) == 0:
    print("No embeddings collected. Registration failed.")
    exit()

final_embedding = np.mean(np.array(embeddings), axis=0)

# inserts into db
cursor.execute(
    "INSERT OR REPLACE INTO users (username, embedding) VALUES (?, ?)",
    (username, final_embedding.astype(np.float32).tobytes())
)

conn.commit()
conn.close()

print(f"User '{username}' registered successfully")
print("Embedding shape:", final_embedding.shape)
