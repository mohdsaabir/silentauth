import cv2
import time
import sqlite3
import numpy as np
from insightface.app import FaceAnalysis
import os

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.environ.get(
    "SILENTAUTH_DB_PATH",
    os.path.abspath(os.path.join(BASE_DIR, "..", "database", "central.db"))
)
REGISTRATION_TIME = 8


def run_face_enrollment(user_name):
    """
    Enroll face for given user_name.
    Returns dict with user_id on success, None on failure.
    """
    yield f"Registering user: {user_name}"
    print(f"Registering user: {user_name}")
    print("Look at the camera normally...")
    yield "Look at the camera normally..."
    # -------- Face registration core --------
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return None

    window_name = "Face Registration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    embeddings = []
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)

        if len(faces) == 1:
            face = faces[0]
            embeddings.append(face.embedding)

            x1, y1, x2, y2 = face.bbox.astype(int)
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

        cv2.imshow(window_name, frame)

        if time.time() - start_time >= REGISTRATION_TIME:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) == 0:
        yield "No face data captured. Enrollment failed."
        print("No embeddings collected. Registration failed.")
        return None

    final_embedding = np.mean(np.array(embeddings), axis=0)

    # -------- Database operations --------
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Insert user if not exists
    cursor.execute(
        "INSERT OR IGNORE INTO users (name) VALUES (?)",
        (user_name,)
    )
    conn.commit()

    # Get user_id
    cursor.execute(
        "SELECT user_id FROM users WHERE name=?",
        (user_name,)
    )
    user_id = cursor.fetchone()[0]

    # Store face embedding
    cursor.execute(
        "INSERT OR REPLACE INTO face (user_id, face_embedding) VALUES (?, ?)",
        (user_id, final_embedding.astype("float32").tobytes())
    )

    conn.commit()
    conn.close()
    #yield "Face enrollment successful"
    print(f"Enrollment successful → {user_name} (user_id={user_id})")
    yield f"Face enrollment successful for user: {user_name}"
    return {"user_id": user_id}


'''

if __name__ == "__main__":
    user_name = input("Enter user name for registration: ")
    run_face_enrollment(user_name)

'''