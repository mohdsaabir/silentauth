# enroll_gesture.py
import sqlite3
import os

ALLOWED_GESTURES = [
    "bang", "comehere", "fist", "four", "greeting",
    "loser", "ILoveU", "ok", "pinkyfinger",
    "pointingfinger", "three", "thumbsup",
    "thumbsdown", "victory", "zero"
]

def run_gesture_enrollment(user_name, gesture_name):
    """
    Enroll gesture for a given user_name and gesture_name.
    Returns the gesture label.
    """

    if gesture_name not in ALLOWED_GESTURES:
        return None

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DB_PATH = os.environ.get(
        "SILENTAUTH_DB_PATH",
        os.path.abspath(os.path.join(BASE_DIR, "..", "database", "central.db"))
    )

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT user_id FROM users WHERE name=?", (user_name,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None

    user_id = row[0]

    cursor.execute(
        "INSERT OR REPLACE INTO gesture (user_id, gesture_label) VALUES (?, ?)",
        (user_id, gesture_name)
    )
    conn.commit()
    conn.close()
    yield f"Gesture '{gesture_name}' enrolled for user: {user_name}"
    return gesture_name
