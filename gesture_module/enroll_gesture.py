# enroll_gesture.py
import sqlite3
import os

def recognize_gesture():
    gestures = [
        "bang", "comehere", "fist", "four", "grab", "greeting",
        "loser", "love", "ok", "palm", "pinkyfinger",
        "pointingfinger", "rock", "three", "thumbsup",
        "thumbsdown", "victory", "w", "zero"
    ]
    print("Gestures available for enrollment:")
    for g in gestures:
        print(f"> {g}")

    gesture = input("Enter gesture name: ").strip()
    return gesture

def run_gesture_enrollment(user_name):
    """
    Enroll gesture for a given user_name.
    Returns the gesture label.
    """
    gesture_name = recognize_gesture()

    if not gesture_name:
        print("No gesture detected. Enrollment failed.")
        return None

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DB_PATH = os.environ.get(
        "SILENTAUTH_DB_PATH",
        os.path.abspath(os.path.join(BASE_DIR, "..", "database", "central.db"))
    )

    # Only insert user_name into users table if not exists
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    """
    cursor.execute("INSERT OR IGNORE INTO users (name) VALUES (?)", (user_name,))
    conn.commit()
    """
    # Get user_id
    cursor.execute("SELECT user_id FROM users WHERE name=?", (user_name,))
    user_id = cursor.fetchone()[0]

    # Insert gesture into gesture table
    cursor.execute(
        "INSERT OR REPLACE INTO gesture (user_id, gesture_label) VALUES (?, ?)",
        (user_id, gesture_name)
    )
    conn.commit()
    conn.close()

    print(f"Enrollment successful → {user_name} : {gesture_name}")
    return gesture_name


'''
if __name__ == "__main__":
    user_name = input("Enter user name for enrollment: ").strip()
    if user_name:
        run_gesture_enrollment(user_name)
    else:
        print("User name cannot be empty. Enrollment aborted.")
'''
