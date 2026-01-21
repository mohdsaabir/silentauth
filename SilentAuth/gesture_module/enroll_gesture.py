# enroll_gesture.py
import sqlite3

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

    # Only insert user_name into users table if not exists
    conn = sqlite3.connect("-----------DBPATH----------")
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