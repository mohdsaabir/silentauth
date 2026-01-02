import sqlite3

def recognize_gesture():
    gesture = input("Enter detected gesture name (temporary): ")
    return gesture


def enroll_user():
    user_name = input("Enter username to enroll: ")
    gesture_name = recognize_gesture()

    if not gesture_name:
        print("No gesture detected. Enrollment failed.")
        return

    conn = sqlite3.connect('gesture_users.db')
    cursor = conn.cursor()

    cursor.execute('''
    INSERT OR REPLACE INTO users (user_name, gesture_name)
    VALUES (?, ?)
    ''', (user_name, gesture_name))

    conn.commit()
    conn.close()

    print(f"Enrollment successful → {user_name} : {gesture_name}")


if __name__ == "__main__":
    enroll_user()