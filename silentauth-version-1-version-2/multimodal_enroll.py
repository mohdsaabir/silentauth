import requests
import sqlite3
import sys

DB_PATH = "database/central.db"

# -------------------------------
# Utility: Update Preset in DB
# -------------------------------
def update_user_preset(user_name, preset):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE users SET preset=? WHERE name=?",
            (preset, user_name)
        )

        conn.commit()
        conn.close()
        print(f"Preset '{preset}' stored for user '{user_name}'")

    except Exception as e:
        print("Failed to update preset:", e)


# -------------------------------
# Start Enrollment
# -------------------------------
user_name = input("Enter username to enroll: ").strip()

if not user_name:
    print("Username cannot be empty.")
    sys.exit(1)

print("\nSelect Enrollment Preset:")
print("1. normal (Face + Voice + Gesture)")
print("2. voice_impaired (Face + Gesture)")
print("3. motor_impaired (Face + Voice)")

choice = input("Enter choice (1/2/3): ").strip()

preset_map = {
    "1": "normal",
    "2": "voice_impaired",
    "3": "motor_impaired"
}

preset = preset_map.get(choice)

if not preset:
    print("Invalid choice.")
    sys.exit(1)

print(f"\nSelected Preset: {preset}\n")

# -------------------------------
# FACE ENROLLMENT (Always Required)
# -------------------------------
try:
    res = requests.post(
        "http://127.0.0.1:5001/enroll",
        json={"user_name": user_name}
    )
    print("Face enrollment:", res.status_code, res.json())
except Exception as e:
    print("Face enrollment failed:", e)
    sys.exit(1)


# -------------------------------
# VOICE ENROLLMENT (Conditional)
# -------------------------------
if preset in ["normal", "motor_impaired"]:
    try:
        res = requests.post(
            "http://127.0.0.1:5002/enroll",
            json={"user_name": user_name}
        )
        print("Voice enrollment:", res.status_code, res.json())
    except Exception as e:
        print("Voice enrollment failed:", e)
else:
    print("Skipping Voice enrollment (preset-based)")


# -------------------------------
# GESTURE ENROLLMENT (Conditional)
# -------------------------------
if preset in ["normal", "voice_impaired"]:
    try:
        res = requests.get("http://127.0.0.1:5003/gestures")
        gestures = res.json()["gestures"]

        print("\nAvailable gestures:")
        for g in gestures:
            print(f"> {g}")

        gesture_name = input("\nEnter gesture to enroll: ").strip()

        res = requests.post(
            "http://127.0.0.1:5003/enroll",
            json={
                "user_name": user_name,
                "gesture_name": gesture_name
            }
        )

        print("Gesture enrollment:", res.status_code, res.json())

    except Exception as e:
        print("Gesture enrollment failed:", e)
else:
    print("Skipping Gesture enrollment (preset-based)")


# -------------------------------
# Store Preset in Database
# -------------------------------
update_user_preset(user_name, preset)

print("\nEnrollment completed successfully!")
