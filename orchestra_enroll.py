import requests

# Ask username once
user_name = input("Enter username to enroll: ").strip()

# --- Face Enrollment ---
try:
    res = requests.post("http://127.0.0.1:5001/enroll", json={"user_name": user_name})
    print("Face enrollment:", res.status_code, res.json())
except Exception as e:
    print("Face enrollment failed:", e)

# --- Voice Enrollment ---
try:
    res = requests.post("http://127.0.0.1:5002/enroll", json={"user_name": user_name})
    print("Voice enrollment:", res.status_code, res.json())
except Exception as e:
    print("Voice enrollment failed:", e)

# --- Gesture Enrollment ---
try:
    res = requests.post("http://127.0.0.1:5003/enroll", json={"user_name": user_name})
    print("Gesture enrollment:", res.status_code, res.json())
except Exception as e:
    print("Gesture enrollment failed:", e)

print("Enrollment completed!")