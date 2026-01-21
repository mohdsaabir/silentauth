# gesture_api.py
from fastapi import FastAPI
from enroll_gesture import run_gesture_enrollment

app = FastAPI()

@app.post("/enroll")
def enroll_gesture(data: dict):
    user_name = data["user_name"]

    # Run gesture enrollment (returns gesture label)
    gesture_label = run_gesture_enrollment(user_name)

    if not gesture_label:
        return {"status": "failed", "message": "Gesture enrollment failed"}

    return {"status": "gesture enrolled", "user_name": user_name, "gesture_label": gesture_label}