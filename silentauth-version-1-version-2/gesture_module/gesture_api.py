# gesture_api.py
from fastapi import FastAPI, HTTPException
from enroll_gesture import run_gesture_enrollment, ALLOWED_GESTURES

app = FastAPI()

@app.get("/gestures")
def list_gestures():
    return {"gestures": ALLOWED_GESTURES}

@app.post("/enroll")
def enroll_gesture(data: dict):
    user_name = data.get("user_name")
    gesture_name = data.get("gesture_name")

    if not user_name or not gesture_name:
        raise HTTPException(status_code=400, detail="user_name and gesture_name required")

    result = run_gesture_enrollment(user_name, gesture_name)

    if not result:
        raise HTTPException(status_code=400, detail="Invalid gesture or enrollment failed")

    return {
        "status": "gesture enrolled",
        "user_name": user_name,
        "gesture_label": result
    }
