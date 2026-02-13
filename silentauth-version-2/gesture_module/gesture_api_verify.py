from fastapi import FastAPI
from realtime_gesture import run_gesture_verification

app = FastAPI()

@app.post("/verify")
def verify_gesture():
    return run_gesture_verification()