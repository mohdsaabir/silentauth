# voice_api.py
from fastapi import FastAPI
from ecapa_realtime_register import run_voice_enrollment

app = FastAPI()

@app.post("/enroll")
def enroll_voice(data: dict):
    user_name = data["user_name"]

    result = run_voice_enrollment(user_name)

    if not result:
        return {"status": "failed", "message": "Voice enrollment failed"}

    return {
        "status": "voice enrolled",
        "user_name": user_name,
        "user_id": result["user_id"]
    }
