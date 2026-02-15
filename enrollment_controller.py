from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
import sqlite3
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "database/central.db"

# -------------------------------
# Request Model
# -------------------------------
class EnrollmentRequest(BaseModel):
    user_name: str
    preset: str
    gesture_name: str | None = None


# -------------------------------
# Utility: Update Preset
# -------------------------------
def update_user_preset(user_name, preset):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET preset=? WHERE name=?",
        (preset, user_name)
    )
    conn.commit()
    conn.close()


# -------------------------------
# STREAMING ENROLLMENT ENDPOINT
# -------------------------------
@app.post("/enroll_stream")
def enroll_user_stream(data: EnrollmentRequest):

    user_name = data.user_name.strip()
    preset = data.preset
    gesture_name = data.gesture_name

    if not user_name:
        raise HTTPException(status_code=400, detail="Username cannot be empty.")

    if preset not in ["normal", "voice_impaired", "motor_impaired"]:
        raise HTTPException(status_code=400, detail="Invalid preset.")

    def event_generator():

        # -------------------------------
        # FACE (Always)
        # -------------------------------
        with requests.post(
            "http://127.0.0.1:5004/enroll_stream",
            json={"user_name": user_name},
            stream=True
        ) as r:

            for line in r.iter_lines():
                if line:
                    yield line.decode() + "\n"

        # -------------------------------
        # VOICE (Conditional)
        # -------------------------------
        if preset in ["normal", "motor_impaired"]:

            with requests.post(
                "http://127.0.0.1:5002/enroll_stream",
                json={"user_name": user_name},
                stream=True
            ) as r:

                for line in r.iter_lines():
                    if line:
                        yield line.decode() + "\n"

        else:
            yield "data: [VOICE] Skipped (preset-based)\n\n"

        # -------------------------------
        # GESTURE (Conditional)
        # -------------------------------
        if preset in ["normal", "voice_impaired"]:

            if not gesture_name:
                yield "data: [GESTURE] Gesture name required.\n\n"
                return

            with requests.post(
                "http://127.0.0.1:5003/enroll_stream",
                json={
                    "user_name": user_name,
                    "gesture_name": gesture_name
                },
                stream=True
            ) as r:

                for line in r.iter_lines():
                    if line:
                        yield line.decode() + "\n"

        else:
            yield "data: [GESTURE] Skipped (preset-based)\n\n"

        # -------------------------------
        # Update Preset
        # -------------------------------
        update_user_preset(user_name, preset)

        yield "data: ENROLLMENT_COMPLETE\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
