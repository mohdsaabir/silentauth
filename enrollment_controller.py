from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
import sqlite3
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
import config

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")



# As of now the endpoint /enroll is not used it is moved to system_controller.py but we keep it here for future
# use when we want to trigger enrollment from dashboard without page reload
@app.get("/enroll", response_class=HTMLResponse)
def load_enrollment_page(request: Request):
    return templates.TemplateResponse("enroll.html", {"request": request})


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



# Endpoint for sequential enrollment
@app.post("/enroll_stream")
def enroll_user_stream(data: EnrollmentRequest):

    user_name = data.user_name.strip()
    preset = data.preset
    gesture_name = data.gesture_name.lower()
    
    if not user_name:
        raise HTTPException(status_code=400, detail="Username cannot be empty.")

    if preset not in ["normal", "voice_impaired", "motor_impaired"]:
        raise HTTPException(status_code=400, detail="Invalid preset.")

    def event_generator():

        # -------------------------------
        # FACE (Always)
        # -------------------------------
        
        with requests.post(
            f"http://127.0.0.1:{config.FACE_ENROLL}/enroll_stream",
            json={"user_name": user_name},
            stream=True
        ) as r:

            for line in r.iter_lines():
                if line:
                    yield line.decode() + "\n\n"
        
        # -------------------------------
        # VOICE (Conditional)
        # -------------------------------
        
        if preset in ["normal", "motor_impaired"]:

            with requests.post(
                f"http://127.0.0.1:{config.VOICE_ENROLL}/enroll_stream",
                json={"user_name": user_name},
                stream=True
            ) as r:

                for line in r.iter_lines():
                    if line:
                        yield line.decode() + "\n\n"

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
                f"http://127.0.0.1:{config.GESTURE_ENROLL}/enroll_stream",
                json={
                    "user_name": user_name,
                    "gesture_name": gesture_name
                },
                stream=True
            ) as r:

                for line in r.iter_lines():
                    if line:
                        yield line.decode() + "\n\n"

        else:
            yield "data: [GESTURE] Skipped (preset-based)\n\n"

        # -------------------------------
        # Update Preset
        # -------------------------------
        update_user_preset(user_name, preset)

        yield "data: ENROLLMENT COMPLETED SUCCESSFULLY\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
