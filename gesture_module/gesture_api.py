from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from enroll_gesture import run_gesture_enrollment

app = FastAPI()

class GestureRequest(BaseModel):
    user_name: str
    gesture_name: str

@app.post("/enroll_stream")
def enroll_stream(data: GestureRequest):

    def event_generator():
        for msg in run_gesture_enrollment(data.user_name, data.gesture_name):
            yield f"data: [GESTURE] {msg}\n\n"
        yield "data: ENROLL_COMPLETE\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
