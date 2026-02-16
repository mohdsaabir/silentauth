# face_api.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from Face_Registration import run_face_enrollment

app = FastAPI()

# Request model
class FaceRequest(BaseModel):
    user_name: str

@app.post("/enroll_stream")
def enroll_stream(data: FaceRequest):

    def event_generator():
        for msg in run_face_enrollment(data.user_name):
            yield f"data: [FACE] {msg}\n\n"

        #yield "data: ENROLL_COMPLETE\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
