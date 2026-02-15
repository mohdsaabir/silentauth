from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ecapa_realtime_register import run_voice_enrollment

app = FastAPI()

class VoiceRequest(BaseModel):
    user_name: str

@app.post("/enroll_stream")
def enroll_stream(data: VoiceRequest):

    def event_generator():
        for msg in run_voice_enrollment(data.user_name):
            yield f"data: [VOICE] {msg}\n\n"

        yield "data: ENROLL_COMPLETE\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
