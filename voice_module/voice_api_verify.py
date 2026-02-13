from fastapi import FastAPI
from ecapa_realtime_verify_db import run_voice_verification

app = FastAPI()

@app.post("/verify")
def verify_voice():
    result = run_voice_verification()
    return result