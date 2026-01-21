from fastapi import FastAPI
from Face_Verification import run_face_verification

app = FastAPI()

@app.post("/verify")
def verify_face():
    result = run_face_verification()
    return result