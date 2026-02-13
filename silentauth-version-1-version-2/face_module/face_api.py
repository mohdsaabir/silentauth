# face_api.py
from fastapi import FastAPI
from Face_Registration import run_face_enrollment

app = FastAPI()

@app.post("/enroll")
def enroll_face(data: dict):
    user_name = data["user_name"]

    result = run_face_enrollment(user_name)

    if not result:
        return {"status": "failed", "message": "Face enrollment failed"}

    return {
        "status": "face enrolled",
        "user_name": user_name,
        "user_id": result["user_id"]
    }
