from fastapi import FastAPI
from multimodal_executor import orchestrate_parallel
from fusion_engine import fuse_results_with_preset
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from verification_logger import log_verification
import sqlite3

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# As of now the endpoint /verify is not used it is moved to system_controller.py but we keep it here for future
# use when we want to trigger verification from dashboard without page reload
@app.get("/verify", response_class=HTMLResponse)
def load_verification_page(request: Request):
    return templates.TemplateResponse("verify.html", {"request": request})



# Endpoint to get the verification log frotend
@app.get("/logs", response_class=HTMLResponse)
def logs_page(request: Request):
    return templates.TemplateResponse("logs.html", {"request": request})



# Endpoint to fusion engine call and get the result of verification
@app.post("/verify_endpoint")
def verify_user():

    results = orchestrate_parallel()

    result_map = {res["modality"]: res for res in results}

    face_res = result_map.get("face", {})
    voice_res = result_map.get("voice", {})
    gesture_res = result_map.get("gesture", {})


    fused = fuse_results_with_preset(face_res, voice_res, gesture_res)

    print("\nFused Result:")
    print(fused)

    if fused["status"] == "success":
        log_verification(fused["username"], True)
        return {
            "status": "success",
            "username": fused["username"],
            "score": fused["fusion_score"], 
        }
    
    log_verification(None, False)
    return {"status": "failure"}



# Endpoint and Logic to get verification log 
@app.get("/verification-logs")
def get_logs():

    conn = sqlite3.connect("database/central.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT timestamp, username, result
        FROM verification_logs
        ORDER BY id DESC
        LIMIT 50
    """)

    rows = cursor.fetchall()
    conn.close()

    logs = []

    for r in rows:
        logs.append({
            "timestamp": r[0],
            "username": r[1],
            "result": r[2]
        })

    return {"logs": logs}