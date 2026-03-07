from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
import os
from starlette.staticfiles import StaticFiles
from health_check import get_system_status
from system_manager import start_camera_service, stop_camera_service
from fastapi import Request
from db_setup import create_central_db
import sqlite3

# Checks and creates database on startup
def ensure_database():
    db_path = os.path.join("database", "central.db")

    if not os.path.exists(db_path):
        print("Database not found. Creating database...")

        # create folder if it doesn't exist
        os.makedirs("database", exist_ok=True)

        create_central_db()

        print("Central Database created successfully.")
    else:
        print("Database already exists.")



app = FastAPI()
TEMPLATE_DIR = "templates"

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def startup_event():
    ensure_database()

# Dashboard endpoint
@app.get("/", response_class=HTMLResponse)
def dashboard():
    with open(os.path.join(TEMPLATE_DIR, "dashboard.html")) as f:
        return f.read()



# Endpoint that Starts camera serivce when hit verification
@app.post("/start-verification")
def start_verify():
    start_camera_service()
    return RedirectResponse(url="/verify", status_code=303)



@app.get("/enroll", response_class=HTMLResponse)
def enroll_page():
    with open(os.path.join(TEMPLATE_DIR, "enroll.html")) as f:
        return f.read()
    

@app.get("/verify", response_class=HTMLResponse)
def verify_page():
    with open(os.path.join(TEMPLATE_DIR, "verify.html")) as f:
        return f.read()


# Endpoint Stops camera service when exit from verification page
@app.post("/stop-verification")
def stop_verify():
    stop_camera_service()
    return {"status": "camera_stopped"}


# Endpoint to check health of system
@app.get("/system-status")
def system_status(request: Request, context: str = None):
    return get_system_status(context)





