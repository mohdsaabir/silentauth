import subprocess
import os
import time

# Store running processes
services = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Logic of start camera service for verification
def start_camera_service():
    name = "camera_service"

    if name in services and services[name].poll() is None:
        print("[SERVICE] Camera already running")
        return

    python_path = os.path.join(BASE_DIR, "env", "Scripts", "python.exe")
    cmd = [python_path, "camera_service.py"]

    process = subprocess.Popen(
        cmd,
        cwd=BASE_DIR,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
    )

    services[name] = process
    print("[SERVICE] Camera service started")


# Logic of stop camera serivce after verification page exit
def stop_camera_service(name="camera_service"):
   
    print("Stopping camera service...")
    process = services.get(name)

    if process and process.poll() is None:
        pid = process.pid
        os.system(f"taskkill /PID {pid} /F /T")
        print(f"[SERVICE] {name} stopped")
        services.pop(name)
        time.sleep(0.3)
    else:
        print(f"[SERVICE] {name} not running")
