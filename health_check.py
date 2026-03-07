import socket
from system_manager import services
from system_manager import start_camera_service
import config

SERVICES = {
    "camera_service": config.CAMERA_SERVICE,
    "face_verify": config.FACE_VERIFY,
    "voice_verify": config.VOICE_VERIFY,
    "gesture_verify": config.GESTURE_VERIFY,
    "verification_controller": config.VERIFICATION_CONTROLLER,
    "face_enroll": config.FACE_ENROLL,
    "voice_enroll": config.VOICE_ENROLL,
    "gesture_enroll": config.GESTURE_ENROLL,
    "enrollment_controller": config.ENROLLMENT_CONTROLLER
}

CONTEXT_SERVICES = {
    "verify": [
        "camera_service",
        "face_verify",
        "voice_verify",
        "gesture_verify",
        "verification_controller"
    ],
    "enroll": [
        "face_enroll",
        "voice_enroll",
        "gesture_enroll",
        "enrollment_controller"
    ]
}


def is_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except:
        return False


# Logic to check health of system and serivces and return context based result
def get_system_status(context=None):

    status = {}

    # Decide which services to check
    print(f"Checking system status for context: {context}")
    if context in CONTEXT_SERVICES:
        services_to_check = CONTEXT_SERVICES[context]
    else:
        # dashboard or unknown context,  show all
        services_to_check = SERVICES.keys()

    for svc in services_to_check:

        port = SERVICES[svc]
        running = is_port_open("127.0.0.1", port)

        # Only restart camera during verification
        if svc == "camera_service" and not running and context == "verify":
            print("Camera service stopped → restarting for verification")
            start_camera_service()
            running = True

        status[svc] = {
            "running": running,
            "port": port
        }

    return {
        "mode": context if context else "dashboard",
        "services": status
    }