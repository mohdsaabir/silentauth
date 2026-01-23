import requests
import json
import time

# ------------------ VERIFY API ENDPOINTS ------------------
FACE_VERIFY_URL = "http://localhost:5001/verify"
VOICE_VERIFY_URL = "http://localhost:5002/verify"
GESTURE_VERIFY_URL = "http://localhost:5003/verify"
# ---------------------------------------------------------

def call_api(name, url):
    print(f"\nCalling {name} Verification API...")
    try:
        response = requests.post(url, timeout=30)
        response.raise_for_status()
        result = response.json()
        print(f"{name} Output:")
        print(json.dumps(result, indent=2))
        return result
    except requests.exceptions.RequestException as e:
        print(f"{name} API error:", e)
        return {
            "modality": name.lower(),
            "user_id": None,
            "confidence": 0,
            "status": "error"
        }
    except json.JSONDecodeError as e:
        print(f"{name} API JSON decode error:", e)
        return {
            "modality": name.lower(),
            "user_id": None,
            "confidence": 0,
            "status": "error"
        }

def orchestrate_verification():
    print("\nMULTI-MODAL ORCHESTRATION STARTED\n")

    results = []

    # ---- FACE ----
    results.append(call_api("Face", FACE_VERIFY_URL))

    # ---- VOICE ----
    results.append(call_api("Voice", VOICE_VERIFY_URL))

    # ---- GESTURE ----
    results.append(call_api("Gesture", GESTURE_VERIFY_URL))

    return results

if __name__ == "__main__":
    orchestrate_verification()
