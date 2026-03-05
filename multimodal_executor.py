import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

#--------VERIFY API ENDPOINTS-----------------------
FACE_VERIFY_URL = "http://localhost:5010/verify"
VOICE_VERIFY_URL = "http://localhost:5013/verify"
GESTURE_VERIFY_URL = "http://localhost:5012/verify"
#---------------------------------------------------

#---------FUNCTION TO CALLING APIS-------------
def call_api(name, url):
    print(f"\nCalling {name} Verification API...")
    try:
        response = requests.post(url, timeout=40)
        response.raise_for_status()
        result = response.json()
        print(f"{name} Output:")
        print(json.dumps(result, indent=2))
        return result
    except Exception as e:
        print(f"{name} API error:", e)
        return {
            "modality": name.lower(),
            "confidence": 0,
            "status": "error"
        }




#--------------PARALLEL RUNNER----------------
def orchestrate_parallel():
#    print("\nMULTI-MODAL PARALLEL ORCHESTRATION STARTED\n")
    
    tasks = {
        "Face": FACE_VERIFY_URL,
        "Gesture": GESTURE_VERIFY_URL,
        "Voice": VOICE_VERIFY_URL
    }

    results = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {
            executor.submit(call_api, name, url): name
            for name, url in tasks.items()
        }

        for future in as_completed(future_map):
            results.append(future.result())

    return results

if __name__ == "__main__":
    orchestrate_parallel()



   