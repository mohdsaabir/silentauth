from fastapi import FastAPI
from multimodal_executor import orchestrate_parallel,orchestrate
from fusion_engine import fuse_results
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/verify")
def verify_user():

    results = orchestrate_parallel()

    result_map = {res["modality"]: res for res in results}

    face_res = result_map.get("face", {})
    voice_res = result_map.get("voice", {})
    gesture_res = result_map.get("gesture", {})


    fused = fuse_results(face_res, voice_res, gesture_res)

    if fused["status"] == "success":
        return {
            "status": "success",
            "username": fused["username"],
            "score": fused["fusion_score"], 
        }

    return {"status": "failure"}
