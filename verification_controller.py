from fastapi import FastAPI
from multimodal_executor import orchestrate_parallel
from fusion_engine import fuse_results_with_preset
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from fusion_engine import fuse_and_log


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/verify", response_class=HTMLResponse)
def load_verification_page(request: Request):
    return templates.TemplateResponse("verify.html", {"request": request})



@app.post("/verify_endpoint")
def verify_user():

    results = orchestrate_parallel()

    result_map = {res["modality"]: res for res in results}

    face_res = result_map.get("face", {})
    voice_res = result_map.get("voice", {})
    gesture_res = result_map.get("gesture", {})

    #fused = fuse_results_with_preset(face_res, voice_res, gesture_res)

    fused = fuse_and_log(
        face_res,
        voice_res,
        gesture_res,
        ground_truth="impostor",      # or "impostor"
        attack_type="spoof_voice"         # or "unknown", "spoof_voice"
    )

    print("\nFused Result:")
    print(fused)

    if fused["status"] == "success":
        return {
            "status": "success",
            "username": fused["username"],
            "score": fused["fusion_score"], 
        }

    return {"status": "failure"}
