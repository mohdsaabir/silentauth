from multimodal_executor import orchestrate_parallel

# ======================= CONFIG =======================
FUSION_THRESHOLD = 0.60

FACE_WEIGHT = 0.5
VOICE_WEIGHT = 0.3
GESTURE_WEIGHT = 0.2


def fuse_results(face_res, voice_res, gesture_res):

    # Extract users from each modality
    face_user = face_res.get("username") if face_res.get("status") == "success" else None
    voice_user = voice_res.get("username") if voice_res.get("status") == "success" else None
    gesture_users = gesture_res.get("username", []) if gesture_res.get("status") == "success" else []

    
    # Anchor gesture with face or voice
    if face_user:
        anchor_user = face_user
        if anchor_user in gesture_users:
            gesture_conf = gesture_res.get("confidence", 0)
        else:
            gesture_conf = 0
    elif voice_user:
        anchor_user = voice_user
        if anchor_user in gesture_users:
            gesture_conf = gesture_res.get("confidence", 0)
        else:
            gesture_conf = 0
    else:
        # No face or voice, cannot authenticate reliably
        anchor_user = None
        gesture_conf = 0

    # Collect confidence for each modality
    face_conf = face_res.get("confidence", 0) 
    voice_conf = voice_res.get("confidence", 0)

    # Computing fusion score
    fusion_score = FACE_WEIGHT*face_conf + VOICE_WEIGHT*voice_conf + GESTURE_WEIGHT*gesture_conf

    # Final decision
    if fusion_score >= FUSION_THRESHOLD and anchor_user:
        return {
            "status": "success",
            "username": anchor_user,
            "fusion_score": round(fusion_score, 3),
            "details": {
                "face": face_conf,
                "voice": voice_conf,
                "gesture": gesture_conf
            }
        }


    return {
        "status": "failure",
        "fusion_score": round(fusion_score, 3),
        "details": {
            "face": face_conf,
            "voice": voice_conf,
            "gesture": gesture_conf
        }
    }


if __name__ == "__main__":
    #results = orchestrate_verification()
    results = orchestrate_parallel()

    result_map = {res["modality"]: res for res in results}

    face_res = result_map.get("face", {})
    voice_res = result_map.get("voice", {})
    gesture_res = result_map.get("gesture", {})

    fused_result = fuse_results(face_res, voice_res, gesture_res)
    print("\nFUSION RESULT:")
    print(fused_result)