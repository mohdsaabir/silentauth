from multimodal_executor import orchestrate_parallel
from tester1 import evaluate_and_log

# ======================= CONFIG =======================
FUSION_THRESHOLD = 0.63
FACE_WEIGHT = 0.35
VOICE_WEIGHT = 0.35
GESTURE_WEIGHT = 0.30
CONTRADICTION_THRESHOLD = 0.65

# ===================== PRESET-BASED ADAPTIVE FUSION =====================

# Preset-based weights and thresholds
PRESET_CONFIG = {
    "normal": {
        "FACE_WEIGHT": 0.35,
        "VOICE_WEIGHT": 0.35,
        "GESTURE_WEIGHT": 0.30,
        "FUSION_THRESHOLD": 0.63
    },
    "voice_impaired": {  # skip voice, increase gesture weight
        "FACE_WEIGHT": 0.7,
        "VOICE_WEIGHT": 0.0,
        "GESTURE_WEIGHT": 0.3,
        "FUSION_THRESHOLD": 0.60
    },
    "motor_impaired": {  # skip gesture, increase voice weight
        "FACE_WEIGHT": 0.6,
        "VOICE_WEIGHT": 0.4,
        "GESTURE_WEIGHT": 0.0,
        "FUSION_THRESHOLD": 0.60
    }
}


def fuse_and_log(face_res, voice_res, gesture_res, ground_truth=None, attack_type=None):
    """
    Wrapper around fuse_results_with_preset
    that logs experiment data if ground_truth is provided.
    """

    fused = fuse_results_with_preset(face_res, voice_res, gesture_res)

    # Only log if running in evaluation mode
    if ground_truth is not None:
        result_for_logger = {
            "identified_user": fused.get("username", "Unknown"),
            "face_score": fused["details"].get("face", 0),
            "voice_score": fused["details"].get("voice", 0),
            "gesture_score": fused["details"].get("gesture", 0),
            "voice_effective": fused.get("voice_conf_effective",0),
            "gesture_effective": fused.get("gesture_conf_effective",0),
            "voice_penalty": fused.get("voice_penalty"),
            "penalty_applied" : fused.get("penalty_applied"),
            "fusion_score": fused.get("fusion_score", 0),
            "decision": fused.get("status"),
            "gesture": gesture_res.get("gesture"),
            "face_user": face_res.get("username"),
            "voice_user": fused.get("voice_username"),
            "gesture_users": gesture_res.get("username"),
            "preset_mismatch": fused.get("preset_mismatch", False),
            "preset": face_res.get("preset", "normal")
        }

        evaluate_and_log(
            result=result_for_logger,
            ground_truth=ground_truth,
            attack_type=attack_type
        )

    return fused

def fuse_results_with_preset(face_res, voice_res, gesture_res):
    """
    Adaptive fusion wrapper:
    Adjusts weights and threshold according to user's preset before calling core fuse_results.
    """
    preset = face_res.get("preset", "normal")  # read preset from face output
    config = PRESET_CONFIG.get(preset, PRESET_CONFIG["normal"])

    
    # Save original global weights/threshold
    global FACE_WEIGHT, VOICE_WEIGHT, GESTURE_WEIGHT, FUSION_THRESHOLD
    old_face, old_voice, old_gesture, old_thresh = FACE_WEIGHT, VOICE_WEIGHT, GESTURE_WEIGHT, FUSION_THRESHOLD

    # Temporarily override weights/threshold
    FACE_WEIGHT = config["FACE_WEIGHT"]
    VOICE_WEIGHT = config["VOICE_WEIGHT"]
    GESTURE_WEIGHT = config["GESTURE_WEIGHT"]
    FUSION_THRESHOLD = config["FUSION_THRESHOLD"]

    # Call original fuse_results
    fused = fuse_results(face_res, voice_res, gesture_res)


    # Restore original values
    FACE_WEIGHT, VOICE_WEIGHT, GESTURE_WEIGHT, FUSION_THRESHOLD = old_face, old_voice, old_gesture, old_thresh
    
    # Check preset match
    preset_user = face_res.get("username")
    preset_mismatch = False

    if fused["status"] == "success" and fused["username"] != preset_user:
        fused["status"] = "failure"
        preset_mismatch = True

    fused["preset_mismatch"] = preset_mismatch

    return fused





def fuse_results(face_res, voice_res, gesture_res):
    

    # Extract users from each modality
    face_user = face_res.get("username") if face_res.get("status") == "success" else None
    voice_user = voice_res.get("username") if voice_res.get("status") == "success" else None
    gesture_users = gesture_res.get("username", []) if gesture_res.get("status") == "success" else []

    
    # ==============================
    # FACE ANCHOR LOGIC
    # ==============================

    if not face_user:

        anchor_user = None
        face_conf = face_res.get("confidence", 0)

        # No reliable anchor → ignore other modalities
        voice_conf = 0
        gesture_conf = 0
        voice_penalty = 0
        penalty_applied = 0

    else:

        anchor_user = face_user
        face_conf = face_res.get("confidence", 0)

        # ==============================
        # GESTURE ANCHORING
        # ==============================

        if anchor_user in gesture_users:
            gesture_conf = gesture_res.get("confidence", 0)
        else:
            gesture_conf = 0

        # ==============================
        # VOICE LOGIC
        # ==============================

        voice_conf_raw = voice_res.get("confidence", 0)
        voice_penalty = 0
        penalty_applied = 0

        if voice_user == anchor_user:

            # consistent identity
            voice_conf = voice_conf_raw

        else:

            # contradiction → ignore score
            voice_conf = 0

            # apply penalty if contradiction confidence is high
            if voice_conf_raw >= CONTRADICTION_THRESHOLD:   # 0.65
                voice_penalty = VOICE_WEIGHT * voice_conf_raw
                penalty_applied = 1


    # ==============================
    # FUSION SCORE COMPUTATION
    # ==============================

    fusion_score = (
        FACE_WEIGHT * face_conf
        + VOICE_WEIGHT * voice_conf
        + GESTURE_WEIGHT * gesture_conf
        - voice_penalty
    )

    # Final decision
    if fusion_score >= FUSION_THRESHOLD and anchor_user:
        return {
            "status": "success",
            "username": anchor_user,
            "fusion_score": round(fusion_score, 3),
            "voice_username": voice_user,
            "gesture_userset": gesture_users,
            "voice_conf_effective": voice_conf,
            "gesture_conf_effective": gesture_conf,
            "voice_penalty": voice_penalty,
            "penalty_applied" : penalty_applied,
            "details": {
                "face": face_conf,
                "voice": voice_conf_raw,
                "gesture": gesture_res.get("confidence", 0),
                "gesture_name": gesture_res.get("gesture")
            }
        }


    return {
        "status": "failure",
        "username": "Unknown",
        "fusion_score": round(fusion_score, 3),
        "voice_username": voice_user,
        "gesture_userset": gesture_users,
        "voice_conf_effective": voice_conf,
        "gesture_conf_effective": gesture_conf,
        "voice_penalty": voice_penalty,
        "penalty_applied" : penalty_applied,
        "details": {
            "face": face_conf,
            "voice": voice_conf_raw,
            "gesture": gesture_res.get("confidence", 0),
            "gesture_name": gesture_res.get("gesture")
        }
    }



'''
# ===================== MAIN ==========================
if __name__ == "__main__":
    #results = orchestrate_verification()
    results = orchestrate_parallel()

    result_map = {res["modality"]: res for res in results}

    face_res = result_map.get("face", {})
    voice_res = result_map.get("voice", {})
    gesture_res = result_map.get("gesture", {})


    fused_result = fuse_results_with_preset(face_res, voice_res, gesture_res)
    print("\nFUSION RESULT:")
    print(fused_result)

'''


