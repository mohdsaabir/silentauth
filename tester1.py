# tester.py

import csv
import os
from datetime import datetime

CSV_PATH = os.path.join(os.path.dirname(__file__), "multimodal_logs.csv")


def log_attempt(data: dict):
    """
    Logs a single multimodal authentication attempt to CSV.
    """

    file_exists = os.path.isfile(CSV_PATH)

    with open(CSV_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write header if file doesn't exist
        if not file_exists:
            writer.writerow([
            "timestamp",

            # Identity
            "identified_user",

            # Raw modality outputs
            "face_username",
            "face_conf_raw",
            "voice_username",
            "voice_conf_raw",
            "gesture",
            "gesture_userset",
            "gesture_conf_raw",

            # Effective scores
            "voice_conf_effective",
            "gesture_conf_effective",

            # Penalty
            "voice_penalty",
            "voice_penalty_applied",

            # Fusion
            "fusion_score",

            # Decision + evaluation
            "decision",
            "ground_truth",
            "attack_type",

            # Preset
            "preset",
            "preset_mismatch"
        ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            data.get("identified_user"),

            data.get("face_username"),
            data.get("face_conf_raw"),
            data.get("voice_username"),
            data.get("voice_conf_raw"),
            data.get("gesture"),
            data.get("gesture_userset"),
            data.get("gesture_conf_raw"),

            data.get("voice_conf_effective"),
            data.get("gesture_conf_effective"),

            data.get("voice_penalty"),
            data.get("voice_penalty_applied"),

            data.get("fusion_score"),

            data.get("decision"),
            data.get("ground_truth"),
            data.get("attack_type"),

            data.get("preset"),
            data.get("preset_mismatch"),
        ])


def evaluate_and_log(result: dict, ground_truth: str, attack_type: str):
    """
    Accepts fusion engine output,
    attaches evaluation metadata,
    and logs structured experiment data.
    """

    result_data = {
        "identified_user": result.get("identified_user"),

        # Raw modality outputs
        "face_username": result.get("face_user"),
        "face_conf_raw": result.get("face_score"),
        "voice_username": result.get("voice_user"),
        "voice_conf_raw": result.get("voice_score"),
        "gesture": result.get("gesture"),
        "gesture_userset": result.get("gesture_users"),
        "gesture_conf_raw": result.get("gesture_score"),

        # Effective scores after identity filtering
        "voice_conf_effective": result.get("voice_effective"),
        "gesture_conf_effective": result.get("gesture_effective"),

        # Penalty
        "voice_penalty": result.get("voice_penalty"),
        "voice_penalty_applied": result.get("penalty_applied"),

        # Fusion
        "fusion_score": result.get("fusion_score"),

        # Decision
        "decision": result.get("decision"),
        "ground_truth": ground_truth,
        "attack_type": attack_type,

        # Preset
        "preset": result.get("preset"),
        "preset_mismatch": result.get("preset_mismatch"),
    }

    log_attempt(result_data)