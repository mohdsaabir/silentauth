import sounddevice as sd
import numpy as np
import torch
import torch.nn.functional as F
import sqlite3
import json
import os

from speechbrain.inference import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy

# ================= CONFIG =================
SAMPLE_RATE = 16000
RECORD_SECONDS = 10
THRESHOLD = 0.40

# ================= DATABASE =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.environ.get(
    "SILENTAUTH_DB_PATH",
    os.path.abspath(os.path.join(BASE_DIR, "..", "database", "central.db"))
)
# ==========================================

# ---------- Load ECAPA ----------
model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    local_strategy=LocalStrategy.COPY
)

# ---------- Helpers ----------
def record_audio(seconds):
    #print(f" Speak for {seconds} seconds...")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    return audio.squeeze()

def get_embedding_from_signal(signal_np):
    signal = torch.tensor(signal_np).float().unsqueeze(0)
    emb = model.encode_batch(signal)
    return emb.squeeze()

# ---------- Load users from CENTRAL DB ----------
def load_users_from_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT users.name, voice.voice_embedding
        FROM voice
        JOIN users ON users.user_id = voice.user_id
    """)

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        raise RuntimeError("No enrolled voice users found")

    users = {}
    for name, blob in rows:
        users[name] = torch.tensor(
            np.frombuffer(blob, dtype=np.float32)
        )

    return users


def run_voice_verification():
    api_output = {
        "modality": "voice",
        "username": "Nil",
        "confidence": 0,
        "status": "Nil"
    }

    enrolled_users = load_users_from_db()

    audio = record_audio(RECORD_SECONDS)
    verify_embedding = get_embedding_from_signal(audio)

    best_user = None
    best_score = -1.0

    for user_name, enrolled_embedding in enrolled_users.items():
        score = F.cosine_similarity(
            enrolled_embedding,
            verify_embedding,
            dim=0
        ).item()

        if score > best_score:
            best_score = score
            best_user = user_name

    if best_score >= THRESHOLD:
        api_output["username"] = best_user
        api_output["confidence"] = round(best_score, 3)
        api_output["status"] = "success"
    else:
        api_output["username"] = "Unknown"
        api_output["confidence"] = round(best_score, 3)
        api_output["status"] = "rejected"

    return api_output


"""
if __name__ == "__main__":
    result = run_voice_verification()
    print(result)
"""