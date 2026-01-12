import sounddevice as sd
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import json
import os
import sqlite3
import json

from speechbrain.inference import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy

# ================= CONFIG =================
SAMPLE_RATE = 16000
RECORD_SECONDS = 10
THRESHOLD = 0.40
DB_PATH = "voice_db.sqlite"
# ==========================================

#--------OUTPUT---------------

api_output = {"modality" : "voice",
              "user_id" : "Nil",
              "confidence" : 0,
              "status" : "Nil"}

# ---------- Load ECAPA ----------
print("Loading ECAPA model...")

model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    local_strategy=LocalStrategy.COPY
)

print("ECAPA loaded")

# ---------- Helpers ----------
def record_audio(seconds):
    print(f"\n🎙️ Speak for {seconds} seconds...")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    return audio.squeeze()

def get_embedding_from_signal(signal_np):
    signal = torch.tensor(signal_np).float()
    signal = signal.unsqueeze(0)  # [1, time]
    emb = model.encode_batch(signal)
    return emb.squeeze()  # [192]

def load_users_from_db():
    if not os.path.exists(DB_PATH):
        raise RuntimeError("Voice DB not found")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT user_id, voice_embedding FROM voice_users")
    rows = cursor.fetchall()

    conn.close()

    if len(rows) == 0:
        raise RuntimeError("Voice DB is empty")

    users = {}
    for user_id, embedding_json in rows:
        users[user_id] = torch.tensor(json.loads(embedding_json))

    return users

# ---------- Load enrolled users ----------
enrolled_users = load_users_from_db()

print(f"Loaded {len(enrolled_users)} enrolled user(s)")

# ---------- Record verification audio ----------
audio = record_audio(RECORD_SECONDS)

verify_embedding = get_embedding_from_signal(audio)

# ---------- Compare against DB ----------
best_user = None
best_score = -1.0

for user_id, enrolled_embedding in enrolled_users.items():
    score = F.cosine_similarity(
        enrolled_embedding,
        verify_embedding,
        dim=0
    ).item()

    print(f"Similarity with {user_id}: {round(score, 3)}")

    if score > best_score:
        best_score = score
        best_user = user_id

# ---------- Decision ----------
#print("\nBest match:", best_user)
#print("Best similarity score:", round(best_score, 3))

print("\n\nResult\n")
if best_score >= THRESHOLD:
    api_output["user_id"] = best_user
    api_output["confidence"] = round(best_score, 3)
    api_output["status"] = "success"
    print(json.dumps(api_output))

else:
    api_output["user_id"] = "Unknown User"
    api_output["confidence"] = round(best_score, 3)
    api_output["status"] = "rejected"
    print(json.dumps(api_output))