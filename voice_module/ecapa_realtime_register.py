import sounddevice as sd
import numpy as np
import torch
import torchaudio
import json
import os
import sqlite3
from datetime import datetime

from speechbrain.inference import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy

# ================= CONFIG =================
SAMPLE_RATE = 16000
CLIP_SECONDS = 10
NUM_CLIPS = 5
DB_PATH = "voice_db.sqlite"
# ==========================================

# ---------- Load ECAPA ----------
print("Loading ECAPA model...")

model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    local_strategy=LocalStrategy.COPY
)

print("ECAPA loaded")

# ---------- DB INIT ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS voice_users (
            user_id TEXT PRIMARY KEY,
            voice_embedding TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_user_to_db(user_id, embedding):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO voice_users (user_id, voice_embedding, created_at)
        VALUES (?, ?, ?)
    """, (
        user_id,
        json.dumps(embedding.tolist()),
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()

# Initialize DB once
init_db()

# ---------- Helpers ----------
def record_audio(seconds):
    print(f"🎙️ Speak for {seconds} seconds...")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    return audio.squeeze()

# ---------- Embedding generation ----------

def get_embedding_from_signal(signal_np):
    signal = torch.tensor(signal_np).float()
    signal = signal.unsqueeze(0)  # [1, time]
    emb = model.encode_batch(signal)
    return emb.squeeze()  # [192]

# ---------- Registration ----------
user_id = input("\nEnter USER ID to register: ").strip()

print(f"\nStarting voice enrollment for: {user_id}")

embeddings = []

for i in range(NUM_CLIPS):
    print(f"\nRecording clip {i+1}/{NUM_CLIPS}")
    audio = record_audio(CLIP_SECONDS)

    emb = get_embedding_from_signal(audio)
    embeddings.append(emb)

    print(f"Clip {i+1} embedding captured")

# ---------- Average embeddings ----------
final_embedding = torch.stack(embeddings).mean(dim=0)

print("\nEnrollment completed")
print("Final embedding shape:", final_embedding.shape)

# ---------- Store in SQLite DB ----------
save_user_to_db(user_id, final_embedding)

print(f"\n✅ User '{user_id}' registered successfully")
print("Voice profile stored in SQLite DB")
