# enroll_voice.py
import sounddevice as sd
import numpy as np
import torch
import sqlite3
import os


from speechbrain.inference import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy

# ================= CONFIG =================
SAMPLE_RATE = 16000
CLIP_SECONDS = 10
NUM_CLIPS = 5

# ================= DATABASE =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.environ.get(
    "SILENTAUTH_DB_PATH",
    os.path.abspath(os.path.join(BASE_DIR, "..", "database", "central.db"))
)

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
    print(f" Speak for {seconds} seconds...")
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


# ================= ENROLLMENT =================
def run_voice_enrollment(user_name):

    print(f"\nStarting voice enrollment for: {user_name}")

    embeddings = []

    for i in range(NUM_CLIPS):
        print(f"\nRecording clip {i+1}/{NUM_CLIPS}")
        audio = record_audio(CLIP_SECONDS)

        emb = get_embedding_from_signal(audio)
        embeddings.append(emb)

        print(f"Clip {i+1} embedding captured")

    if len(embeddings) == 0:
        print("No voice data captured. Enrollment failed.")
        return None

    final_embedding = torch.stack(embeddings).mean(dim=0)

    print("\nEnrollment completed")
    print("Final embedding shape:", final_embedding.shape)

    # ---------- Database operations ----------
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get user_id
    cursor.execute(
        "SELECT user_id FROM users WHERE name=?",
        (user_name,)
    )
    row = cursor.fetchone()

    if not row:
        print("User not found in DB")
        conn.close()
        return None

    user_id = row[0]

    # Insert voice embedding
    cursor.execute(
        "INSERT OR REPLACE INTO voice (user_id, voice_embedding) VALUES (?, ?)",
        (user_id, final_embedding.detach().numpy().astype("float32").tobytes())
    )

    conn.commit()
    conn.close()

    print(f"Voice enrollment successful → {user_name} (user_id={user_id})")

    return {"user_id": user_id}


'''

if __name__ == "__main__":
    test_user = "sabir"
    run_voice_enrollment(test_user)

'''