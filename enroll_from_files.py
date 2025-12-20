import numpy as np
import torch
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav

# ---------------- CONFIG ----------------
ENROLL_DIR = Path("data/enroll")
EMBED_DIR = Path("embeddings")
EMBED_DIR.mkdir(exist_ok=True)
# ---------------------------------------

encoder = VoiceEncoder()

def enroll_user(user_id):
    user_dir = ENROLL_DIR / user_id
    wav_files = list(user_dir.glob("*.wav"))

    if len(wav_files) < 1:
        print("No WAV files found for enrollment!")
        return

    embeddings = []
    print(f"\n--- Enrolling user: {user_id} ---")

    for wav in wav_files:
        print(f"Processing {wav.name}")
        wav_processed = preprocess_wav(str(wav))
        emb = encoder.embed_utterance(wav_processed)
        embeddings.append(emb)

    avg_emb = np.mean(embeddings, axis=0).flatten()

    emb_path = EMBED_DIR / f"{user_id}.pt"
    torch.save(torch.tensor(avg_emb), emb_path)

    print(f"\nEnrollment complete for {user_id}")
    print(f"Embedding saved to {emb_path}\n")

if __name__ == "__main__":
    user_id = input("Enter user ID (folder name): ")
    enroll_user(user_id)
