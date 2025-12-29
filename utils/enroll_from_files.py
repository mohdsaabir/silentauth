import numpy as np
import torch
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav
from db_utils import insert_embedding

# ---------------- CONFIG ----------------
ENROLL_DIR = Path("data/enroll")
EMBED_DIR = Path("embeddings")
EMBED_DIR.mkdir(exist_ok=True)
# ---------------------------------------

encoder = VoiceEncoder()


def l2_normalize(x, eps=1e-10):
    return x / (np.linalg.norm(x) + eps)


def enroll_user(user_id):

    user_dir = ENROLL_DIR / user_id
    wav_files = list(user_dir.glob("*.wav"))

    if not wav_files:
        print("No WAV files found for enrollment!")
        return
    
    embeddings = []
    print(f"\n--- Enrolling user: {user_id} ---")

    for wav in wav_files:
        print(f"Processing {wav.name}")
        wav_processed = preprocess_wav(str(wav))
        emb = encoder.embed_utterance(wav_processed).flatten()
        embeddings.append(emb)

    # Average + normalize
    avg_emb = np.mean(embeddings, axis=0)
    avg_emb = l2_normalize(avg_emb)         

    emb_path = EMBED_DIR / f"{user_id}.pt"
    torch.save(torch.tensor(avg_emb, dtype=torch.float32), emb_path)
    insert_embedding(user_id, avg_emb)

    print(f"\nEnrollment complete for {user_id}")
    print(f"Embedding saved to {emb_path}\n")


if __name__ == "__main__":
    user_id = input("Enter user ID (folder name): ")
    enroll_user(user_id)
