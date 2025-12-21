import numpy as np
import torch
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav

# ---------------- CONFIG ----------------
VERIFY_AUDIO = Path("data/verify/test.wav")
EMBED_DIR = Path("embeddings")
THRESHOLD = 0.78
# ---------------------------------------

encoder = VoiceEncoder()

def verify():
    if not VERIFY_AUDIO.exists():
        print("Verification audio not found!")
        return

    enrolled_files = list(EMBED_DIR.glob("*.pt"))
    if not enrolled_files:
        print("No enrolled users found!")
        return

    print("\n--- Verification ---")
    print(f"Test file: {VERIFY_AUDIO.name}")

    wav_processed = preprocess_wav(str(VERIFY_AUDIO))
    test_emb = encoder.embed_utterance(wav_processed).flatten()

    best_score = -1
    best_user = None

    for file in enrolled_files:
        user_id = file.stem
        stored_emb = torch.load(file).flatten().numpy()

        score = np.dot(test_emb, stored_emb) / (
            np.linalg.norm(test_emb) * np.linalg.norm(stored_emb)
        )

        print(f"Similarity with {user_id}: {score*100:.2f}%")

        if score > best_score:
            best_score = score
            best_user = user_id

    print("\n--- RESULT ---")
    print(f"Best match: {best_user}")
    print(f"Score: {best_score*100:.2f}%")

    if best_score >= THRESHOLD:
        print("✅ AUTHENTICATED")
    else:
        print("❌ REJECTED")

if __name__ == "__main__":
    verify()
