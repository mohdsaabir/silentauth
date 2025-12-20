import numpy as np
import torch
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav

# ---------------- CONFIG ----------------
VERIFY_DIR = Path("data/verify")
EMBED_DIR = Path("embeddings")
THRESHOLD = 0.75
# ---------------------------------------

encoder = VoiceEncoder()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def verify_all():
    verify_files = list(VERIFY_DIR.glob("*.wav"))
    enrolled_files = list(EMBED_DIR.glob("*.pt"))

    if not verify_files:
        print("No verification WAV files found!")
        return

    if not enrolled_files:
        print("No enrolled users found!")
        return

    print("\n========= BATCH VERIFICATION =========\n")

    summary = []

    for wav_file in verify_files:
        print(f"\n▶ Verifying: {wav_file.name}")

        wav_processed = preprocess_wav(str(wav_file))
        test_emb = encoder.embed_utterance(wav_processed).flatten()

        best_score = -1
        best_user = None

        for emb_file in enrolled_files:
            user_id = emb_file.stem
            stored_emb = torch.load(emb_file).flatten().numpy()

            score = cosine_similarity(test_emb, stored_emb)
            print(f"   Similarity with {user_id}: {score*100:.2f}%")

            if score > best_score:
                best_score = score
                best_user = user_id

        decision = "ACCEPT" if best_score >= THRESHOLD else "REJECT"

        print(f"   ➜ Best match: {best_user}")
        print(f"   ➜ Score: {best_score*100:.2f}%")
        print(f"   ➜ Decision: {decision}")

        summary.append({
            "file": wav_file.name,
            "user": best_user,
            "score": best_score,
            "decision": decision
        })

    # -------- FINAL SUMMARY --------
    print("\n=========== SUMMARY ===========")
    print(f"{'File':20} {'User':15} {'Score (%)':10} {'Result'}")
    print("-" * 55)

    for s in summary:
        print(f"{s['file']:20} {s['user']:15} {s['score']*100:10.2f} {s['decision']}")

    print("\n================================\n")

if __name__ == "__main__":
    verify_all()
