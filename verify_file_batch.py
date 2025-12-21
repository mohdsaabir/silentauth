import numpy as np
import torch
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav
from db_utils import fetch_all_embeddings

# ---------------- CONFIG ----------------
VERIFY_DIR = Path("data/verify")
THRESHOLD = 0.78
# ---------------------------------------

encoder = VoiceEncoder()

def l2_normalize(x, eps=1e-10):
    return x / (np.linalg.norm(x) + eps)

def verify_all():
    verify_files = sorted(VERIFY_DIR.glob("*.wav"))
    enrolled = fetch_all_embeddings()  # returns list of tuples (user_name, embedding) from db_utils.py

    if not verify_files:
        print("No verification WAV files found!")
        return

    if not enrolled:
        print("No enrolled users found in the database!")
        return

    print("\n========= BATCH VERIFICATION =========\n")
    summary = []

    for wav_file in verify_files:
        print(f"\n▶ Verifying: {wav_file.name}")

        wav_processed = preprocess_wav(str(wav_file))
        test_emb = encoder.embed_utterance(wav_processed).flatten()
        test_emb = l2_normalize(test_emb)

        best_score = -1
        best_user = None

        for user_name, stored_emb in enrolled:
            stored_emb = stored_emb.flatten()
            stored_emb = l2_normalize(stored_emb)  # normalize DB embeddings

            if stored_emb.shape != test_emb.shape:
                print(f"   Skipping {user_name}: shape mismatch")
                continue

            score = np.dot(test_emb, stored_emb)
            print(f"   Similarity with {user_name}: {score*100:.2f}%")

            if score > best_score:
                best_score = score
                best_user = user_name

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
    # Just printing a final summary of all verifications not relevent to core logic
    # -------- FINAL SUMMARY --------
    print("\n=========== SUMMARY ===========")
    print(f"{'File':20} {'User':15} {'Score (%)':10} {'Result'}")
    print("-" * 55)
    for s in summary:
        print(
            f"{s['file']:20} {s['user']:15} {s['score']*100:10.2f} {s['decision']}"
        )
    print("\n================================\n")

if __name__ == "__main__":
    verify_all()
