import numpy as np
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav
import sounddevice as sd
import scipy.io.wavfile as wavfile

# These are the modules we created
from db_utils import fetch_all_embeddings 

# -------------------- CONFIG --------------------
SAMPLE_RATE = 16000
DURATION = 10
THRESHOLD = 0.78
AUDIO_DIR = Path("enrolled_audio")
AUDIO_DIR.mkdir(exist_ok=True)
# ------------------------------------------------

encoder = VoiceEncoder()


def l2_normalize(x, eps=1e-10):
    return x / (np.linalg.norm(x) + eps)



def verify_user(verify_path):
    enrolled = fetch_all_embeddings()  # returns list of (user_name, embedding) from db_utils.py
    if not enrolled:
        print("No enrolled users found in the database!")
        return

    wav_processed = preprocess_wav(str(verify_path))
    test_emb = encoder.embed_utterance(wav_processed).flatten()
    test_emb = l2_normalize(test_emb)

    best_score = -1
    best_user = None
    keyword= None
    for user_name, stored_emb, keywords in enrolled:
        stored_emb = l2_normalize(stored_emb)  # normalize db embeddings
        if stored_emb.shape != test_emb.shape:
            print(f"Skipping {user_name}: shape mismatch {stored_emb.shape}")
            continue

        score = np.dot(test_emb, stored_emb)  # cosine similarity

        # print(f"Similarity with {user_name}: {score*100:.2f}%")

        if score > best_score:
            best_score = score
            best_user = user_name
            keyword = keywords

    #print("\n--- RESULT ---")
    #print(f"Best match: {best_user}")
    #print(f"Score: {best_score*100:.2f}%")
    best_score_rounded = round(best_score * 100, 2)
    result = {"modality":"Voice", "user_name": best_user, "similarity_score": best_score_rounded}
    if best_score >= THRESHOLD:
        #print("✅ AUTHENTICATED")
        #print(f"Keyword: {keyword}")
        result["status"] = "Sucess"
        return keyword, result
    else:
        #print("❌ REJECTED")
        result["status"] = "Rejected"
        result["user_name"] = None
        result["similarity_score"] = 0
        return False, result


if __name__ == "__main__":
    verify_user()
